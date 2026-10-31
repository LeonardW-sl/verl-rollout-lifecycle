# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import getpass
import logging
import os
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from torch.distributed.device_mesh import DeviceMesh
from vllm.config import LoRAConfig

from verl.utils.ray_utils import get_event_loop

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


# Guard so the class-level patch is installed at most once per worker process.
_KV_PROFILING_BREAKDOWN_PATCHED = False


def _install_kv_profiling_breakdown_patch() -> None:
    """Decompose vLLM's KV-cache profiling (``determine_available_memory``).

    ``rpc_determine_available_memory`` is a round-trip wall-clock call from the
    EngineCore side whose internal split is otherwise invisible. It splits into
    two parts (verified against v0.12.0 ``vllm/v1/worker/gpu_worker.py`` line 282):

        determine_available_memory
          ├── model_runner.profile_run()   # dummy forward -- CANNOT be skipped
          └── empty_cache + memory_profiling snapshot + KV size math  # skippable

    The skippable half is exactly what ``kv_cache_memory_bytes`` would let us
    bypass (that branch still calls ``profile_run`` -- see the same source). So
    ``profile_run``'s share of the call is the *complement* of that bypass's
    savings ceiling: small profile_run => large potential saving.

    This patch is measurement-only (pure spans, no behavior change) and lives in
    the worker process, where ``VERL_RESIZE_TRACE_PATH`` is now propagated
    (see ray_trainer ``_get_resize_worker_group_kwargs``). It is a no-op unless
    that env is set. Patch strategy = runtime monkey-patch.
    """
    global _KV_PROFILING_BREAKDOWN_PATCHED
    if _KV_PROFILING_BREAKDOWN_PATCHED:
        return

    from verl.experimental.one_step_off_policy.trace_utils import (
        resize_trace_config_from_env,
        resize_trace_span,
    )

    trace_cfg = resize_trace_config_from_env()
    if not trace_cfg.enabled:
        # No trace path in this worker process -> nothing to measure. Leave the
        # class untouched so the whole thing is byte-for-byte inert by default.
        return

    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as exc:  # pragma: no cover - version/layout guard
        logger.warning("[kv-profiling-breakdown] cannot import vLLM Worker, skip patch: %s", exc)
        return

    if getattr(Worker, "_verl_kv_profiling_patched", False):
        _KV_PROFILING_BREAKDOWN_PATCHED = True
        return

    original_determine = Worker.determine_available_memory

    def _derive_worker_identity(worker) -> dict[str, Any]:
        # P2: a rollout worker group is created in one shot but its workers span
        # multiple replicas, so a single injected constant rank is wrong for all
        # but one worker. Instead derive identity from what each worker process
        # actually knows: RANK (set per-worker by RayWorkerGroup._create_worker)
        # and the engine's tensor_parallel_size. replica_rank = RANK // tp_size.
        # We emit BOTH the raw fields and the derived replica_rank so the
        # analysis side can reconcile per replica (group by replica_rank, take
        # max across that replica's workers -- NOT sum -- vs the RPC wall clock).
        try:
            worker_rank = int(os.environ["RANK"])
        except (KeyError, ValueError):
            worker_rank = None
        tp_size = None
        try:
            tp_size = int(worker.vllm_config.parallel_config.tensor_parallel_size)
        except Exception:  # noqa: BLE001 - config layout guard across vLLM versions
            tp_size = None
        replica_rank = worker_rank // tp_size if (worker_rank is not None and tp_size) else None
        return {"worker_rank": worker_rank, "tp_size": tp_size, "replica_rank": replica_rank}

    def determine_available_memory(self):
        cfg = resize_trace_config_from_env()
        pid = os.getpid()
        ident = _derive_worker_identity(self)
        model_runner = getattr(self, "model_runner", None)
        original_profile_run = getattr(model_runner, "profile_run", None) if model_runner is not None else None
        # P3: record whether profile_run already lived in the instance __dict__.
        # If it was inherited from the class (the normal case), restoring via
        # assignment would leave a permanent instance-level shadow; we delattr
        # instead so the instance is byte-for-byte as we found it.
        had_own_profile_run = (
            model_runner is not None and "profile_run" in vars(model_runner) if model_runner is not None else False
        )

        # Inner span: wrap profile_run in-place for the duration of this call so
        # its wall clock is captured separately. Restored in finally regardless.
        if original_profile_run is not None:

            def _traced_profile_run(*args, **kwargs):
                with resize_trace_span(
                    cfg,
                    "kv_profile_run",
                    step=None,
                    lane=f"rollout_worker_pid{pid}",
                    metadata=ident,
                ):
                    return original_profile_run(*args, **kwargs)

            model_runner.profile_run = _traced_profile_run

        # Outer span: whole determine_available_memory. Should reconcile with
        # M1's rpc_determine_available_memory (minus ZMQ round-trip overhead),
        # per replica (group by replica_rank, take max across the replica).
        try:
            with resize_trace_span(
                cfg,
                "kv_determine_available_memory",
                step=None,
                lane=f"rollout_worker_pid{pid}",
                metadata=ident,
            ):
                return original_determine(self)
        finally:
            if original_profile_run is not None:
                if had_own_profile_run:
                    model_runner.profile_run = original_profile_run
                else:
                    # Was inherited from the class; remove our instance shadow.
                    try:
                        del model_runner.profile_run
                    except AttributeError:
                        model_runner.profile_run = original_profile_run

    Worker.determine_available_memory = determine_available_memory
    Worker._verl_kv_profiling_patched = True
    _KV_PROFILING_BREAKDOWN_PATCHED = True
    logger.info("[kv-profiling-breakdown] patched vLLM Worker.determine_available_memory (pid=%s)", os.getpid())


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = get_event_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _build_inference_engine(self) -> WorkerWrapperBase:
        """Create a vLLM worker wrapper across vLLM versions.

        vLLM changed WorkerWrapperBase signature (i.e., removing vllm_config from
        __init__). We keep a small runtime fallback to support multiple versions.

        https://github.com/vllm-project/vllm/commit/aafd4d23548ae54adeca1d4898cc15a4d2c390ac
        """
        try:
            return WorkerWrapperBase(vllm_config=self.vllm_config)
        except TypeError:
            return WorkerWrapperBase()

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        worker_rank = int(os.environ["RANK"])
        all_kwargs[0]["rank"] = worker_rank
        device_name = "NPU" if is_npu_available else "GPU"
        if ray_noset_visible_devices():
            local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        else:
            # Ray normally gives each worker a single-card CUDA_VISIBLE_DEVICES, so
            # local_rank=0 is the only valid index. The sleep pool's elastic zone
            # breaks that assumption: its groups hold zero Ray GPU quota
            # (use_gpu=False) and receive a group-wide CVD listing all TP cards, so
            # every worker in the group would bind cuda:0 and collide, whereas one
            # card per CVD separates them cleanly.
            #
            # Derive the index from RANK whenever more than one card is visible.
            # Single-card workers keep local_rank=0 exactly as before, so the
            # standard path is unchanged.
            visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
            local_rank = worker_rank % visible if visible > 1 else 0
        all_kwargs[0]["local_rank"] = local_rank
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if self.config.quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(
                    f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {self.config.quantization}"
                )

            if self.config.quantization == "fp8":
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()

        self.inference_engine = self._build_inference_engine()
        self.inference_engine.init_worker(all_kwargs)

        # Install the KV-profiling breakdown patch now, before any RPC triggers
        # determine_available_memory. No-op unless VERL_RESIZE_TRACE_PATH is set
        # in this worker process (propagated via resize worker_env).
        _install_kv_profiling_breakdown_patch()

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model_runner = self.inference_engine.worker.model_runner
            model = model_runner.model
            patch_vllm_moe_model_weight_loader(model)

            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: vLLMAsyncRollout uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "vLLMAsyncRollout does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and AsyncLLMServerManager, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/volcengine/verl/issues/4682 for more details."
        )

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
