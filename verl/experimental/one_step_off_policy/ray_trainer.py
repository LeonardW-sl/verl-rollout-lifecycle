# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
This trainer supports model-agonistic model initialization with huggingface
"""

import asyncio
import contextlib
import inspect
import logging
import math
import os
import shutil
import socket
import time
import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.experimental.state.api import get_actor as ray_get_actor_state
from ray.util.collective import collective
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.one_step_off_policy.awake_topology import AwakeTopology, ZoneLayout, diff_engines
from verl.experimental.one_step_off_policy.elastic_engine_pool import ElasticEnginePool
from verl.experimental.one_step_off_policy.communicator_cache import (
    CommunicatorCacheConfig,
    WeightSyncCommunicatorCache,
    build_topology_key,
)
from verl.experimental.one_step_off_policy.resize_budget import (
    ResizeBudgetConfig,
    ResizeBudgetController,
    ResizeBudgetSnapshot,
)
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.experimental.one_step_off_policy.resize_controller import (
    ACTION_EXPAND_ROLLOUT,
    ACTION_EXPAND_TRAIN,
    ACTION_HOLD,
    ACTION_TO_CODE,
    ResizeController,
    ResizeControllerConfig,
)
from verl.experimental.one_step_off_policy.resize_metrics import build_resize_observation
from verl.experimental.one_step_off_policy.staging_backend import HostStagingConfig, has_restore_session_manifest, read_restore_session_manifest
from verl.experimental.one_step_off_policy.trace_utils import build_resize_trace_config, resize_trace_span
from verl.experimental.one_step_off_policy.utils import need_critic
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import SubRayResourcePool, create_colocated_worker_cls, split_resource_pool
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    calculate_workload,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils import omega_conf_to_dataclass
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import ValidationGenerationsLogger


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _scope_elastic_worker_env(worker_env: dict | None, rank: int) -> dict | None:
    """Narrow a TP-group CVD to one physical token before actor startup."""
    if not worker_env:
        return worker_env
    visible = [device.strip() for device in worker_env.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()]
    if len(visible) <= 1:
        return worker_env
    scoped_env = dict(worker_env)
    scoped_env["CUDA_VISIBLE_DEVICES"] = visible[int(rank) % len(visible)]
    return scoped_env


class ElasticPoolRayWorkerGroup(RayWorkerGroup):
    """Create zero-quota elastic actors with one logical CUDA device each."""

    def _create_worker(
        self,
        rank,
        pg_idx,
        pg,
        local_rank,
        resource_pool,
        ray_cls_with_init,
        worker_env,
        detached,
    ):
        return super()._create_worker(
            rank=rank,
            pg_idx=pg_idx,
            pg=pg,
            local_rank=local_rank,
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            worker_env=_scope_elastic_worker_env(worker_env, rank),
            detached=detached,
        )


class OneStepOffRayTrainer(RayPPOTrainer):
    # Staged resize execution model for shared-pool in-place rebalance:
    #
    # Layer 1: static prepare
    #   - parse resize spec / split pools / build plans
    #   - do not consume new shared-pool worker slots yet
    #
    # Layer 2: resource transition
    #   - release the shrinking side first (e.g. old rollout 4 -> 2 releases 2 slots)
    #   - create/rebuild the new actor on the freed slots

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for one-step-off worker groups.

        Keep rollout profiling separate from actor profiling and avoid passing the
        PPO trainer's `role` kwarg into torch profiler implementations.
        """
        if do_profile:
            self.actor_wg.start_profile(profile_step=self.global_steps)
            if not self.hybrid_engine:
                self.rollout_wg.start_profile(profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for one-step-off worker groups."""
        if do_profile:
            self.actor_wg.stop_profile()
            if not self.hybrid_engine:
                self.rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()
    #   - release the other old side if needed, then create the new rollout
    #
    # Layer 3: final commit/publish
    #   - commit new worker/model init
    #   - restore/copy weights
    #   - publish new actor/rollout pair and rebuild dependent managers/groups
    #
    # For shared-pool in-place rebalance, trying to build the full new topology
    # before shrinking the old one can over-allocate the pool and leave new
    # workers stuck in PENDING_CREATION.
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def _trace_step(self) -> int:
        return int(getattr(self, "global_steps", 0))

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger()
        # Track active worker groups per role for dynamic switching.
        self.role_groups: dict[Role, dict[str, RayWorkerGroup]] = {
            Role.Actor: {},
            Role.Rollout: {},
        }
        self._active_role_world_sizes: dict[Role, int] = {}
        self._weight_sync_group_version = 0
        self._active_weight_sync_group_name = "actor_rollout"
        self._pending_weight_sync_group_cleanup: list[tuple[str, RayWorkerGroup, RayWorkerGroup]] = []
        self._last_runtime_batch_plan: dict[str, int] | None = None
        self._resize_padding_history: list[DataProto] = []
        self._resize_padding_history_size = 8
        self._active_topology_specs: dict[Role, dict | None] = {Role.Actor: None, Role.Rollout: None}
        # Round one: schedule is still the source of candidate topologies.
        # The controller only decides whether a scheduled switch should fire.
        self._dynamic_resize_mode = self._resolve_dynamic_resize_mode()
        self._resize_controller = self._build_resize_controller()
        self._latest_resize_control_metrics: dict[str, float | str | int] = self._default_resize_control_metrics()
        self._resize_trace_config = build_resize_trace_config(self.config)
        self._dynamic_resize_enabled = self._build_dynamic_resize_flag("enable", default=False)
        self._dynamic_resize_phased_init_enabled = self._build_dynamic_resize_flag("phased_init.enable", default=True)
        self._dynamic_resize_async_comm_prewarm_enabled = self._build_dynamic_resize_flag(
            "async_comm_prewarm.enable", default=True
        )
        # Phase 5: reuse rollout worker processes whose GPU bundle set is unchanged
        # across a resize (survivor engines), instead of killing+respawning them.
        # Default false keeps the legacy full-rebuild path byte-for-byte.
        self._dynamic_resize_worker_reuse_enabled = self._build_dynamic_resize_flag(
            "worker_reuse.enable", default=False
        )
        # Extend worker reuse to the MIXED path (survivors coexist with newly
        # built engines), i.e. expansion. The pure-shrink gate below
        # (to_build == 0) stays the default; enabling this flag lets survivors be
        # kept even when some new engines must be built, which requires the
        # (a)/(b)/(c) fixes: merged group ordered by new_rank, delta workers built
        # for the to_build bundles, and num_replicas derived from the new topology.
        self._dynamic_resize_worker_reuse_mixed_enabled = self._build_dynamic_resize_flag(
            "worker_reuse.allow_mixed", default=False
        )
        # Keep elastic rollout engines alive but asleep across a resize instead of
        # killing and cold-rebuilding them. Default false keeps the
        # survivor/full-rebuild paths byte-for-byte.
        self._sleep_pool_enabled = self._build_dynamic_resize_flag("sleep_pool.enable", default=False)
        self._sleep_pool_layout = self._build_sleep_pool_layout()
        self._elastic_pool = None
        self._memory_transition_counter = 0
        self._pending_memory_transition: dict | None = None
        print(
            f"[INIT] worker_reuse_enabled={self._dynamic_resize_worker_reuse_enabled} "
            f"allow_mixed={self._dynamic_resize_worker_reuse_mixed_enabled} "
            f"sleep_pool={self._sleep_pool_enabled} layout={self._sleep_pool_layout}",
            flush=True,
        )
        # Round two: handoff is configured through a staging backend abstraction
        # so the staged resize path can evolve from disk-backed transfer toward
        # richer host-memory implementations without changing the trainer flow.
        self._handoff_staging_config = self._build_handoff_staging_config()
        self._latest_resize_execution_metrics: dict[str, float | str | int] = self._default_resize_execution_metrics()
        self._communicator_cache_config = self._build_communicator_cache_config()
        self._weight_sync_communicator_cache = WeightSyncCommunicatorCache(self._communicator_cache_config)
        self._latest_communicator_cache_metrics: dict[str, float | str | int] = self._default_communicator_cache_metrics()
        self._resize_budget_config = self._build_resize_budget_config()
        self._resize_budget_controller = ResizeBudgetController(self._resize_budget_config)
        self._latest_resize_budget_metrics: dict[str, float | str | int] = self._default_resize_budget_metrics()

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = lora_rank > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _resolve_dynamic_resize_mode(self) -> str:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return "schedule"
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return str(cfg.get("mode", "schedule"))

    def _build_resize_controller(self) -> ResizeController | None:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return None
        cfg = OmegaConf.to_container(cfg, resolve=True)
        mode = str(cfg.get("mode", "schedule"))
        if mode != "schedule_with_hysteresis":
            return None
        controller_cfg = ResizeControllerConfig.from_dict(cfg.get("hysteresis", {}))
        if not controller_cfg.enable:
            return None
        return ResizeController(controller_cfg)

    def _build_handoff_staging_config(self) -> HostStagingConfig:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return HostStagingConfig()
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return HostStagingConfig.from_dict(cfg.get("handoff", {}))

    def _build_communicator_cache_config(self) -> CommunicatorCacheConfig:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return CommunicatorCacheConfig()
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return CommunicatorCacheConfig.from_dict(cfg.get("communicator_cache", {}))

    def _build_dynamic_resize_flag(self, key: str, *, default: bool) -> bool:
        value = OmegaConf.select(self.config.trainer, f"dynamic_resize.{key}")
        if value is None:
            return default
        return bool(value)

    def _build_dynamic_resize_int(self, key: str, *, default: int) -> int:
        """Integer sibling of _build_dynamic_resize_flag.

        Deliberately separate: the flag helper ends in a bool cast, which would
        silently turn R=2 into True.
        """
        value = OmegaConf.select(self.config.trainer, f"dynamic_resize.{key}")
        if value is None:
            return default
        return int(value)

    def _build_sleep_pool_layout(self) -> ZoneLayout | None:
        """Resolve the A/R/E zone geometry, or None when the pool is disabled.

        N comes from the existing shared_pool card count, so the only genuinely
        new deployment parameters are R and E; the schedule keeps being derived
        from FIRST/SECOND_SPLIT_PLAN via ZoneLayout.schedule_from_rollout_cards.

        SINGLE NODE ONLY. Despite its name, shared_pool.n_gpus_per_node is read
        here as the cluster-wide card count. That is correct only because every
        one_step_off script sets nnodes=1 (phase6_overlap_off.sh passes NUM_GPUS
        for both). On multiple nodes this under-counts N by a factor of nnodes,
        and the zone split would land on the wrong cards instead of failing
        loudly. Multiply by nnodes before trusting this multi-node.
        """
        if not self._sleep_pool_enabled:
            return None
        total_gpus = self._build_dynamic_resize_int("shared_pool.n_gpus_per_node", default=0)
        permanent_rollout_gpus = self._build_dynamic_resize_int("sleep_pool.permanent_rollout_gpus", default=0)
        elastic_gpus = self._build_dynamic_resize_int("sleep_pool.elastic_gpus", default=0)
        if elastic_gpus <= 0:
            raise ValueError(
                "trainer.dynamic_resize.sleep_pool.enable=true requires elastic_gpus > 0; "
                "an empty elastic zone can never sleep anything, so the run would look like a "
                "passing sleep-pool experiment while doing nothing"
            )
        tp = int(self.config.actor_rollout_ref.rollout.tensor_model_parallel_size)
        # ZoneLayout validates divisibility and oversubscription and raises on
        # violation -- no silent fallback to a full rebuild.
        return ZoneLayout(
            total_gpus=total_gpus,
            permanent_rollout_gpus=permanent_rollout_gpus,
            elastic_gpus=elastic_gpus,
            tp=tp,
        )

    def _build_resize_budget_config(self) -> ResizeBudgetConfig:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return ResizeBudgetConfig()
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return ResizeBudgetConfig.from_dict(cfg.get("budget_protection", {}))

    def _build_step_path_debug_metrics(self, timing_raw: dict[str, float]) -> dict[str, float | int]:
        rollout_background_s = float(timing_raw.get("generate_async", 0.0) or 0.0)
        rollout_wait_exposed_s = float(timing_raw.get("gen", 0.0) or 0.0)
        sync_s = float(timing_raw.get("sync_rollout_weights", 0.0) or 0.0)
        reward_s = float(timing_raw.get("reward", 0.0) or 0.0)
        adv_s = float(timing_raw.get("adv", 0.0) or 0.0)
        old_log_prob_s = float(timing_raw.get("old_log_prob", 0.0) or 0.0)
        ref_s = float(timing_raw.get(str(Role.RefPolicy), 0.0) or 0.0)
        values_s = float(timing_raw.get("values", 0.0) or 0.0)
        update_actor_s = float(timing_raw.get("update_actor", 0.0) or 0.0)
        update_critic_s = float(timing_raw.get("update_critic", 0.0) or 0.0)
        step_s = float(timing_raw.get("step", 0.0) or 0.0)

        train_update_s = update_actor_s + update_critic_s
        train_post_wait_s = sync_s + reward_s + adv_s + old_log_prob_s + ref_s + values_s + train_update_s
        rollout_hidden_by_overlap_s = max(rollout_background_s - rollout_wait_exposed_s, 0.0)

        # For async off-policy execution, rollout and train happen on different ranks and
        # overlap in wall-clock time. A per-step system time should therefore be compared
        # against the longest rank path rather than a train+rollout sum.
        step_driver_path_s = rollout_wait_exposed_s + train_post_wait_s
        step_reconstructed_s = max(rollout_background_s, train_post_wait_s)

        actor_world_size = len(getattr(getattr(self, "actor_wg", None), "workers", []) or [])
        rollout_world_size = len(getattr(getattr(self, "rollout_wg", None), "workers", []) or [])
        agent_loop_workers = len(getattr(getattr(self, "async_rollout_manager", None), "agent_loop_workers", []) or [])

        return {
            "debug/actor_world_size": actor_world_size,
            "debug/rollout_world_size": rollout_world_size,
            "debug/agent_loop_worker_count": agent_loop_workers,
            "debug/timing_rollout_background_total_s": rollout_background_s,
            "debug/timing_rollout_wait_exposed_s": rollout_wait_exposed_s,
            "debug/timing_rollout_hidden_by_overlap_s": rollout_hidden_by_overlap_s,
            "debug/timing_train_update_total_s": train_update_s,
            "debug/timing_train_post_wait_total_s": train_post_wait_s,
            "debug/timing_step_driver_path_s": step_driver_path_s,
            "debug/timing_step_reconstructed_s": step_reconstructed_s,
            "debug/timing_step_reconstruction_error_s": step_s - step_reconstructed_s,
        }

    def _default_resize_control_metrics(self) -> dict[str, float | str | int]:
        enabled = 1.0 if self._resize_controller is not None else 0.0
        return {
            "resize/mode": self._dynamic_resize_mode,
            "resize/controller_enabled": enabled,
            "resize/rollout_train_ratio": 0.0,
            "resize/hysteresis_signal": ACTION_HOLD,
            "resize/hysteresis_signal_code": 0.0,
            "resize/hysteresis_decision": ACTION_HOLD,
            "resize/hysteresis_decision_code": 0.0,
            "resize/required_action": ACTION_HOLD,
            "resize/required_action_code": 0.0,
            "resize/window_fill": 0,
            "resize/dwell_remaining": 0,
            "resize/cooldown_remaining": 0,
            "resize/gate_pass": -1.0,
        }

    def _default_resize_execution_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/host_stage_enabled": 1.0 if self._handoff_staging_config.enable else 0.0,
            "resize/phased_init_enabled": 1.0 if self._dynamic_resize_phased_init_enabled else 0.0,
            "resize/host_stage_backend": self._handoff_staging_config.effective_backend(),
            "resize/host_stage_requested_backend": self._handoff_staging_config.backend,
            "resize/host_stage_export_s": 0.0,
            "resize/host_stage_export_initial_barrier_s": 0.0,
            "resize/host_stage_export_load_model_to_gpu_s": 0.0,
            "resize/host_stage_export_load_optimizer_to_gpu_s": 0.0,
            "resize/host_stage_export_sanitize_optimizer_s": 0.0,
            "resize/host_stage_export_model_state_dict_s": 0.0,
            "resize/host_stage_export_optimizer_state_dict_s": 0.0,
            "resize/host_stage_export_build_model_pages_s": 0.0,
            "resize/host_stage_export_build_optimizer_pages_s": 0.0,
            "resize/host_stage_export_create_session_s": 0.0,
            "resize/host_stage_export_stage_model_pages_s": 0.0,
            "resize/host_stage_export_stage_optimizer_pages_s": 0.0,
            "resize/host_stage_export_host_put_s": 0.0,
            "resize/host_stage_export_host_file_write_s": 0.0,
            "resize/host_stage_export_disk_spill_s": 0.0,
            "resize/host_stage_export_extra_state_s": 0.0,
            "resize/host_stage_export_manifest_update_s": 0.0,
            "resize/host_stage_export_final_barrier_s": 0.0,
            "resize/host_stage_export_offload_s": 0.0,
            "resize/host_stage_export_model_page_count": 0.0,
            "resize/host_stage_export_optimizer_page_count": 0.0,
            "resize/host_stage_export_model_state_host_pages": 0.0,
            "resize/host_stage_export_model_state_disk_pages": 0.0,
            "resize/host_stage_export_model_state_host_bytes": 0.0,
            "resize/host_stage_export_model_state_disk_bytes": 0.0,
            "resize/host_stage_export_optim_state_host_pages": 0.0,
            "resize/host_stage_export_optim_state_disk_pages": 0.0,
            "resize/host_stage_export_optim_state_host_bytes": 0.0,
            "resize/host_stage_export_optim_state_disk_bytes": 0.0,
            "resize/host_stage_import_s": 0.0,
            "resize/host_stage_import_wait_s": 0.0,
            "resize/post_switch_weight_sync_s": 0.0,
            "resize/post_switch_kv_clear_s": 0.0,
            "resize/progressive_swap_s": 0.0,
            "resize/kv_cache_preclear_s": 0.0,
            "resize/host_stage_cleanup": 0.0,
            "resize/optimizer_deferred_restore": 0.0,
            "resize/optimizer_pending_pages": 0.0,
            "resize/optimizer_materialize_s": 0.0,
            "resize/optimizer_load_pages_s": 0.0,
            "resize/optimizer_set_state_dict_s": 0.0,
            "resize/optimizer_streaming_restore": 0.0,
            "resize/optimizer_streamed_pages": 0.0,
            "resize/optimizer_full_state_restore": 0.0,
            "resize/optimizer_materialize_count": 0.0,
            "resize/model_restore_device_sync_s": 0.0,
            "resize/model_device_preload_pages": 0.0,
            "resize/model_device_preload_bytes": 0.0,
            "resize/model_device_preload_max_pending_pages": 0.0,
            "resize/model_device_preload_max_pending_bytes": 0.0,
            "resize/optimizer_device_preload_pages": 0.0,
            "resize/optimizer_device_preload_bytes": 0.0,
            "resize/optimizer_device_preload_max_pending_pages": 0.0,
            "resize/optimizer_device_preload_max_pending_bytes": 0.0,
            "resize/actor_prepare_worker_init_s": 0.0,
            "resize/actor_commit_worker_init_s": 0.0,
            "resize/actor_prepare_model_init_s": 0.0,
            "resize/actor_commit_model_init_s": 0.0,
            "resize/rollout_prepare_worker_init_s": 0.0,
            "resize/rollout_commit_worker_init_s": 0.0,
            "resize/rollout_prepare_model_init_s": 0.0,
            "resize/rollout_commit_model_init_s": 0.0,
            "resize/restore_failed": 0.0,
            "resize/partial_restore_cleanup_count": 0.0,
            "pool/prebuild_s": 0.0,
        }

    def _default_communicator_cache_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/comm_cache_enabled": 1.0 if self._communicator_cache_config.enable else 0.0,
            "resize/async_comm_prewarm_enabled": 1.0 if self._dynamic_resize_async_comm_prewarm_enabled else 0.0,
            "resize/comm_cache_hit": 0.0,
            "resize/comm_cache_miss": 0.0,
            "resize/comm_live_cache_hit": 0.0,
            "resize/comm_registry_hit": 0.0,
            "resize/comm_registry_miss": 0.0,
            "resize/comm_prewarm_ready": 0.0,
            "resize/comm_prewarm_create_s": 0.0,
            "resize/comm_prewarm_warmup_s": 0.0,
            "resize/comm_prewarm_warmup_broadcast_s_max": 0.0,
            "resize/comm_prewarm_warmup_broadcast_s_mean": 0.0,
            "resize/comm_prewarm_warmup_total_s_max": 0.0,
            "resize/comm_prewarm_warmup_total_s_mean": 0.0,
            "resize/comm_activate_s": 0.0,
            "resize/comm_full_build_s": 0.0,
            "resize/comm_prepare_path": "",
            "resize/comm_cache_reused_group": "",
        }

    def _default_resize_budget_metrics(self) -> dict[str, float | str | int]:
        return {
            "resize/budget_enabled": 1.0 if self._resize_budget_config.enable else 0.0,
            "resize/budget_ratio": self._resize_budget_config.memory_budget_ratio,
            "resize/budget_blocked": 0.0,
            "resize/budget_reason": "",
            "resize/budget_effective_backend": self._handoff_staging_config.effective_backend(),
        }

    def _reset_resize_execution_metrics(self) -> dict[str, float | str | int]:
        self._latest_resize_execution_metrics = self._default_resize_execution_metrics()
        return dict(self._latest_resize_execution_metrics)

    def _reset_communicator_cache_metrics(self) -> dict[str, float | str | int]:
        self._latest_communicator_cache_metrics = self._default_communicator_cache_metrics()
        return dict(self._latest_communicator_cache_metrics)

    def _reset_resize_budget_metrics(self) -> dict[str, float | str | int]:
        self._latest_resize_budget_metrics = self._default_resize_budget_metrics()
        return dict(self._latest_resize_budget_metrics)

    def _update_resize_execution_metrics(self, **kwargs) -> dict[str, float | str | int]:
        self._latest_resize_execution_metrics.update(kwargs)
        return dict(self._latest_resize_execution_metrics)

    def _update_communicator_cache_metrics(self, **kwargs) -> dict[str, float | str | int]:
        self._latest_communicator_cache_metrics.update(kwargs)
        return dict(self._latest_communicator_cache_metrics)

    def _update_resize_budget_metrics(self, **kwargs) -> dict[str, float | str | int]:
        self._latest_resize_budget_metrics.update(kwargs)
        return dict(self._latest_resize_budget_metrics)

    def _start_deferred_optimizer_materialize(self, actor_wg: RayWorkerGroup) -> None:
        if getattr(self, "_pending_optimizer_materialize_refs", None):
            return
        try:
            self._pending_optimizer_materialize_refs = actor_wg.execute_all_async(f"{str(Role.Actor)}_materialize_pending_optimizer_restore")
            logger.info("[one-step-off][resize][host-stage] started async optimizer materialize on new actor group")
        except Exception as exc:  # pragma: no cover - runtime safety
            self._pending_optimizer_materialize_refs = None
            logger.warning("[one-step-off][resize][host-stage] failed to start async optimizer materialize: %r", exc)

    def _collect_deferred_optimizer_materialize_metrics(self, *, block: bool = False) -> dict[str, float | int]:
        refs = getattr(self, "_pending_optimizer_materialize_refs", None)
        if not refs:
            return {}
        timeout = None if block else 0
        ready, remaining = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        if remaining:
            return {}
        self._pending_optimizer_materialize_refs = None
        results = ray.get(ready)
        merged: dict[str, float | int] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    merged[key] = max(float(merged.get(key, 0.0)), float(value))
        return merged

    @staticmethod
    def _merge_resize_numeric_metrics(results) -> dict[str, float]:
        merged: dict[str, float] = {}
        for item in results or []:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key.startswith("resize/") and isinstance(value, (int, float)):
                    merged[key] = max(float(merged.get(key, 0.0)), float(value))
        return merged

    def _start_async_actor_host_restore(self, actor_wg: RayWorkerGroup, actor_resume_path: str, runtime_staging_cfg: dict):
        refs = actor_wg.execute_all_async(
            f"{str(Role.Actor)}_load_actor_handoff_state_from_host",
            actor_resume_path,
            staging_config=runtime_staging_cfg,
        )
        return {"refs": refs, "started_at": time.monotonic()}

    def _finish_async_actor_host_restore(
        self,
        restore_task,
        *,
        actor_wg: RayWorkerGroup,
        actor_resume_path: str,
        runtime_staging_cfg: dict,
    ) -> dict[str, float]:
        if restore_task is None:
            return {}
        wait_started_at = time.monotonic()
        try:
            restore_results = ray.get(restore_task["refs"])
        except Exception as exc:
            self._update_resize_execution_metrics(
                **{
                    "resize/restore_failed": 1.0,
                    "resize/partial_restore_cleanup_count": 1.0,
                }
            )
            logger.exception(
                "[one-step-off][resize][host-stage] async restore failed: path=%s "
                "resize/restore_failed=1.0 resize/partial_restore_cleanup_count=1.0 error=%r",
                actor_resume_path,
                exc,
            )
            try:
                actor_wg.cleanup_actor_handoff_restore_session(actor_resume_path)
            finally:
                raise
        wait_duration = time.monotonic() - wait_started_at
        import_duration = time.monotonic() - float(restore_task["started_at"])
        restore_preload_metrics = self._merge_resize_numeric_metrics(restore_results)
        metrics = {
            "resize/host_stage_import_s": import_duration,
            "resize/host_stage_import_wait_s": wait_duration,
            "resize/progressive_swap_s": import_duration if runtime_staging_cfg.get("progressive_swap", False) else 0.0,
            **restore_preload_metrics,
        }
        self._update_resize_execution_metrics(**metrics)
        return metrics

    @staticmethod
    def _extract_max_metric_value(results, key: str) -> float:
        values: list[float] = []

        def _collect(item) -> None:
            if item is None:
                return
            if isinstance(item, list):
                for sub in item:
                    _collect(sub)
                return
            if isinstance(item, dict) and key in item:
                try:
                    values.append(float(item[key]))
                except (TypeError, ValueError):
                    return

        _collect(results)
        return max(values) if values else 0.0

    @staticmethod
    def _estimate_weights_info_bytes(weights_info) -> int:
        total_bytes = 0
        for _, shape, dtype in weights_info:
            element_size = torch.empty((), dtype=dtype).element_size()
            total_bytes += math.prod(shape) * element_size
        return int(total_bytes)

    @staticmethod
    def _estimate_host_export_peak_bytes(model_bytes: int, *, stage_optimizer: bool) -> int:
        # Conservative heuristic: full model state plus CPU-side optimizer state.
        multiplier = 3.0 if stage_optimizer else 1.25
        return int(model_bytes * multiplier)

    @staticmethod
    def _estimate_gpu_restore_peak_bytes(
        model_bytes: int,
        *,
        progressive_swap: bool = False,
        chunk_mb: int | None = None,
    ) -> int:
        # When paged restore is enabled, the extra transient GPU pressure is
        # dominated by the current page rather than a second full model copy.
        working_bytes = model_bytes
        if progressive_swap and chunk_mb is not None and chunk_mb > 0:
            working_bytes = min(model_bytes, int(chunk_mb) * 1024 * 1024)
        return int(working_bytes * 1.2)

    def _make_runtime_handoff_staging_dict(self, *, effective_backend: str) -> dict[str, float | str | bool | int]:
        runtime_cfg = self._handoff_staging_config.to_manifest_dict()
        runtime_cfg["backend"] = effective_backend
        return runtime_cfg

    def _get_role_group_resource_snapshot(self, role: Role, role_wg: RayWorkerGroup, staging_path: str | None = None) -> ResizeBudgetSnapshot:
        # Query one representative worker before resize so the trainer can make
        # a budget decision without pushing large tensors through the switch path.
        snapshot = role_wg.execute_rank_zero_sync(f"{str(role)}_get_resize_resource_snapshot", staging_path)
        return ResizeBudgetSnapshot.from_dict(snapshot)

    def _current_topology_key(self) -> str:
        return build_topology_key(
            actor_spec=self._active_topology_specs.get(Role.Actor),
            rollout_spec=self._active_topology_specs.get(Role.Rollout),
        )

    def _candidate_topology_key(self, *, actor_spec: dict | None, rollout_spec: dict | None) -> str:
        return build_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec)

    def _warmup_communicator_cache_keys(self) -> None:
        if not self._communicator_cache_config.enable or not self._communicator_cache_config.reserve_schedule_topologies:
            return

        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        schedule = []
        if cfg is not None:
            cfg = OmegaConf.to_container(cfg, resolve=True)
            schedule = cfg.get("schedule", []) or []
            if isinstance(schedule, dict):
                schedule = list(schedule.values())

        for item in schedule:
            if not isinstance(item, dict):
                continue
            topology_key = self._candidate_topology_key(
                actor_spec=item.get("actor_pool"),
                rollout_spec=item.get("rollout_pool"),
            )
            self._weight_sync_communicator_cache.register_topology(
                topology_key,
                actor_spec=item.get("actor_pool"),
                rollout_spec=item.get("rollout_pool"),
                actor_world_size=self._pool_world_size_from_spec(Role.Actor, item.get("actor_pool")),
                rollout_world_size=self._pool_world_size_from_spec(Role.Rollout, item.get("rollout_pool")),
            )
            self._weight_sync_communicator_cache.reserve(topology_key)

    def _build_resize_control_metrics(self, snapshot: dict[str, float | str | int] | None = None) -> dict[str, float | str | int]:
        snapshot = snapshot or {}
        base = self._default_resize_control_metrics()
        for key, value in snapshot.items():
            base[f"resize/{key}"] = value
        return base

    def _record_resize_control_observation(self, *, timing_raw: dict, batch: DataProto | None) -> dict[str, float | str | int]:
        if self._resize_controller is None:
            self._latest_resize_control_metrics = self._default_resize_control_metrics()
            return dict(self._latest_resize_control_metrics)

        observation = build_resize_observation(timing_raw=timing_raw, batch=batch)
        snapshot = self._resize_controller.observe(step=self.global_steps, observation=observation)
        self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
        return dict(self._latest_resize_control_metrics)

    def _resolve_scheduled_resize_action(self, item: dict) -> str:
        actor_spec = item.get("actor_pool")
        rollout_spec = item.get("rollout_pool")
        if actor_spec is None or rollout_spec is None:
            return ACTION_HOLD

        active_actor = self._active_role_world_sizes.get(Role.Actor)
        active_rollout = self._active_role_world_sizes.get(Role.Rollout)
        if active_actor is None or active_rollout is None:
            return ACTION_HOLD

        actor_target = self._pool_world_size_from_spec(Role.Actor, actor_spec)
        rollout_target = self._pool_world_size_from_spec(Role.Rollout, rollout_spec)
        if self._resize_controller is None:
            actor_delta = actor_target - active_actor
            rollout_delta = rollout_target - active_rollout
            if rollout_delta > 0 and actor_delta < 0:
                return ACTION_EXPAND_ROLLOUT
            if actor_delta > 0 and rollout_delta < 0:
                return ACTION_EXPAND_TRAIN
            return ACTION_HOLD

        return self._resize_controller.infer_required_action(
            active_actor=active_actor,
            active_rollout=active_rollout,
            actor_target=actor_target,
            rollout_target=rollout_target,
        )

    def _gate_dynamic_resize_schedule_item(self, item: dict) -> tuple[bool, str, dict[str, float | str | int]]:
        required_action = self._resolve_scheduled_resize_action(item)
        if self._dynamic_resize_mode != "schedule_with_hysteresis" or self._resize_controller is None:
            snapshot = {
                "required_action": required_action,
                "required_action_code": ACTION_TO_CODE[required_action],
                "gate_pass": 1.0,
            }
            metrics = self._build_resize_control_metrics(snapshot)
            self._latest_resize_control_metrics = metrics
            return True, required_action, metrics

        allow, snapshot = self._resize_controller.gate(step=self.global_steps, required_action=required_action)
        metrics = self._build_resize_control_metrics(snapshot)
        self._latest_resize_control_metrics = metrics
        return allow, required_action, metrics

    def _maybe_patch_reward_metadata(self, non_tensor_batch) -> None:
        # Keep this fallback intentionally narrow. We only synthesize reward
        # metadata for samples that already look like GSM8K-style supervised
        # examples (i.e. they contain `answer`). This avoids silently forcing a
        # GSM8K router onto unrelated datasets while still unblocking the
        # one-step-off path that currently depends on the shared reward-loop
        # contract.
        answers = non_tensor_batch.get("answer")
        if answers is None:
            return

        if "data_source" not in non_tensor_batch:
            non_tensor_batch["data_source"] = np.array(["openai/gsm8k"] * len(answers), dtype=object)

        if "reward_model" not in non_tensor_batch:
            non_tensor_batch["reward_model"] = np.array(
                [{"ground_truth": answer, "style": "rule"} for answer in answers],
                dtype=object,
            )

    def _validate(self):
        @contextlib.contextmanager
        def _patched_val_dataloader():
            original_val_dataloader = self.val_dataloader

            def _iter_with_required_reward_fields():
                for test_data in original_val_dataloader:
                    # One-step-off validation still delegates to the shared PPO
                    # validation pipeline. That shared path assumes each sample
                    # already carries reward-routing metadata (`data_source`) and
                    # rule/model reward metadata (`reward_model`). Some validation
                    # datasets used by this experimental trainer only contain the
                    # textual fields (for example `prompt` and `answer`), so we
                    # patch the missing metadata here instead of changing the base
                    # trainer's contract for all PPO variants.
                    self._maybe_patch_reward_metadata(test_data)

                    yield test_data

            self.val_dataloader = _iter_with_required_reward_fields()
            try:
                yield
            finally:
                self.val_dataloader = original_val_dataloader

        self.actor_rollout_wg = self.actor_wg
        try:
            with _patched_val_dataloader():
                return super()._validate()
        finally:
            self.actor_rollout_wg = self.actor_wg

    def _build_runtime_batch_plan(self, batch: DataProto) -> dict[str, int]:
        total_samples = len(batch)
        existing_plan = batch.meta_info.get("runtime_batch_plan") or {}
        base_total_samples = int(existing_plan.get("base_total_samples", total_samples))
        actor_world_size = max(self._active_role_world_sizes.get(Role.Actor, 1), 1)
        critic_world_size = max(self._active_role_world_sizes.get(Role.Critic, actor_world_size), 1)

        actor_micro = self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", None)
        critic_micro = self.config.critic.get("ppo_micro_batch_size_per_gpu", None) if self.use_critic else None
        actor_micro = actor_micro if actor_micro is not None else max(base_total_samples // actor_world_size, 1)
        critic_micro = critic_micro if critic_micro is not None else max(base_total_samples // critic_world_size, 1)

        actor_alignment = actor_world_size * actor_micro
        critic_alignment = critic_world_size * critic_micro if self.use_critic else actor_alignment
        target_alignment = math.lcm(actor_alignment, critic_alignment)

        padded_total_samples = base_total_samples
        pad_samples = 0
        if base_total_samples % target_alignment != 0:
            padded_total_samples = math.ceil(base_total_samples / target_alignment) * target_alignment
            pad_samples = padded_total_samples - base_total_samples

        actor_runtime_mini = max(actor_alignment, padded_total_samples // actor_world_size)
        if actor_runtime_mini % actor_micro != 0:
            actor_runtime_mini = math.ceil(actor_runtime_mini / actor_micro) * actor_micro

        critic_runtime_mini = 0
        if self.use_critic:
            critic_runtime_mini = max(critic_alignment, padded_total_samples // critic_world_size)
            if critic_runtime_mini % critic_micro != 0:
                critic_runtime_mini = math.ceil(critic_runtime_mini / critic_micro) * critic_micro

        return {
            "base_total_samples": base_total_samples,
            "original_total_samples": total_samples,
            "padded_total_samples": padded_total_samples,
            "pad_samples": pad_samples,
            "net_pad_delta": padded_total_samples - total_samples,
            "target_alignment": target_alignment,
            "actor_world_size": actor_world_size,
            "actor_micro_batch_size_per_gpu": actor_micro,
            "actor_runtime_mini_batch_size": actor_runtime_mini,
            "critic_world_size": critic_world_size,
            "critic_micro_batch_size_per_gpu": critic_micro if self.use_critic else 0,
            "critic_runtime_mini_batch_size": critic_runtime_mini,
            "padding_source": "none",
            "global_step": self.global_steps,
        }

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each DP rank gets similar total tokens.

        This mirrors the shared PPO trainer implementation, except the actor DP
        size is queried from `actor_wg` directly because one-step-off keeps actor
        and rollout as separate groups instead of a composite `actor_rollout_wg`.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size(self.actor_wg, "actor")

        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)

        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _strip_resize_history_training_state(self, batch: DataProto) -> DataProto:
        tensor_exclude_keys = {
            "old_log_probs",
            "ref_log_prob",
            "values",
            "advantages",
            "returns",
            "token_level_rewards",
            "token_level_scores",
            "acc",
            "verifier_scores",
            "reward_baselines",
            "kl",
            "is_weights",
            "rollout_keep_mask",
        }
        tensor_keys = [key for key in batch.batch.keys() if key not in tensor_exclude_keys]
        return batch.select(batch_keys=tensor_keys, non_tensor_batch_keys=list(batch.non_tensor_batch.keys()), meta_info_keys=[])

    def _maybe_record_resize_history_batch(self, batch: DataProto) -> None:
        batch = self._strip_resize_history_training_state(batch)
        batch_size = len(batch)
        if batch_size <= 0:
            return
        snapshot_size = min(batch_size, 64)
        if batch_size == snapshot_size:
            snapshot = batch[:snapshot_size]
        else:
            sample_indices = np.random.choice(batch_size, size=snapshot_size, replace=False)
            snapshot = batch[sample_indices]
        snapshot = snapshot.select(meta_info_keys=[])

        self._resize_padding_history.append(snapshot)
        if len(self._resize_padding_history) > self._resize_padding_history_size:
            self._resize_padding_history = self._resize_padding_history[-self._resize_padding_history_size :]

    def _sample_padding_from_history(self, pad_samples: int) -> DataProto | None:
        if pad_samples <= 0 or not self._resize_padding_history:
            return None
        history_pool = DataProto.concat(self._resize_padding_history)
        if len(history_pool) <= 0:
            return None
        replace = len(history_pool) < pad_samples
        indices = np.random.choice(len(history_pool), size=pad_samples, replace=replace)
        padding_batch = history_pool[indices]
        if "uid" in padding_batch.non_tensor_batch:
            historical_uids = padding_batch.non_tensor_batch["uid"]
            padding_batch.non_tensor_batch["uid"] = np.array(
                [f"{uid}::histpad::step{self.global_steps}::{i}" for i, uid in enumerate(historical_uids)], dtype=object
            )
        return padding_batch

    def _apply_runtime_batch_plan(self, batch: DataProto) -> DataProto:
        existing_plan = batch.meta_info.get("runtime_batch_plan")
        if existing_plan is not None:
            self._last_runtime_batch_plan = existing_plan
            return batch

        plan = self._build_runtime_batch_plan(batch)
        self._last_runtime_batch_plan = plan

        if plan["net_pad_delta"] < 0:
            batch = batch[: plan["padded_total_samples"]]

            if "uid" in batch.non_tensor_batch:
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"][: plan["padded_total_samples"]]

        if plan["net_pad_delta"] > 0:
            padding_batch = self._sample_padding_from_history(plan["net_pad_delta"])
            repeat_indices = None
            if padding_batch is None:
                repeat_indices = np.arange(plan["net_pad_delta"]) % plan["original_total_samples"]
                padding_batch = batch[repeat_indices.tolist()]
                plan["padding_source"] = "current"
            else:
                plan["padding_source"] = "history"
            print(
                "[one-step-off][padding] step=%s base=%s current=%s target=%s net_delta=%s source=%s history_snapshots=%s",
                self.global_steps,
                plan["base_total_samples"],
                plan["original_total_samples"],
                plan["padded_total_samples"],
                plan["net_pad_delta"],
                plan["padding_source"],
                len(self._resize_padding_history),
            )
            batch = DataProto.concat([batch, padding_batch])

            repeated_from = batch.non_tensor_batch.get("uid", None)
            if repeated_from is not None and repeat_indices is not None:
                original_uids = repeated_from[: plan["original_total_samples"]]
                padded_uids = original_uids[repeat_indices]
                padded_uids = np.array(
                    [f"{uid}::currpad::step{self.global_steps}::{i}" for i, uid in enumerate(padded_uids)], dtype=object
                )
                batch.non_tensor_batch["uid"] = np.concatenate([original_uids, padded_uids])

        plan["pad_samples"] = max(plan["net_pad_delta"], 0)

        batch.meta_info["runtime_batch_plan"] = plan
        batch.meta_info["mini_batch_size"] = plan["actor_runtime_mini_batch_size"]
        batch.meta_info["critic_mini_batch_size"] = plan["critic_runtime_mini_batch_size"]

        return batch

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        # A startup gap between comm_full_build.end and sync_rollout_weights.start
        # is exactly this call plus _finalize_elastic_pool_replicas. Without a
        # trainer-lane envelope here the cold build of the initially-awake engines is
        # only recoverable by cross-lane inference.
        with resize_trace_span(
            self._resize_trace_config,
            "init_async_rollout_manager",
            step=self._trace_step(),
            lane="trainer_main",
        ):
            self._init_async_rollout_manager()
        self._finalize_elastic_pool_replicas()

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    def _create_worker_classes(self):
        self._create_actor_rollout_classes()
        self._create_critic_class()
        self._create_reference_policy_class()
        self._create_reward_model_class()

    def _create_actor_rollout_classes(self):
        for role in [Role.Actor, Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_critic_class(self):
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    def _create_reference_policy_class(self):
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
                # profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    def _create_reward_model_class(self):
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    def _get_detached_workers_cfg(self) -> dict | None:
        cfg = OmegaConf.select(self.config.trainer, "detached_workers")
        if cfg is None:
            return None
        return OmegaConf.to_container(cfg, resolve=True)

    def _get_worker_group_kwargs(self) -> dict:
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name
        return wg_kwargs

    def _get_resize_worker_group_kwargs(self) -> dict:
        wg_kwargs = self._get_worker_group_kwargs()

        # For SubRayResourcePool overlap-resize, scheduling an extra helper task
        # (`get_master_addr_port`) onto the reused placement group can block
        # because the old topology may still occupy the bundles. Reusing the old
        # worker group's rendezvous metadata is also unsafe because it couples the
        # new group to the old process group's endpoint. Instead, allocate a fresh
        # driver-side rendezvous endpoint and pass it explicitly to the new group.
        master_addr = None
        try:
            if getattr(self.actor_wg, "workers", None):
                master_addr = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
        except Exception as exc:
            logger.warning("[one-step-off][resize] failed to query current actor node ip for new rendezvous: %s", exc)

        if master_addr is None:
            master_addr = ray.util.get_node_ip_address().strip("[]")

        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        wg_kwargs["master_addr"] = str(master_addr)
        wg_kwargs["master_port"] = str(master_port)

        # Propagate the resize trace env into rebuilt worker processes so that
        # worker-side spans (e.g. the KV-cache profiling breakdown patched onto
        # vLLM's Worker.determine_available_memory) can write to the same trace
        # file. Without this, rebuilt rollout worker processes have no
        # VERL_RESIZE_TRACE_PATH and any worker-side resize_trace_span is
        # silently disabled -- the same root cause as the M0 "0 spans" gap on
        # the vLLMHttpServer actors. Keys here never collide with the protected
        # system env vars merged in RayWorkerGroup._create_worker.
        trace_cfg = getattr(self, "_resize_trace_config", None)
        if trace_cfg is not None and trace_cfg.enabled and trace_cfg.path:
            wg_kwargs["worker_env"] = {
                "VERL_RESIZE_TRACE_PATH": trace_cfg.path,
                "VERL_RESIZE_TRACE_RUN_NAME": trace_cfg.run_name,
            }
        return wg_kwargs

    def _init_worker_groups(self):
        detached_cfg = self._get_detached_workers_cfg()
        initial_pool_overrides = {
            Role.Actor: self._resolve_initial_role_pool_override(Role.Actor),
            Role.Rollout: self._resolve_initial_role_pool_override(Role.Rollout),
        }

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = self._get_worker_group_kwargs()

        # detached ray workers
        if detached_cfg and detached_cfg.get("enable"):
            name_prefix = detached_cfg.get("name_prefix", "verl_detached_")
            attach_only = detached_cfg.get("attach_only", False)
            worker_names_map = detached_cfg.get("worker_names", {}) or {}

            pool_name_map = {pool: name for name, pool in self.resource_pool_manager.resource_pool_dict.items()}

            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                pool_name = pool_name_map.get(resource_pool, "default")
                pool_worker_names = worker_names_map.get(pool_name)

                if pool_worker_names:
                    wg_dict = self.ray_worker_group_cls.from_detached(
                        name_prefix=name_prefix,
                        worker_names=pool_worker_names,
                        ray_cls_with_init=worker_dict_cls,
                        **wg_kwargs,
                    )
                else:
                    if attach_only:
                        raise ValueError(
                            f"detached_workers.attach_only is True but worker_names missing for pool '{pool_name}'"
                        )
                    wg_dict = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=worker_dict_cls,
                        name_prefix=name_prefix,
                        detached=True,
                        **wg_kwargs,
                    )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)
        # normal ray workers
        else:
            # IMPORTANT:
            # We intentionally avoid using `spawn(prefix_set=...)` here.
            # `spawn` creates multiple WorkerGroups that share the same underlying Ray Actors
            # (via from_detached), which will make actor/rollout overlap on the same ActorID.
            # That overlap will deadlock weight sync (broadcast) in detached / dynamic resize mode.
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                for role_name, role_cls in class_dict.items():
                    role = Role.from_string(role_name)
                    role_resource_pool = initial_pool_overrides.get(role) or resource_pool
                    worker_dict_cls = create_colocated_worker_cls(class_dict={role_name: role_cls})
                    wg = self.ray_worker_group_cls(
                        resource_pool=role_resource_pool,
                        ray_cls_with_init=worker_dict_cls,
                        **wg_kwargs,
                    )
                    # The returned dict is keyed by the role name.
                    spawn_wg = wg.spawn(prefix_set=[role_name])
                    all_wg.update(spawn_wg)
        self.all_wg = all_wg

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        if self._dynamic_resize_enabled and self._dynamic_resize_phased_init_enabled:
            self._prepare_role_group_init(self.actor_wg, role=Role.Actor)
            self._prepare_role_group_init(self.rollout_wg, role=Role.Rollout)
            self._commit_role_group_init(self.actor_wg, role=Role.Actor)
            self._commit_role_group_init(self.rollout_wg, role=Role.Rollout)
        else:
            self.actor_wg.init_model()
            self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg
        # Register the initial groups as the default targets.
        self.role_groups[Role.Actor] = {"primary": self.actor_wg}
        self.role_groups[Role.Rollout] = {"primary": self.rollout_wg}
        self._initialize_active_topology_state()
        # Elastic workers must exist before the manager launches replicas. The
        # missing replicas themselves are launched after the initial manager is
        # available, so initially active replicas can be adopted rather than
        # built twice.
        self._prepare_elastic_pool_workers()
        with resize_trace_span(
            self._resize_trace_config,
            "warmup_communicator_cache_keys",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            self._warmup_communicator_cache_keys()
        weights_info = self._get_actor_weights_info(self.actor_wg)[0]
        self._set_actor_weights_info(self.rollout_wg, weights_info)
        with resize_trace_span(
            self._resize_trace_config,
            "create_weight_sync_group_full",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            self._create_weight_sync_group()

    def _prepare_elastic_pool_workers(self) -> None:
        """Adopt active elastic workers and spawn only the missing groups.

        Replica launch is finalized after the initial rollout manager exists.
        This phase only establishes all worker handles while preserving the
        zero-Ray-quota placement recipe for missing groups.

        Every element of the spawn recipe is load-bearing:

        * ``use_gpu=False`` -> zero Ray GPU quota, so the actor can later claim
          these same bundles instead of queueing behind a booking that outlives
          the sleep (RayClassWithInitArgs.__call__ only sets num_gpus when
          use_gpu, ray/base.py:322-325).
        * ``placement_groups=<parent pgs>`` -> keeps the GPU bundle. use_gpu=False
          alone would drop the ``GPU: 1`` entry (ray/base.py:128-131); the bundle
          survives only because get_placement_groups early-returns when pgs are
          supplied (:116-117). split_resource_pool cannot be used here: it copies
          use_gpu from the parent (:229).
        * ``RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`` -> ray 2.53 otherwise rewrites
          CUDA_VISIBLE_DEVICES="" at zero-GPU actor creation and the injected
          value never reaches the process. Precedent: pinned_cpu_staging.py:87.
        * one group per TP group, each with its own group-wide bootstrap
          CUDA_VISIBLE_DEVICES. ElasticPoolRayWorkerGroup narrows that list to
          one physical token per actor before process startup. This is forced,
          not stylistic. _create_worker never sets LOCAL_RANK
          (ray/base.py:541-549), worker_env is a single dict per group, and both
          Worker.__init__ (base/worker.py:212) and vLLM's _init_worker
          (vllm_rollout.py:339-343) then compute local_rank=0, so both workers
          report local_rank_env="0" and land on one card. The
          executor even sends local_rank=None over ZMQ
          (vllm_async_server.py:118) and lets each worker recompute it. Scoping
          what each process can see makes vLLM's cuda:{local_rank} correct by
          construction rather than requiring a vLLM patch.

        Never set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES: under zero quota
        worker.py:288 indexes an empty accelerator_ids list and raises IndexError
        at actor init.
        """
        layout = self._sleep_pool_layout
        if layout is None:
            return

        # The zone model partitions ONE pool spanning every card, which only
        # exists when dynamic_resize.shared_pool.enable is set (main_ppo.py:50-64
        # registers it and maps every role onto it). Without it there is no single
        # pool whose bundles the elastic zone can carve out of, so fail with the
        # fix rather than a bare KeyError.
        pool_dict = self.resource_pool_manager.resource_pool_dict
        if "shared_pool" not in pool_dict:
            raise ValueError(
                "sleep_pool.enable=true requires dynamic_resize.shared_pool.enable=true: "
                "the elastic zone carves its bundles out of the shared pool, and without it "
                f"actor/rollout own separate pools. Available pools: {sorted(pool_dict)}"
            )
        shared_pool = pool_dict["shared_pool"]
        pgs = shared_pool.get_placement_groups(strategy="STRICT_PACK", device_name="cuda")

        initial_rollout_cards = self._active_role_world_sizes.get(Role.Rollout)
        if initial_rollout_cards is None:
            raise ValueError("elastic prebuild runs after the rollout group is committed")
        k0 = layout.schedule_from_rollout_cards(int(initial_rollout_cards))
        initial_awake = set(layout.awake_elastic_group_indices(k0))
        first_awake = layout.num_elastic_groups - k0

        pool = ElasticEnginePool(layout)
        self._elastic_prebuild_started_at = time.monotonic()
        for group_index in range(layout.num_elastic_groups):
            if group_index in initial_awake:
                initial_rank = group_index - first_awake
                start = initial_rank * layout.tp
                handles = self.rollout_wg.workers[start : start + layout.tp]
                if len(handles) != layout.tp:
                    raise ValueError(
                        f"initial elastic group {group_index} has {len(handles)} workers, expected TP={layout.tp}"
                    )
                pool.register_group(group_index, handles, awake=True)
                continue

            bundles = layout.elastic_group_bundles(group_index)
            elastic_pool = SubRayResourcePool(
                process_on_nodes=list(shared_pool.store),
                use_gpu=False,
                name_prefix=f"{shared_pool.name_prefix}elastic{group_index}_",
                max_colocate_count=shared_pool.max_colocate_count,
                placement_groups=pgs,
                start_bundle_index=bundles[0],
                subgroup_world_size=layout.tp,
            )
            # Ray actor names are f"{name_prefix}{cls}_{pg_idx}:{local_rank}"
            # (ray/base.py:565). Every group here spans its own TP-sized sub-pool,
            # so pg_idx/local_rank repeat across groups and a shared prefix would
            # collide -- same reasoning as the mixed survivor path.
            group_meta = {
                "group_index": group_index,
                "bundles": list(bundles),
                "tp": layout.tp,
                "initially_awake": False,
            }
            with resize_trace_span(
                self._resize_trace_config,
                "elastic_group_build_wg",
                step=self._trace_step(),
                lane="trainer_main",
                metadata=dict(group_meta),
            ):
                group_wg = self._build_role_group(
                    Role.Rollout,
                    resource_pool=elastic_pool,
                    name_prefix=f"elastic{group_index}_",
                    worker_env={
                        "CUDA_VISIBLE_DEVICES": ",".join(str(b) for b in bundles),
                        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                    },
                    worker_group_cls=ElasticPoolRayWorkerGroup,
                )
            with resize_trace_span(
                self._resize_trace_config,
                "elastic_group_prepare",
                step=self._trace_step(),
                lane="trainer_main",
                metadata=dict(group_meta),
            ):
                self._prepare_role_group_init(group_wg, role=Role.Rollout)
            with resize_trace_span(
                self._resize_trace_config,
                "elastic_group_commit",
                step=self._trace_step(),
                lane="trainer_main",
                metadata=dict(group_meta),
            ):
                self._commit_role_group_init(group_wg, role=Role.Rollout)
            if len(group_wg.workers) != layout.tp:
                raise ValueError(
                    f"elastic group {group_index} has {len(group_wg.workers)} workers, expected TP={layout.tp}"
                )
            pool.register_group(group_index, group_wg.workers, awake=False)

        self._elastic_pool = pool
        self._sleep_pool_initial_k = k0

    def _finalize_elastic_pool_replicas(self) -> None:
        """Adopt live replicas, launch missing ones, then record prebuild cost."""
        if self._elastic_pool is None:
            return

        with resize_trace_span(
            self._resize_trace_config,
            "finalize_elastic_pool_replicas",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={
                "initial_k": self._sleep_pool_initial_k,
                "awake_groups": list(self._elastic_pool.awake_indices),
                "sleeping_groups": list(self._elastic_pool.sleeping_indices),
                "num_replicas": self._sleep_pool_layout.num_elastic_groups,
            },
        ):
            pool = self._elastic_pool
            layout = self._sleep_pool_layout
            initial_awake = layout.awake_elastic_group_indices(self._sleep_pool_initial_k)
            initial_replicas = self.async_rollout_manager.rollout_replicas
            if len(initial_replicas) < len(initial_awake):
                raise ValueError(
                    f"initial rollout manager has {len(initial_replicas)} replicas, "
                    f"expected at least {len(initial_awake)} elastic replicas"
                )
            for initial_rank, group_index in enumerate(initial_awake):
                pool.register_replica(group_index, initial_replicas[initial_rank])

            missing_indices = [index for index in pool.all_registered if index not in initial_awake]
            replicas = []
            if missing_indices:
                handles = [handle for index in missing_indices for handle in pool.handles(index)]
                temp_wg = RayWorkerGroup.from_detached(
                    worker_handles=handles,
                    worker_names=[None] * len(handles),
                    ray_cls_with_init=self.rollout_wg.ray_cls_with_init,
                )
                replica_class = self.async_rollout_manager.rollout_replica_class
                replicas = [
                    replica_class(
                        replica_rank=rank,
                        config=self.config.actor_rollout_ref.rollout,
                        model_config=self.config.actor_rollout_ref.model,
                        gpus_per_node=self.config.trainer.n_gpus_per_node,
                    )
                    for rank in range(len(missing_indices))
                ]

                async def _launch_and_sleep():
                    with resize_trace_span(
                        self._resize_trace_config,
                        "elastic_pool_launch_engines",
                        step=self._trace_step(),
                        lane="trainer_main",
                        metadata={"num_replicas": len(replicas)},
                    ):
                        await asyncio.gather(*(replica.init_hybrid(temp_wg) for replica in replicas))
                    with resize_trace_span(
                        self._resize_trace_config,
                        "elastic_pool_initial_sleep",
                        step=self._trace_step(),
                        lane="trainer_main",
                        metadata={"num_replicas": len(replicas)},
                    ):
                        await asyncio.gather(*(replica.sleep() for replica in replicas))

                asyncio.run(_launch_and_sleep())

            for group_index, replica in zip(missing_indices, replicas, strict=True):
                pool.register_replica(group_index, replica)

            if set(pool.all_registered) != set(range(layout.num_elastic_groups)):
                raise ValueError("elastic worker prebuild is incomplete")
            if len(pool.all_replicas) != layout.num_elastic_groups:
                raise ValueError("elastic replica prebuild is incomplete")

            prebuild_s = time.monotonic() - self._elastic_prebuild_started_at
            self._update_resize_execution_metrics(**{"pool/prebuild_s": prebuild_s})
            logger.info(
                "[one-step-off][sleep-pool] prebuilt %d elastic replicas in %.2fs; "
                "k0=%d awake=%s sleeping=%s (startup peak-memory gate)",
                layout.num_elastic_groups,
                prebuild_s,
                self._sleep_pool_initial_k,
                pool.awake_indices,
                pool.sleeping_indices,
            )

        # B-line startup baseline. Deliberately after the prebuild accounting so
        # the probe's RPC cost is never attributed to A-line finalize timing.
        awake_result = self._sample_pool_memory(list(pool.awake_indices))
        sleeping_result = self._sample_pool_memory(list(pool.sleeping_indices))
        with resize_trace_span(
            self._resize_trace_config,
            "memory_probe_startup",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={
                "phase": "startup_baseline",
                "initial_k": self._sleep_pool_initial_k,
                "awake_groups": list(pool.awake_indices),
                "sleeping_groups": list(pool.sleeping_indices),
                "awake": awake_result,
                "sleeping": sleeping_result,
                "config_fingerprint": self._get_memory_config_fingerprint(),
            },
        ):
            pass

        # Fail here rather than after a full GPU run: an unreachable RPC or a
        # renamed method makes every later sample empty, and the trace would
        # still look successful. The condition keys on the groups actually
        # requested, not on expected_samples: a probe that reached no worker at
        # all reports expected_samples == 0, which used to short-circuit this
        # guard into silence. A side with no groups is legitimate and stays quiet.
        for label, groups, result in (
            ("awake", pool.awake_indices, awake_result),
            ("sleeping", pool.sleeping_indices, sleeping_result),
        ):
            if groups and not self._memory_samples_complete(result):
                raise RuntimeError(
                    f"startup memory probe incomplete for {label} groups: "
                    f"expected={result['expected_samples']} observed={result['observed_samples']} "
                    f"errors={result['error_count']}"
                )

    async def _apply_sleep_pool_schedule(self, rollout_spec: dict | None) -> tuple | None:
        """Sleep/wake the elastic zone to match the new topology's schedule.

        Replaces kill-and-cold-rebuild for the elastic groups: the engines stay
        alive across the resize and only give up their GPU memory. Returns the
        AwakeTopology the caller must rebuild rollout_wg from, or None when the
        pool is disabled (leaving every existing path byte-for-byte).

        Ordering here is not stylistic -- each step maps to a guardrail whose
        violation is silent rather than loud:

        * The caller MUST invoke this after _harvest_pending_generation (G3).
          Sleeping an engine with a request in flight is as fatal as killing it,
          and the harvest is the only barrier that guarantees quiescence.
        * sleep completes before wake begins (D6, strict serialization). The
          tempting overlap can look fine at a small model size and then OOM at a
          larger one with no clear attribution; waiting also keeps the actor's
          release latency (V4) observable instead of hidden.
        * This method never touches weights. Under VLLM_SLEEP_LEVEL=2 wake maps
          empty memory, and the driver's single sync entry point fills it (G1).
          A second weight path here would race that broadcast.
        * The returned topology's awake set is what rollout_wg must contain (G4);
          sleeping handles stay owned by the pool and out of the group, because
          sync_rollout_weights broadcasts to every member.
        """
        # Instrumentation state only. A resize that raised between wake and the
        # post-sync probe leaves a transition id behind; clearing it on entry is
        # what stops the NEXT resize from pairing its own probe with that corpse.
        # Every path that reads the slot runs after this line in the same resize.
        self._pending_memory_transition = None

        pool = self._elastic_pool
        layout = self._sleep_pool_layout
        if pool is None or layout is None:
            return None

        new_rollout_cards = int(self._pool_world_size_from_spec(Role.Rollout, rollout_spec))
        k = layout.schedule_from_rollout_cards(new_rollout_cards)
        topology = AwakeTopology.from_schedule(
            layout=layout,
            schedule_k=k,
            permanent_rollout_workers=self._permanent_rollout_workers(),
            elastic_group_workers=[pool.handles(i) for i in range(layout.num_elastic_groups)],
            actor_workers=list(getattr(self.actor_wg, "workers", [])),
        )

        to_sleep, to_wake = pool.reconcile(topology)
        logger.info(
            "[one-step-off][sleep-pool] schedule k=%d (rollout_cards=%d): sleep=%s wake=%s",
            k,
            new_rollout_cards,
            to_sleep,
            to_wake,
        )

        sleep_s = 0.0
        wake_s = 0.0
        transition_id = self._transition_id(k)
        fingerprint = self._get_memory_config_fingerprint()

        if to_sleep:
            sleep_before = self._sample_pool_memory(list(to_sleep))
            with resize_trace_span(
                self._resize_trace_config,
                "pool_sleep",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"groups": list(to_sleep), "schedule_k": k, "transition_id": transition_id},
            ):
                sleep_s = await pool.sleep_groups(to_sleep)
            sleep_after = self._sample_pool_memory(list(to_sleep))
            with resize_trace_span(
                self._resize_trace_config,
                "memory_probe_sleep",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={
                    "transition_id": transition_id,
                    "groups": list(to_sleep),
                    "schedule_k": k,
                    "sleep_before": sleep_before,
                    "sleep_after": sleep_after,
                    "config_fingerprint": fingerprint,
                },
            ):
                pass

        if to_wake:
            with resize_trace_span(
                self._resize_trace_config,
                "pool_wake",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"groups": list(to_wake), "schedule_k": k, "transition_id": transition_id},
            ):
                wake_s = await pool.wake_groups(to_wake)
            wake_mapped = self._sample_pool_memory(list(to_wake))
            with resize_trace_span(
                self._resize_trace_config,
                "memory_probe_wake_mapped",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={
                    "transition_id": transition_id,
                    "groups": list(to_wake),
                    "schedule_k": k,
                    "wake_mapped_after": wake_mapped,
                    "config_fingerprint": fingerprint,
                },
            ):
                pass
            # Under level 2 wake only maps empty pages; the driver refills them in
            # the post-resize sync. Occupancy "ready to generate" is sampled there.
            self._pending_memory_transition = {
                "transition_id": transition_id,
                "groups": list(to_wake),
                "schedule_k": k,
            }

        self._update_resize_execution_metrics(
            **{
                "pool/sleep_s": sleep_s,
                "pool/wake_s": wake_s,
                "pool/schedule_k": float(k),
                "pool/slept_groups": float(len(to_sleep)),
                "pool/woken_groups": float(len(to_wake)),
            }
        )
        return topology

    def _permanent_rollout_workers(self) -> list:
        """Handles of the permanent rollout zone: the last R cards, always awake.

        Card order is A | E | R (see ZoneLayout.elastic_group_bundles), so the
        permanent zone is the TAIL of the current rollout group, not its head.
        """
        layout = self._sleep_pool_layout
        workers = list(getattr(self.rollout_wg, "workers", []))
        if layout is None or layout.permanent_rollout_gpus == 0:
            return []
        if len(workers) < layout.permanent_rollout_gpus:
            raise ValueError(
                f"rollout group has {len(workers)} workers, fewer than the permanent zone "
                f"R={layout.permanent_rollout_gpus}"
            )
        return workers[-layout.permanent_rollout_gpus :]

    @staticmethod
    def _memory_sample_invalid(sample: object) -> bool:
        """Return whether one rank sample is unusable by the startup gate."""
        if not isinstance(sample, dict) or sample.get("sampling_error"):
            return True

        metric_names = (
            "memory_allocated_bytes",
            "memory_reserved_bytes",
            "memory_free_bytes",
            "memory_total_bytes",
        )
        metrics = [sample.get(name) for name in metric_names]
        if any(type(value) is not int or value < 0 for value in metrics):
            return True
        if sample["memory_free_bytes"] > sample["memory_total_bytes"]:
            return True
        return sample.get("actual_sleep_level") is None

    def _transition_id(self, schedule_k: int) -> str:
        """Stable key that pairs wake_mapped with its later wake_after_sync record."""
        counter = getattr(self, "_memory_transition_counter", 0) + 1
        self._memory_transition_counter = counter
        return f"step{self._trace_step()}_k{int(schedule_k)}_t{counter}"

    def _sample_pool_memory(self, group_indices: list[int], *, timeout_s: float = 30.0) -> dict:
        """Sample GPU memory on every TP rank of the named elastic groups.

        Reaches workers through the pool's public handles(): RolloutReplica only
        stores `workers`, and the pool deliberately imports no ray, so the
        trainer is the right place to own the RPC.

        All ObjectRefs are fired before the single bounded wait. Sampling one
        group at a time would make the probe cost scale with group count, and an
        unbounded wait on a wedged actor would hang the run instead of
        reporting an incomplete sample.

        Handle lookup and per-handle dispatch are guarded for the same reason as
        the wait: a probe that raises here would abort a live resize. Failures
        become error samples, which the startup gate still turns into a hard
        failure while a resize only records them.
        """
        pool = self._elastic_pool
        if pool is None or not group_indices:
            return {
                "samples": [],
                "expected_samples": 0,
                "observed_samples": 0,
                "error_count": 0,
                "sampling_rpc_ms": 0.0,
            }

        refs = []
        coordinates = []
        dispatch_failures: list[dict] = []
        # Derived from the layout, never from what dispatch produced. With
        # `len(refs) + len(dispatch_failures)` the comparison against
        # observed_samples was a tautology, so a truncated handle list (one handle
        # returned at TP=2) read as a complete sample set and the startup
        # fail-fast never fired. A missing layout leaves this 0, which
        # _memory_samples_complete rejects rather than waves through.
        layout = getattr(self, "_sleep_pool_layout", None)
        expected_samples = len(group_indices) * int(getattr(layout, "tp", 0) or 0)
        for group_index in group_indices:
            try:
                handles = list(pool.handles(group_index))
            except Exception as exc:
                # tp_rank -1: the group's rank count is exactly what we failed to learn.
                dispatch_failures.append(
                    {
                        "group_index": int(group_index),
                        "tp_rank": -1,
                        "sampling_error": f"handles failed: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            for tp_rank, handle in enumerate(handles):
                try:
                    refs.append(handle.get_memory_snapshot.remote())
                except Exception as exc:
                    dispatch_failures.append(
                        {
                            "group_index": int(group_index),
                            "tp_rank": int(tp_rank),
                            "sampling_error": f"dispatch failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                coordinates.append((int(group_index), int(tp_rank)))

        started_at = time.monotonic()
        samples: list[dict] = []
        try:
            payloads = ray.get(refs, timeout=timeout_s)
        except Exception as exc:
            payloads = [{"sampling_error": f"{type(exc).__name__}: {exc}"}] * len(refs)
        sampling_rpc_ms = round((time.monotonic() - started_at) * 1000.0, 3)

        for (group_index, tp_rank), payload in zip(coordinates, payloads, strict=True):
            if isinstance(payload, dict):
                record = dict(payload)
            else:
                record = {"sampling_error": f"unexpected payload type {type(payload).__name__}"}
            record["group_index"] = group_index
            record["tp_rank"] = tp_rank
            samples.append(record)
        samples.extend(dispatch_failures)

        error_count = sum(self._memory_sample_invalid(sample) for sample in samples)
        return {
            "samples": samples,
            "expected_samples": expected_samples,
            "observed_samples": len(samples),
            "error_count": error_count,
            "sampling_rpc_ms": sampling_rpc_ms,
        }

    @classmethod
    def _memory_samples_complete(cls, result: dict) -> bool:
        """True when metadata and every rank sample agree and are usable."""
        if not isinstance(result, dict):
            return False
        samples = result.get("samples")
        if not isinstance(samples, list):
            return False
        try:
            expected = int(result.get("expected_samples", 0))
            observed = int(result.get("observed_samples", 0))
            reported_error_count = int(result.get("error_count", 0))
        except (TypeError, ValueError):
            return False
        if expected <= 0:
            return False
        if len(samples) != observed or observed != expected:
            return False

        actual_error_count = sum(cls._memory_sample_invalid(sample) for sample in samples)
        if reported_error_count != actual_error_count or actual_error_count:
            return False

        sleep_level = samples[0].get("actual_sleep_level")
        return all(sample.get("actual_sleep_level") == sleep_level for sample in samples)

    def _get_memory_config_fingerprint(self) -> dict:
        """Config that determines the memory footprint, read from real fields.

        RolloutConfig keeps these flat (workers/config/rollout.py): there is no
        `.inference` subconfig or nested engine kwargs. The effective sleep level is
        deliberately absent here -- only the worker knows it.
        """
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return {
            "model_path": self.config.actor_rollout_ref.model.path,
            "gpu_memory_utilization": rollout_cfg.gpu_memory_utilization,
            "tensor_model_parallel_size": rollout_cfg.tensor_model_parallel_size,
            "max_num_seqs": rollout_cfg.max_num_seqs,
            "enforce_eager": rollout_cfg.enforce_eager,
            "free_cache_engine": rollout_cfg.free_cache_engine,
        }

    def _pool_world_size_from_spec(self, role: Role, pool_spec: dict | None) -> int:
        pool = self._resolve_role_pool(role, pool_spec)
        if pool is None:
            pool = self.resource_pool_manager.get_resource_pool(role)
        world_size = getattr(pool, "world_size", None)
        if world_size is None:
            raise ValueError(f"Cannot determine world_size for role={role}, pool_spec={pool_spec}, pool={pool}")
        return int(world_size)

    def _initialize_active_topology_state(self) -> None:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is not None:
            cfg = OmegaConf.to_container(cfg, resolve=True)
            if not cfg.get("enable", False):
                self._active_role_world_sizes[Role.Actor] = self._pool_world_size_from_spec(Role.Actor, None)
                self._active_role_world_sizes[Role.Rollout] = self._pool_world_size_from_spec(Role.Rollout, None)
                return

        initial_item = self._get_initial_dynamic_resize_schedule_item()

        if initial_item is not None:
            actor_spec = initial_item.get("actor_pool")
            rollout_spec = initial_item.get("rollout_pool")
            self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)
            return

        self._active_role_world_sizes[Role.Actor] = self._pool_world_size_from_spec(Role.Actor, None)
        self._active_role_world_sizes[Role.Rollout] = self._pool_world_size_from_spec(Role.Rollout, None)

    def _publish_active_topology_state(self, *, actor_spec: dict | None, rollout_spec: dict | None) -> None:
        self._active_topology_specs[Role.Actor] = actor_spec
        self._active_topology_specs[Role.Rollout] = rollout_spec
        self._active_role_world_sizes[Role.Actor] = self._pool_world_size_from_spec(Role.Actor, actor_spec)
        self._active_role_world_sizes[Role.Rollout] = self._pool_world_size_from_spec(Role.Rollout, rollout_spec)

    def _get_initial_dynamic_resize_schedule_item(self) -> dict | None:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return None
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not cfg.get("enable", False):
            return None

        schedule = cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())

        for item in schedule:
            if not isinstance(item, dict):
                continue
            if item.get("step", 0) < 0 and item.get("actor_pool") and item.get("rollout_pool"):
                return item
        return None

    def _resolve_initial_role_pool_override(self, role: Role):
        if role not in {Role.Actor, Role.Rollout}:
            return None

        initial_item = self._get_initial_dynamic_resize_schedule_item()
        if initial_item is None:
            return None

        pool_spec = initial_item.get("actor_pool") if role == Role.Actor else initial_item.get("rollout_pool")
        if not pool_spec:
            return None
        return self._resolve_role_pool(role, pool_spec)

    def _build_role_group(
        self,
        role: Role,
        *,
        resource_pool=None,
        name_prefix: str | None = None,
        detached: bool = False,
        worker_env: dict | None = None,
        worker_group_cls: type[RayWorkerGroup] | None = None,
    ) -> RayWorkerGroup:
        # Build a single-role group to keep dynamic add/switch logic reusable.
        resource_pool = resource_pool or self.resource_pool_manager.get_resource_pool(role)
        role_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[role],
            config=self.config.actor_rollout_ref,
            role=str(role),
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict={str(role): role_cls})
        wg_kwargs = self._get_resize_worker_group_kwargs()
        if worker_env:
            # RayWorkerGroup forwards this into each actor's runtime_env.env_vars
            # (ray/base.py:388 -> :550-559). It is per GROUP, not per worker; the
            # elastic subclass performs per-worker narrowing before startup.
            merged_worker_env = dict(wg_kwargs.get("worker_env") or {})
            merged_worker_env.update(worker_env)
            wg_kwargs["worker_env"] = merged_worker_env
        selected_worker_group_cls = worker_group_cls or self.ray_worker_group_cls
        wg_dict = selected_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            name_prefix=name_prefix,
            detached=detached,
            **wg_kwargs,
        )
        spawn_wg = wg_dict.spawn(prefix_set=[str(role)])
        role_wg = spawn_wg[str(role)]
        return role_wg

    def _prepare_role_group_init(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """尽量提前执行不依赖最终切换发布的初始化准备。

        对 one-step-off 实验 worker，优先调用显式 prepare 生命周期；
        对未实现该接口的旧 worker，静默回退，保持兼容。
        """
        if not self._dynamic_resize_phased_init_enabled:
            return
        role_prefix = str(role)
        for method_name in ("prepare_worker_init", "prepare_model_init"):
            remote_method_name = f"{role_prefix}_{method_name}"
            with resize_trace_span(
                self._resize_trace_config,
                f"{role_prefix}_{method_name}_group",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"role": role_prefix, "phase": "prepare"},
            ):
                rank0_result = role_wg.execute_rank_zero_sync(remote_method_name)
                all_results = role_wg.execute_all_sync(remote_method_name)
            duration = self._extract_max_metric_value([rank0_result, all_results], f"resize/{method_name}_s")
            if duration > 0:
                self._update_resize_execution_metrics(**{f"resize/{role_prefix}_{method_name}_s": duration})

    async def _prepare_role_group_init_async(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """异步 prepare 接口骨架。

        当前实现先复用同步版本，统一 trainer 侧接口，便于后续把 prepare
        真正下沉到后台任务/并发编排中。
        """
        result = self._prepare_role_group_init(role_wg, role=role)
        if inspect.isawaitable(result):
            await result

    def _commit_role_group_init(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """执行依赖最终 worker/runtime 状态的初始化提交。

        优先走新的显式 commit 生命周期；如果 worker 还未实现，回退到原始
        `init_model()` 入口，确保非实验路径与已有逻辑不受影响。
        """
        role_prefix = str(role)
        commit_worker_method = f"{role_prefix}_commit_worker_init"
        commit_model_method = f"{role_prefix}_commit_model_init"
        with resize_trace_span(
            self._resize_trace_config,
            f"{role_prefix}_commit_worker_init_group",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"role": role_prefix, "phase": "commit_worker"},
        ):
            worker_results = role_wg.execute_all_sync(commit_worker_method)
        with resize_trace_span(
            self._resize_trace_config,
            f"{role_prefix}_commit_model_init_group",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"role": role_prefix, "phase": "commit_model"},
        ):
            model_results = role_wg.execute_all_sync(commit_model_method)

        self._update_resize_execution_metrics(
            **{
                f"resize/{role_prefix}_prepare_worker_init_s": self._extract_max_metric_value(
                    worker_results, "resize/prepare_worker_init_s"
                ),
                f"resize/{role_prefix}_commit_worker_init_s": self._extract_max_metric_value(
                    worker_results, "resize/commit_worker_init_s"
                ),
                f"resize/{role_prefix}_prepare_model_init_s": self._extract_max_metric_value(
                    model_results, "resize/prepare_model_init_s"
                ),
                f"resize/{role_prefix}_commit_model_init_s": self._extract_max_metric_value(
                    model_results, "resize/commit_model_init_s"
                ),
            }
        )

    async def _commit_role_group_init_async(self, role_wg: RayWorkerGroup, *, role: Role) -> None:
        """异步 commit 接口骨架。

        语义上该接口预留给“旧 worker 释放后再提交”的场景；当前阶段先同步执
        行，保证调用链已成型，后续可以把真正的异步调度填进来。
        """
        result = self._commit_role_group_init(role_wg, role=role)
        if inspect.isawaitable(result):
            await result

    def _get_actor_weights_info(self, actor_wg: RayWorkerGroup):
        return actor_wg.execute_all_sync(f"{str(Role.Actor)}_get_actor_weights_info")

    def _set_actor_weights_info(self, rollout_wg: RayWorkerGroup, weights_info) -> None:
        rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_set_actor_weights_info", weights_info)

    def _create_actor_weight_sync_group(
        self,
        actor_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ):
        return actor_wg.execute_all_sync(
            f"{str(Role.Actor)}_create_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _create_rollout_weight_sync_group(
        self,
        rollout_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
    ):
        return rollout_wg.execute_all_sync(
            f"{str(Role.Rollout)}_create_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _prepare_actor_weight_sync_group(
        self,
        actor_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
        explicit_ranks: list[int] | None = None,
    ):
        # explicit_ranks: optional per-worker absolute collective ranks. Actor
        # workers are freshly spawned on the resize path so get_rank()+rank_offset
        # is already correct, but forwarding the driver-pinned ranks keeps both
        # roles symmetric and robust. See _prepare_rollout_weight_sync_group for
        # the per-worker list dispatch rationale.
        if explicit_ranks is not None:
            n = len(actor_wg.workers)
            if len(explicit_ranks) != n:
                raise ValueError(
                    f"explicit_ranks length {len(explicit_ranks)} != actor worker count {n}"
                )
            return actor_wg.execute_all_sync(
                f"{str(Role.Actor)}_prepare_weight_sync_group",
                [master_address] * n,
                [master_port] * n,
                [rank_offset] * n,
                [world_size] * n,
                [group_name] * n,
                [int(r) for r in explicit_ranks],
            )
        return actor_wg.execute_all_sync(
            f"{str(Role.Actor)}_prepare_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _prepare_rollout_weight_sync_group(
        self,
        rollout_wg: RayWorkerGroup,
        master_address,
        master_port,
        rank_offset: int,
        world_size: int,
        group_name: str,
        explicit_ranks: list[int] | None = None,
    ):
        # explicit_ranks: optional per-worker absolute collective ranks. Required
        # for the survivor-reuse path, where adopted rollout workers keep a stale
        # torch.distributed world so worker-side get_rank()+rank_offset is wrong.
        # The driver already pins ranks by list position in create_collective_group
        # (rollout workers occupy ranks rank_offset..rank_offset+R-1), so we forward
        # those exact ranks. execute_all_sync distributes list args per worker when
        # every arg is a list of length == #workers, so we broadcast the scalar
        # args as per-worker lists alongside explicit_ranks.
        if explicit_ranks is not None:
            n = len(rollout_wg.workers)
            if len(explicit_ranks) != n:
                raise ValueError(
                    f"explicit_ranks length {len(explicit_ranks)} != rollout worker count {n}"
                )
            return rollout_wg.execute_all_sync(
                f"{str(Role.Rollout)}_prepare_weight_sync_group",
                [master_address] * n,
                [master_port] * n,
                [rank_offset] * n,
                [world_size] * n,
                [group_name] * n,
                [int(r) for r in explicit_ranks],
            )
        return rollout_wg.execute_all_sync(
            f"{str(Role.Rollout)}_prepare_weight_sync_group",
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
        )

    def _destroy_actor_weight_sync_group(self, actor_wg: RayWorkerGroup, group_name: str):
        return actor_wg.execute_all_sync(f"{str(Role.Actor)}_destroy_weight_sync_group", group_name)

    def _destroy_rollout_weight_sync_group(self, rollout_wg: RayWorkerGroup, group_name: str):
        return rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_destroy_weight_sync_group", group_name)

    def _set_weight_sync_src_rank(self, actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, group_name: str, src_rank: int) -> None:
        actor_wg.execute_all_sync(f"{str(Role.Actor)}_set_weight_sync_src_rank", group_name, int(src_rank))
        rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_set_weight_sync_src_rank", group_name, int(src_rank))

    def _clear_weight_sync_src_rank(self, actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, group_name: str) -> None:
        actor_wg.execute_all_sync(f"{str(Role.Actor)}_clear_weight_sync_src_rank", group_name)
        rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_clear_weight_sync_src_rank", group_name)

    def _warmup_weight_sync_group(self, actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, group_name: str):
        # The first NCCL collective on a freshly prepared Ray collective group can
        # pay a large lazy-initialization cost. Exercise the exact group with a
        # one-element broadcast before it becomes the full weight-sync path.
        rollout_refs = rollout_wg.execute_all_async(f"{str(Role.Rollout)}_warmup_weight_sync_group", group_name)
        actor_refs = actor_wg.execute_all_async(f"{str(Role.Actor)}_warmup_weight_sync_group", group_name)
        return ray.get(rollout_refs + actor_refs)

    @staticmethod
    def _collect_weight_sync_group_warmup_metrics(results) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {}
        if not results:
            return metrics
        for field in ("broadcast_s", "total_s"):
            values = [float(item[field]) for item in results if isinstance(item, dict) and isinstance(item.get(field), (int, float))]
            if values:
                metrics[f"resize/comm_prewarm_warmup_{field}_max"] = max(values)
                metrics[f"resize/comm_prewarm_warmup_{field}_mean"] = sum(values) / len(values)
        role_counts: dict[str, int] = {}
        for item in results:
            if isinstance(item, dict):
                role = str(item.get("role", "unknown"))
                role_counts[role] = role_counts.get(role, 0) + 1
        for role, count in role_counts.items():
            metrics[f"resize/comm_prewarm_warmup_{role}_workers"] = float(count)
        first = next((item for item in results if isinstance(item, dict)), None)
        if first is not None:
            for field in ("group_name", "collective_world_size", "collective_initialized"):
                value = first.get(field)
                if isinstance(value, (int, float, bool)):
                    metrics[f"resize/comm_prewarm_warmup_{field}"] = float(value)
                elif value is not None:
                    metrics[f"resize/comm_prewarm_warmup_{field}"] = str(value)
        return metrics

    def add_role_group(
        self,
        role: Role,
        *,
        name: str | None = None,
        resource_pool=None,
        detached: bool = False,
        name_prefix: str | None = None,
        prepare_only: bool = False,
    ) -> RayWorkerGroup:
        role_wg = self._build_role_group(
            role,
            resource_pool=resource_pool,
            name_prefix=name_prefix,
            detached=detached,
        )
        self._prepare_role_group_init(role_wg, role=role)
        if not prepare_only:
            self._commit_role_group_init(role_wg, role=role)
        group_map = self.role_groups.setdefault(role, {})
        group_name = name or f"{str(role)}_group_{len(group_map)}"
        group_map[group_name] = role_wg
        return role_wg

    async def add_role_group_async(
        self,
        role: Role,
        *,
        name: str | None = None,
        resource_pool=None,
        detached: bool = False,
        name_prefix: str | None = None,
        prepare_only: bool = False,
    ) -> RayWorkerGroup:
        """异步 role-group 创建接口骨架。

        目标语义：
        - 可以在旧 worker 生命周期未结束时，先把新 group spawn 出来并 prepare
        - commit 则视调度策略稍后执行

        当前阶段底层 spawn 仍是同步封装，因此这里先统一 async API 形态。
        """
        role_wg = self._build_role_group(
            role,
            resource_pool=resource_pool,
            name_prefix=name_prefix,
            detached=detached,
        )
        await self._prepare_role_group_init_async(role_wg, role=role)
        if not prepare_only:
            await self._commit_role_group_init_async(role_wg, role=role)
        group_map = self.role_groups.setdefault(role, {})
        group_name = name or f"{str(role)}_group_{len(group_map)}"
        group_map[group_name] = role_wg
        return role_wg

    def switch_role_group(
        self,
        role: Role,
        role_wg: RayWorkerGroup,
        *,
        resume_from_path: str | None = None,
        release_old: bool = False,
    ) -> None:
        # Switch the active group and sync weights/manager as needed.
        if role == Role.Actor:
            old_wg = self.actor_wg
            if resume_from_path is not None:
                role_wg.load_checkpoint(
                    resume_from_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )
            else:
                weights_info = self._get_actor_weights_info(old_wg)[0]
                self._set_actor_weights_info(role_wg, weights_info)

            self.actor_wg = role_wg
            self.actor_rollout_wg = role_wg
            weights_info = self._get_actor_weights_info(role_wg)[0]
            self._set_actor_weights_info(self.rollout_wg, weights_info)

        elif role == Role.Rollout:
            old_wg = self.rollout_wg
            if resume_from_path is not None:
                print("Warning: rollout group does not support checkpoint restore; ignoring resume_from_path")
            weights_info = self._get_actor_weights_info(self.actor_wg)[0]
            self._set_actor_weights_info(role_wg, weights_info)
            self.rollout_wg = role_wg
            # Rollout manager must be rebuilt to bind the new rollout workers.
            self._init_async_rollout_manager()

        else:
            raise ValueError(f"Unsupported role for switch: {role}")

        self._create_weight_sync_group()

        if release_old:
            self.remove_role_group(role, old_wg)

    def remove_role_group(
        self,
        role: Role,
        role_wg: RayWorkerGroup,
        *,
        keep_workers: set | None = None,
    ) -> None:
        """Remove a role group, killing its worker actors.

        Args:
            keep_workers: optional set of worker actor handles to preserve. Their
                underlying processes are NOT killed (survivor reuse). The group is
                still unregistered from ``role_groups`` regardless. Matched by
                Ray ``_actor_id`` so a handle re-fetched elsewhere still matches.
        """
        keep_ids = set()
        if keep_workers:
            for w in keep_workers:
                actor_id = getattr(w, "_actor_id", None)
                if actor_id is not None:
                    keep_ids.add(actor_id)
        for group_name, group in list(self.role_groups.get(role, {}).items()):
            if group is role_wg:
                self.role_groups[role].pop(group_name)
                break
        for worker in role_wg.workers:
            if keep_ids and getattr(worker, "_actor_id", None) in keep_ids:
                logger.info("[one-step-off][resize][survivor] preserving %s worker %s", role, worker)
                continue
            try:
                ray.kill(worker)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                print(f"Warning: failed to kill {role} worker {worker}: {exc}")

    def add_actor_group(self, **kwargs) -> RayWorkerGroup:
        return self.add_role_group(Role.Actor, **kwargs)

    def switch_actor_group(self, actor_wg: RayWorkerGroup, **kwargs) -> None:
        return self.switch_role_group(Role.Actor, actor_wg, **kwargs)

    def remove_actor_group(self, actor_wg: RayWorkerGroup) -> None:
        return self.remove_role_group(Role.Actor, actor_wg)

    def add_rollout_group(self, **kwargs) -> RayWorkerGroup:
        return self.add_role_group(Role.Rollout, **kwargs)

    def switch_rollout_group(self, rollout_wg: RayWorkerGroup, **kwargs) -> None:
        return self.switch_role_group(Role.Rollout, rollout_wg, **kwargs)

    def remove_rollout_group(self, rollout_wg: RayWorkerGroup) -> None:
        return self.remove_role_group(Role.Rollout, rollout_wg)

    def _next_weight_sync_group_name(self) -> str:
        self._weight_sync_group_version += 1
        new_name = f"actor_rollout_v{self._weight_sync_group_version}"
        import traceback
        msg = (
            f"[TRACE] _next_weight_sync_group_name() called, version: {self._weight_sync_group_version - 1} -> {self._weight_sync_group_version}, new_name: {new_name}\n"
            f"{''.join(traceback.format_stack()[-5:-1])}"
        )
        print(msg, flush=True)
        logger.info(msg)
        return new_name

    def _is_weight_sync_group_name_busy(self, group_name: str | None) -> bool:
        if group_name is None:
            return False
        if group_name == getattr(self, "_active_weight_sync_group_name", None):
            return True
        return any(pending_group_name == group_name for pending_group_name, _, _ in self._pending_weight_sync_group_cleanup)

    def _register_weight_sync_group_for_cleanup(
        self, group_name: str, actor_wg: RayWorkerGroup | None, rollout_wg: RayWorkerGroup | None
    ) -> None:
        if actor_wg is None or rollout_wg is None:
            return
        self._pending_weight_sync_group_cleanup.append((group_name, actor_wg, rollout_wg))

    @staticmethod
    def _is_expected_weight_sync_cleanup_error(exc: Exception) -> bool:
        message = str(exc)
        return "`ray.kill`" in message or "does not exist" in message

    def _cleanup_pending_weight_sync_groups(self) -> None:
        pending = self._pending_weight_sync_group_cleanup
        self._pending_weight_sync_group_cleanup = []
        for group_name, actor_wg, rollout_wg in pending:
            self._weight_sync_communicator_cache.discard_by_group_name(group_name)
            try:
                self._destroy_actor_weight_sync_group(actor_wg, group_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if self._is_expected_weight_sync_cleanup_error(exc):
                    logger.debug(
                        "[one-step-off][resize] skip actor-side weight sync cleanup for terminated worker "
                        f"group_name={group_name}: {exc}"
                    )
                else:
                    logger.warning(
                        "[one-step-off][resize] warning: failed to destroy actor-side weight sync group "
                        f"group_name={group_name}: {exc}"
                    )
            try:
                self._destroy_rollout_weight_sync_group(rollout_wg, group_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if self._is_expected_weight_sync_cleanup_error(exc):
                    logger.debug(
                        "[one-step-off][resize] skip rollout-side weight sync cleanup for terminated worker "
                        f"group_name={group_name}: {exc}"
                    )
                else:
                    logger.warning(
                        "[one-step-off][resize] warning: failed to destroy rollout-side weight sync group "
                        f"group_name={group_name}: {exc}"
                    )
            try:
                collective.destroy_collective_group(group_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if self._is_expected_weight_sync_cleanup_error(exc):
                    logger.debug(
                        "[one-step-off][resize] skip driver-side weight sync cleanup for missing group "
                        f"group_name={group_name}: {exc}"
                    )
                else:
                    logger.warning(
                        "[one-step-off][resize] warning: failed to destroy driver-side weight sync group "
                        f"group_name={group_name}: {exc}"
                    )

    def _prepare_weight_sync_group(
        self,
        actor_wg: RayWorkerGroup,
        rollout_wg: RayWorkerGroup,
        *,
        group_name: str | None = None,
        topology_key: str | None = None,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ) -> str:
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = actor_wg.workers + rollout_wg.workers
        n_workers = len(actor_rollout_workers)
        topology_key = topology_key or self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec)
        actor_spec = actor_spec if actor_spec is not None else self._active_topology_specs.get(Role.Actor)
        rollout_spec = rollout_spec if rollout_spec is not None else self._active_topology_specs.get(Role.Rollout)
        had_registry_entry = self._weight_sync_communicator_cache.get_topology(topology_key) is not None
        self._weight_sync_communicator_cache.register_topology(
            topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(actor_wg.workers),
            rollout_world_size=len(rollout_wg.workers),
            world_size=n_workers,
        )

        cached_entry = self._weight_sync_communicator_cache.get(
            topology_key=topology_key,
            actor_wg=actor_wg,
            rollout_wg=rollout_wg,
        )
        if cached_entry is not None:
            self._update_communicator_cache_metrics(
                **{
                    "resize/comm_registry_hit": 1.0 if had_registry_entry else 0.0,
                    "resize/comm_registry_miss": 0.0 if had_registry_entry else 1.0,
                    "resize/comm_prewarm_ready": 1.0 if cached_entry.is_prewarmed else 0.0,
                    "resize/comm_prepare_path": "live_hit",
                    "resize/comm_cache_reused_group": cached_entry.group_name,
                }
            )
            return cached_entry.group_name

        candidate_group_name = group_name or self._weight_sync_communicator_cache.reserve(topology_key)
        if self._is_weight_sync_group_name_busy(candidate_group_name):
            candidate_group_name = None
        group_name = candidate_group_name or self._next_weight_sync_group_name()

        with resize_trace_span(
            self._resize_trace_config,
            "comm_prewarm_create",
            step=self._trace_step(),
            lane="comm_prewarm",
            metadata={"topology_key": topology_key, "group_name": group_name},
        ):
            started_at = time.monotonic()
            if self.device_name == "npu":
                master_address = ray.get(actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(actor_wg.workers[0]._get_free_port.remote())
                self._prepare_actor_weight_sync_group(actor_wg, master_address, master_port, 0, n_workers, group_name)
                self._prepare_rollout_weight_sync_group(
                    rollout_wg,
                    master_address,
                    master_port,
                    len(actor_wg.workers),
                    n_workers,
                    group_name,
                )
            else:
                collective.create_collective_group(
                    actor_rollout_workers,
                    n_workers,
                    list(range(0, n_workers)),
                    backend=get_nccl_backend(),
                    group_name=group_name,
                )
                master_address = ray.get(actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(actor_wg.workers[0]._get_free_port.remote())
                # Pass absolute collective ranks explicitly. The driver registered
                # ranks by list position in create_collective_group above: actor
                # workers occupy 0..A-1, rollout workers A..A+R-1. Passing these as
                # per-worker lists (execute_all distributes element i to worker i)
                # makes the worker use the authoritative rank instead of recomputing
                # torch.distributed.get_rank()+offset, which is stale for survivor
                # rollout workers adopted via from_detached (their old torch world
                # gives an out-of-range rank -> empty-message AssertionError ->
                # "topology not initialized"). Fresh workers would compute the same
                # value, so this is zero-regression on the full-rebuild path.
                n_actor = len(actor_wg.workers)
                n_rollout = len(rollout_wg.workers)
                self._prepare_actor_weight_sync_group(
                    actor_wg, master_address, master_port, 0, n_workers, group_name,
                    explicit_ranks=list(range(0, n_actor)),
                )
                self._prepare_rollout_weight_sync_group(
                    rollout_wg,
                    master_address,
                    master_port,
                    n_actor,
                    n_workers,
                    group_name,
                    explicit_ranks=list(range(n_actor, n_actor + n_rollout)),
                )
            warmup_started_at = time.monotonic()
            warmup_results = self._warmup_weight_sync_group(actor_wg, rollout_wg, group_name)
            warmup_duration = time.monotonic() - warmup_started_at
            warmup_metrics = self._collect_weight_sync_group_warmup_metrics(warmup_results)
            prewarm_duration = time.monotonic() - started_at

        self._weight_sync_communicator_cache.put(
            topology_key=topology_key,
            group_name=group_name,
            actor_wg=actor_wg,
            rollout_wg=rollout_wg,
            is_prewarmed=True,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(actor_wg.workers),
            rollout_world_size=len(rollout_wg.workers),
            world_size=n_workers,
        )
        self._update_communicator_cache_metrics(
            **{
                "resize/comm_registry_hit": 1.0 if had_registry_entry else 0.0,
                "resize/comm_registry_miss": 0.0 if had_registry_entry else 1.0,
                "resize/comm_prewarm_ready": 1.0,
                "resize/comm_prewarm_create_s": prewarm_duration,
                "resize/comm_prewarm_warmup_s": warmup_duration,
                "resize/comm_prepare_path": "registry_prewarm" if had_registry_entry else "full_build",
                "resize/comm_cache_reused_group": group_name,
                **warmup_metrics,
            }
        )
        return group_name

    def _create_weight_sync_group(
        self,
        *,
        group_name: str | None = None,
        topology_key: str | None = None,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ):
        from verl.utils.device import get_nccl_backend

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(actor_rollout_workers)
        topology_key = topology_key or self._current_topology_key()
        actor_spec = actor_spec if actor_spec is not None else self._active_topology_specs.get(Role.Actor)
        rollout_spec = rollout_spec if rollout_spec is not None else self._active_topology_specs.get(Role.Rollout)
        had_registry_entry = self._weight_sync_communicator_cache.get_topology(topology_key) is not None
        # Track topology-scoped metadata separately from live worker bindings.
        # This commit does not prewarm communicators yet; it only makes the
        # future prewarm input explicit and reusable across worker lifecycles.
        self._weight_sync_communicator_cache.register_topology(
            topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(self.actor_wg.workers),
            rollout_world_size=len(self.rollout_wg.workers),
            world_size=n_workers,
        )

        cached_entry = self._weight_sync_communicator_cache.get(
            topology_key=topology_key,
            actor_wg=self.actor_wg,
            rollout_wg=self.rollout_wg,
        )
        if cached_entry is not None:
            # Fast path: the communicator for this exact live actor/rollout pair
            # is already available, so only switch the active group binding.
            reused_group_name = cached_entry.group_name
            was_prewarmed = cached_entry.is_prewarmed
            with resize_trace_span(
                self._resize_trace_config,
                "comm_activate",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"topology_key": topology_key, "group_name": reused_group_name},
            ):
                activate_started_at = time.monotonic()
                self.actor_wg.execute_all_sync(f"{str(Role.Actor)}_activate_weight_sync_group", reused_group_name)
                self.rollout_wg.execute_all_sync(f"{str(Role.Rollout)}_activate_weight_sync_group", reused_group_name)
                activate_duration = time.monotonic() - activate_started_at
            if was_prewarmed:
                self._weight_sync_communicator_cache.mark_activated(
                    topology_key=topology_key,
                    actor_wg=self.actor_wg,
                    rollout_wg=self.rollout_wg,
                )
            self._active_weight_sync_group_name = reused_group_name
            self._update_communicator_cache_metrics(
                **{
                    "resize/comm_cache_hit": 1.0,
                    "resize/comm_cache_miss": 0.0,
                    "resize/comm_live_cache_hit": 1.0,
                    "resize/comm_registry_hit": 1.0 if had_registry_entry else 0.0,
                    "resize/comm_registry_miss": 0.0 if had_registry_entry else 1.0,
                    "resize/comm_prewarm_ready": 1.0 if was_prewarmed else 0.0,
                    "resize/comm_activate_s": activate_duration,
                    "resize/comm_prepare_path": "prewarmed_activate" if was_prewarmed else "live_hit",
                    "resize/comm_cache_reused_group": reused_group_name,
                }
            )
            return

        # Slow path: either this topology has never been seen, or it is being
        # revisited with newly spawned workers. In both cases we rebuild the
        # communicator and then refresh the cache entry.
        candidate_group_name = group_name or self._weight_sync_communicator_cache.reserve(topology_key)
        if self._is_weight_sync_group_name_busy(candidate_group_name):
            candidate_group_name = None
        group_name = candidate_group_name or self._next_weight_sync_group_name()
        registry_hit = had_registry_entry

        with resize_trace_span(
            self._resize_trace_config,
            "comm_full_build",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"topology_key": topology_key, "group_name": group_name},
        ):
            started_at = time.monotonic()
            if self.device_name == "npu":
                master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
                self._create_actor_weight_sync_group(
                    self.actor_wg,
                    master_address,
                    master_port,
                    0,
                    n_workers,
                    group_name,
                )
                self._create_rollout_weight_sync_group(
                    self.rollout_wg,
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                    group_name,
                )
            else:
                # Create Ray collective group for fallback communication
                collective.create_collective_group(
                    actor_rollout_workers,
                    n_workers,
                    list(range(0, n_workers)),
                    backend=get_nccl_backend(),
                    group_name=group_name,
                )
                # NOTE: collective init not finished before broadcast, so we init here to avoid potential issues
                # may not be necessary for all cases, but safer to have it
                master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
                master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
                self._create_actor_weight_sync_group(
                    self.actor_wg,
                    master_address,
                    master_port,
                    0,
                    n_workers,
                    group_name,
                )
                self._create_rollout_weight_sync_group(
                    self.rollout_wg,
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                    group_name,
                )
            full_build_duration = time.monotonic() - started_at
        self._active_weight_sync_group_name = group_name
        self._weight_sync_communicator_cache.put(
            topology_key=topology_key,
            group_name=group_name,
            actor_wg=self.actor_wg,
            rollout_wg=self.rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
            actor_world_size=len(self.actor_wg.workers),
            rollout_world_size=len(self.rollout_wg.workers),
            world_size=n_workers,
        )
        self._update_communicator_cache_metrics(
            **{
                "resize/comm_cache_hit": 0.0,
                "resize/comm_cache_miss": 1.0,
                "resize/comm_live_cache_hit": 0.0,
                "resize/comm_registry_hit": 1.0 if registry_hit else 0.0,
                "resize/comm_registry_miss": 0.0 if registry_hit else 1.0,
                "resize/comm_prewarm_ready": 0.0,
                "resize/comm_full_build_s": full_build_duration,
                "resize/comm_prepare_path": "full_build",
                "resize/comm_cache_reused_group": group_name,
            }
        )

    def _switch_weight_sync_group(
        self,
        new_actor_wg: RayWorkerGroup,
        new_rollout_wg: RayWorkerGroup,
        *,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ) -> None:
        # Cache lookup is topology-aware, but reuse is still constrained by the
        # concrete worker-group identities for safety.
        old_group_name = getattr(self, "_active_weight_sync_group_name", None)
        old_actor_wg = getattr(self, "actor_wg", None)
        old_rollout_wg = getattr(self, "rollout_wg", None)
        topology_key = self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec)
        reserved_group_name = self._weight_sync_communicator_cache.reserve(topology_key)
        if self._is_weight_sync_group_name_busy(reserved_group_name):
            reserved_group_name = None
        new_group_name = reserved_group_name or self._next_weight_sync_group_name()

        self.actor_wg = new_actor_wg
        self.rollout_wg = new_rollout_wg
        self.actor_rollout_wg = new_actor_wg
        self._create_weight_sync_group(
            group_name=new_group_name,
            topology_key=topology_key,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )

        if old_group_name is not None and old_actor_wg is not None and old_rollout_wg is not None:
            self._register_weight_sync_group_for_cleanup(old_group_name, old_actor_wg, old_rollout_wg)

    def _ensure_actor_training_group_binding(self) -> None:
        """Keep shared PPO paths bound to the current actor-side group.

        In one-step-off mode, rollout generation is handled explicitly by
        `rollout_wg` / `async_rollout_manager`, while shared PPO helpers such as
        `_balance_batch`, `compute_log_prob`, `update_actor`, profiling and
        snapshot dumping still implicitly access `actor_rollout_wg` expecting an
        actor-dispatch-capable group. Rebind it proactively to avoid stale or
        rollout-only references leaking into those paths after resize.
        """
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.one_step_off_policy.agent_loop import OneStepOffAgentLoopManager

        self.async_rollout_mode = True
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = OneStepOffAgentLoopManager(
            config=self.config, worker_group=self.rollout_wg, rm_resource_pool=rm_resource_pool
        )

    async def _init_async_rollout_manager_async(self, reuse_replicas: dict | None = None):
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.one_step_off_policy.agent_loop import OneStepOffAgentLoopManager

        self.async_rollout_mode = True
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = await OneStepOffAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
            rm_resource_pool=rm_resource_pool,
            reuse_replicas=reuse_replicas,
        )

    def _resolve_role_pool(self, role: Role, pool_spec: dict | None):
        if not pool_spec:
            return None

        mode = pool_spec.get("mode", "pool")
        if mode == "pool":
            # Directly pick a named pool (or the role default if name is omitted).
            pool_name = pool_spec.get("name")
            if pool_name is None:
                return self.resource_pool_manager.get_resource_pool(role)
            return self.resource_pool_manager.resource_pool_dict[pool_name]

        if mode == "split":
            # Split a base pool into sub-pools and select one by index.
            split_size = pool_spec.get("size")
            split_index = pool_spec.get("index", 0)
            # NOTE: split only partitions the base pool's existing world_size.
            # To switch from 4/4 -> 6/2, base_pool must be an 8-GPU pool
            # (e.g., shared_pool). Splitting a 4-GPU pool can never yield 6.
            base_pool_name = pool_spec.get("from_pool")
            if base_pool_name is not None:
                base_pool = self.resource_pool_manager.resource_pool_dict[base_pool_name]
            else:
                base_role = pool_spec.get("from_role", role)
                if isinstance(base_role, str):
                    base_role = Role[base_role]
                base_pool = self.resource_pool_manager.get_resource_pool(base_role)
            assert not (base_pool_name == "shared_pool" and isinstance(split_size, list)) or sum(split_size) == base_pool.world_size, (
                f"shared_pool world_size {base_pool.world_size} != sum(split_plan) {sum(split_size)}"
            )
            sub_pools = split_resource_pool(base_pool, split_size)
            if split_index >= len(sub_pools):
                raise IndexError(f"split index {split_index} out of range for {len(sub_pools)} sub-pools")
            return sub_pools[split_index]

        raise ValueError(f"Unsupported pool spec mode: {mode}")

    def _is_split_resize_spec(self, pool_spec: dict | None) -> bool:
        return bool(pool_spec) and pool_spec.get("mode", "pool") == "split"

    def _get_staged_shared_pool_resize_plan(self, item: dict) -> dict:
        actor_spec = item["actor_pool"]
        rollout_spec = item["rollout_pool"]

        old_actor_count = self._active_role_world_sizes.get(Role.Actor)
        old_rollout_count = self._active_role_world_sizes.get(Role.Rollout)
        if old_actor_count is None or old_rollout_count is None:
            raise ValueError(
                "staged shared-pool resize requires active topology state before resize, got "
                f"active_sizes={self._active_role_world_sizes}"
            )

        actor_target = actor_spec["size"][actor_spec.get("index", 0)]
        rollout_target = rollout_spec["size"][rollout_spec.get("index", 0)]
        actor_delta = actor_target - old_actor_count
        rollout_delta = rollout_target - old_rollout_count

        if actor_delta == 0 and rollout_delta == 0:
            raise ValueError("staged shared-pool resize got a no-op target topology")
        if actor_delta * rollout_delta >= 0:
            raise ValueError(
                "staged shared-pool resize requires one role to expand and the other to shrink, got "
                f"actor_delta={actor_delta}, rollout_delta={rollout_delta}"
            )
        if abs(actor_delta) != abs(rollout_delta):
            raise ValueError(
                "staged shared-pool resize requires balanced split reallocation, got "
                f"actor_delta={actor_delta}, rollout_delta={rollout_delta}"
            )

        shrinking_role = Role.Actor if actor_delta < 0 else Role.Rollout
        expanding_role = Role.Rollout if shrinking_role == Role.Actor else Role.Actor
        delta = abs(actor_delta)

        return {
            "actor_spec": actor_spec,
            "rollout_spec": rollout_spec,
            "old_actor_spec": self._active_topology_specs.get(Role.Actor),
            "old_rollout_spec": self._active_topology_specs.get(Role.Rollout),
            "old_actor_count": old_actor_count,
            "old_rollout_count": old_rollout_count,
            "actor_target": actor_target,
            "rollout_target": rollout_target,
            "actor_delta": actor_delta,
            "rollout_delta": rollout_delta,
            "shrinking_role": shrinking_role,
            "expanding_role": expanding_role,
            "delta": delta,
        }

    @staticmethod
    def _rollout_spec_bundles(rollout_spec: dict | None) -> list[int]:
        """Global GPU bundle indices owned by a rollout split spec.

        In shared_pool, split index ``i`` owns the contiguous bundle range
        ``[sum(size[:i]), sum(size[:i]) + size[i])``. Bundle indices are stable
        across resizes because ``split_resource_pool`` reuses the same PGs and
        only varies ``start_bundle_index`` (see single_controller/ray/base.py).
        Returns [] for a non-split spec (survivor reuse only applies to splits).
        """
        if not rollout_spec or rollout_spec.get("mode", "pool") != "split":
            return []
        size = list(rollout_spec["size"])
        index = rollout_spec.get("index", 0)
        start = sum(size[:index])
        return list(range(start, start + size[index]))

    def _compute_rollout_survivor_plan(
        self,
        old_rollout_spec: dict | None,
        new_rollout_spec: dict | None,
        rollout_world_size: int,
    ) -> dict:
        """Diff old vs new rollout engine GPU-sets to find survivors.

        Thin adapter over awake_topology.diff_engines. The algorithm lives there
        so axis 1 (survivor: not killed) and axis 2 (pool: not slept) cannot
        drift apart -- they are the same question. Return shape is deliberately
        unchanged so every existing caller and the resize_survivor_plan span
        stay comparable.

        Returns a dict with:
          - has_plan: bool (False if either spec is non-split -> fall back to full rebuild)
          - survivors: list of {old_rank, new_rank, bundles}
          - to_kill_old_ranks: list[int]  (old engines with no new counterpart)
          - to_build_new_ranks: list[int] (new engines with no old counterpart)
        """
        return diff_engines(
            self._rollout_spec_bundles(old_rollout_spec),
            self._rollout_spec_bundles(new_rollout_spec),
            rollout_world_size,
        )

    def _resolve_staged_actor_resume(self, actor_resume_path: str | None) -> tuple[str | None, bool, str | None]:
        if actor_resume_path is not None:
            logger.info("[one-step-off][resize][resume] using explicit actor checkpoint path: %s", actor_resume_path)
            return actor_resume_path, False, "checkpoint"

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        handoff_dir = os.path.join(checkpoint_folder, f"dynamic_resize_actor_handoff_step_{self.global_steps}")
        # When host staging is enabled, the same directory becomes a staging root
        # rather than a plain checkpoint handoff directory.
        if self._handoff_staging_config.enable:
            logger.info(
                "[one-step-off][resize][host-stage] using temporary actor host staging dir: %s backend=%s",
                handoff_dir,
                self._handoff_staging_config.effective_backend(),
            )
            return handoff_dir, self._handoff_staging_config.cleanup_after_load, "host_staging"
        logger.info("[one-step-off][resize][handoff] using temporary actor handoff dir: %s", handoff_dir)
        return handoff_dir, True, "handoff"

    def _normalize_actor_resume_load_path(self, actor_resume_path: str | None) -> str | None:
        if actor_resume_path is None:
            return None

        actor_subdir = os.path.join(actor_resume_path, "actor")
        if os.path.isdir(actor_subdir):
            return actor_subdir
        return actor_resume_path

    async def _shutdown_async_rollout_manager_async(self, keep_replica_ranks: set | None = None) -> None:
        manager = getattr(self, "async_rollout_manager", None)
        if manager is None:
            return

        shutdown_async = getattr(manager, "shutdown_async", None)
        if shutdown_async is None:
            self.async_rollout_manager = None
            return

        # When survivors are preserved, keep the manager object alive so the
        # survivor replica objects (with their live servers) can be reused by the
        # next init. Otherwise drop it entirely as before.
        if keep_replica_ranks:
            try:
                result = shutdown_async(keep_replica_ranks=keep_replica_ranks)
            except TypeError:
                result = shutdown_async()
            if inspect.isawaitable(result):
                await result
            return

        result = shutdown_async()
        if inspect.isawaitable(result):
            await result
        self.async_rollout_manager = None

    async def _harvest_pending_generation(self, pending_gen_future) -> tuple | None:
        # Await an in-flight raw-generation task at the resize deadline: after the
        # old actor is removed but BEFORE any rollout engine is destroyed. The raw
        # batch is topology-agnostic and is finalized on the next (post-resize)
        # step. Returns None when there is no pending future (flag-off / no
        # in-flight generation), leaving the resize path byte-for-byte unchanged.
        if pending_gen_future is None:
            return None
        with resize_trace_span(
            self._resize_trace_config,
            "resize_harvest_await",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            return await pending_gen_future

    async def _clear_rollout_kv_cache_before_resize_async(self) -> float:
        manager = getattr(self, "async_rollout_manager", None)
        if manager is None:
            return 0.0

        started_at = time.monotonic()
        clear_async = getattr(manager, "clear_kv_cache_async", None)
        if clear_async is not None:
            await clear_async()
            return time.monotonic() - started_at

        clear_sync = getattr(manager, "clear_kv_cache", None)
        if clear_sync is not None:
            result = clear_sync()
            if inspect.isawaitable(result):
                await result
            return time.monotonic() - started_at

        return 0.0

    def _is_overlap_prefetch_enabled(self) -> bool:
        # When True, dynamic-resize steps no longer drain the generation
        # pipeline: rollout keeps generating the next batch with the current
        # (pre-resize) weights so it overlaps actor train + export. Off by
        # default -- byte-for-byte the original defer-until-resize behavior.
        #
        # KNOWN LIMITATION: when enabled, EVERY step (not just resize-crossing
        # ones) routes generation through the raw/finalize split, so the
        # topology-bound tail -- including the async reward launch guarded by
        # `reward_model.launch_reward_fn_async` -- is deferred to the consuming
        # step. With async rewards ON, the original code launched rewards during
        # background generation so they overlapped actor train; the split moves
        # that launch to consume time and loses that overlap. Current experiments
        # run synchronous rewards (`launch_reward_fn_async=False`), so this path
        # is not exercised. Revisit before enabling async rewards with this flag.
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return False
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg, dict):
            return False
        # Mirror `_has_dynamic_resize_scheduled_at_step`: overlap-prefetch is a
        # dynamic-resize optimization, so a disabled resize feature disables it.
        if not cfg.get("enable", True):
            return False
        return bool(cfg.get("overlap_prefetch", False))

    def _has_dynamic_resize_scheduled_at_step(self, step: int) -> bool:
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return False
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not cfg.get("enable", True):
            return False
        schedule = cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())
        if not isinstance(schedule, list):
            return False
        return any(isinstance(item, dict) and item.get("step") == step for item in schedule)

    def _should_use_staged_shared_pool_resize(self, item: dict) -> bool:
        actor_spec = item.get("actor_pool")
        rollout_spec = item.get("rollout_pool")
        if not (self._is_split_resize_spec(actor_spec) and self._is_split_resize_spec(rollout_spec)):
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: split spec mismatch, "
                f"actor_spec={actor_spec}, rollout_spec={rollout_spec}"
            )
            return False

        actor_size = actor_spec.get("size")
        rollout_size = rollout_spec.get("size")
        if not (isinstance(actor_size, list) and isinstance(rollout_size, list) and len(actor_size) == len(rollout_size)):
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: size list mismatch, "
                f"actor_size={actor_size}, rollout_size={rollout_size}"
            )
            return False

        same_base_pool = actor_spec.get("from_pool") == rollout_spec.get("from_pool")
        if not same_base_pool:
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: actor/rollout base pool mismatch, "
                f"actor_from={actor_spec.get('from_pool')}, rollout_from={rollout_spec.get('from_pool')}"
            )
            return False

        old_actor = self._active_role_world_sizes.get(Role.Actor)
        old_rollout = self._active_role_world_sizes.get(Role.Rollout)
        if old_actor is None or old_rollout is None:
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: active topology sizes missing, "
                f"active_sizes={self._active_role_world_sizes}"
            )
            return False

        actor_target = actor_size[actor_spec.get("index", 0)]
        rollout_target = rollout_size[rollout_spec.get("index", 0)]
        actor_delta = actor_target - old_actor
        rollout_delta = rollout_target - old_rollout
        enabled = same_base_pool and actor_delta * rollout_delta < 0
        if not enabled:
            logger.debug(
                "[one-step-off][resize][staged-check] disabled: staged resize needs opposite deltas, "
                f"actor_delta={actor_delta}, rollout_delta={rollout_delta}"
            )
        return enabled

    async def _staged_resize_shared_pool(
        self, item: dict, *, actor_resume_path: str | None = None, pending_gen_future=None
    ) -> bool:
        plan = self._get_staged_shared_pool_resize_plan(item)
        self._last_harvested_raw = None
        actor_spec = plan["actor_spec"]
        rollout_spec = plan["rollout_spec"]
        detached = item.get("detached", False)
        name_prefix = item.get("name_prefix")

        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg
        old_actor_count = plan["old_actor_count"]
        old_rollout_count = plan["old_rollout_count"]
        actor_target = plan["actor_target"]
        rollout_target = plan["rollout_target"]
        actor_delta = plan["actor_delta"]
        rollout_delta = plan["rollout_delta"]
        shrinking_role = plan["shrinking_role"]
        model_bytes = self._estimate_weights_info_bytes(self._get_actor_weights_info(old_actor_wg)[0])

        actor_resume_path, should_cleanup_actor_resume, actor_resume_kind = self._resolve_staged_actor_resume(
            actor_resume_path,
        )
        runtime_staging_cfg = self._make_runtime_handoff_staging_dict(
            effective_backend=self._handoff_staging_config.effective_backend()
        )

        if actor_resume_kind in {"host_staging", "handoff"}:
            # The same budget gate covers both host-staging export and legacy
            # handoff export. The only difference is whether we are allowed to
            # downgrade the requested backend to disk fallback.
            export_snapshot = self._get_role_group_resource_snapshot(Role.Actor, old_actor_wg, staging_path=actor_resume_path)
            export_decision = self._resize_budget_controller.evaluate_export(
                requested_backend=self._handoff_staging_config.backend if actor_resume_kind == "host_staging" else "disk_fallback",
                snapshot=export_snapshot,
                estimated_host_peak_bytes=self._estimate_host_export_peak_bytes(
                    model_bytes,
                    stage_optimizer=self._handoff_staging_config.stage_optimizer if actor_resume_kind == "host_staging" else True,
                ),
                estimated_stage_bytes=model_bytes,
            )
            self._update_resize_budget_metrics(
                **{
                    "resize/budget_blocked": 1.0 if export_decision.blocked else 0.0,
                    "resize/budget_reason": export_decision.reason,
                    "resize/budget_effective_backend": export_decision.effective_backend,
                }
            )
            if export_decision.blocked:
                logger.warning(
                    "[one-step-off][resize][budget] skip staged resize before export: step=%s reason=%s",
                    self.global_steps,
                    export_decision.reason,
                )
                return False
            if actor_resume_kind == "host_staging":
                runtime_staging_cfg = self._make_runtime_handoff_staging_dict(effective_backend=export_decision.effective_backend)

        # Phase 0: export the train-side state into the host staging backend
        # before shared-pool slots are reclaimed.
        if actor_resume_kind == "host_staging":
            logger.info("[one-step-off][resize][host-stage] exporting actor state before staged switch")
            with resize_trace_span(
                self._resize_trace_config,
                "resize_export",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
            ):
                export_started_at = time.monotonic()
                export_results = old_actor_wg.stage_actor_handoff_state_to_host(
                    actor_resume_path,
                    staging_config=runtime_staging_cfg,
                )
                export_duration = time.monotonic() - export_started_at
            export_metrics = {"resize/host_stage_export_s": export_duration}
            export_metrics.update(self._merge_resize_numeric_metrics(export_results))
            export_metrics["resize/host_stage_split_optimizer_export"] = 0.0
            self._update_resize_execution_metrics(**export_metrics)
            if pending_gen_future is None and self._handoff_staging_config.preclear_rollout_kv_cache:
                # Flag-off / no-overlap path keeps the original pre-export preclear
                # timing. When a future is pending, preclear is deferred to after
                # the harvest await (see the harvest deadline below) so it cannot
                # race an in-flight decode.
                kv_cache_preclear_s = await self._clear_rollout_kv_cache_before_resize_async()
                self._update_resize_execution_metrics(**{"resize/kv_cache_preclear_s": kv_cache_preclear_s})
        elif actor_resume_kind == "handoff":
            logger.info("[one-step-off][resize][handoff] exporting actor state before staged switch")
            with resize_trace_span(
                self._resize_trace_config,
                "resize_export",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={"backend": "handoff"},
            ):
                export_started_at = time.monotonic()
                old_actor_wg.save_actor_handoff_state(actor_resume_path)
                export_duration = time.monotonic() - export_started_at
            self._update_resize_execution_metrics(**{"resize/host_stage_export_s": export_duration})

        # GPU restore is checked after KV-cache preclear/export, because the
        # effective free memory at that moment is what matters for the switch.
        gpu_role = Role.Rollout if shrinking_role == Role.Rollout else Role.Actor
        gpu_wg = old_rollout_wg if shrinking_role == Role.Rollout else old_actor_wg
        restore_snapshot = self._get_role_group_resource_snapshot(gpu_role, gpu_wg, staging_path=actor_resume_path)
        restore_decision = self._resize_budget_controller.evaluate_restore(
            snapshot=restore_snapshot,
            estimated_gpu_peak_bytes=self._estimate_gpu_restore_peak_bytes(
                model_bytes,
                progressive_swap=bool(runtime_staging_cfg.get("progressive_swap", False)),
                chunk_mb=int(runtime_staging_cfg.get("chunk_mb", 0) or 0),
            ),
        )
        self._update_resize_budget_metrics(
            **{
                "resize/budget_blocked": 1.0 if restore_decision.blocked else 0.0,
                "resize/budget_reason": restore_decision.reason,
                "resize/budget_effective_backend": runtime_staging_cfg.get("backend", "disk_fallback"),
            }
        )
        if restore_decision.blocked:
            logger.warning(
                "[one-step-off][resize][budget] skip staged resize before restore: step=%s reason=%s",
                self.global_steps,
                restore_decision.reason,
            )
            if should_cleanup_actor_resume and actor_resume_path is not None:
                shutil.rmtree(actor_resume_path, ignore_errors=True)
                self._update_resize_execution_metrics(**{"resize/host_stage_cleanup": 1.0})
            return False

        logger.info(
            "[one-step-off][resize][staged] start: "
            f"old_actor={old_actor_count}, old_rollout={old_rollout_count}, "
            f"target_actor={actor_target}, target_rollout={rollout_target}, "
            f"actor_delta={actor_delta}, rollout_delta={rollout_delta}, "
            f"shrinking_role={shrinking_role}"
        )

        # Phase 1: release the old pair, then build the new actor and rollout
        # groups concurrently. The resize handoff already preserves the actor
        # model state.
        #
        # When worker reuse is enabled and survivors cover 100% of the new
        # rollout topology (pure shrink), keep survivor workers alive and adopt
        # them via from_detached instead of spawning fresh processes.
        logger.info(
            "[one-step-off][resize][worker-reuse] DEBUG: worker_reuse_enabled=%s",
            self._dynamic_resize_worker_reuse_enabled,
        )
        rollout_survivor_plan = None
        if self._dynamic_resize_worker_reuse_enabled or self._sleep_pool_enabled:
            rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            old_rollout_spec = plan.get("old_rollout_spec")
            rollout_survivor_plan = self._compute_rollout_survivor_plan(
                old_rollout_spec,
                rollout_spec,
                rollout_world_size,
            )
            logger.info(
                "[one-step-off][resize][worker-reuse] survivor plan: has_plan=%s survivors=%d to_kill=%d to_build=%d",
                rollout_survivor_plan["has_plan"],
                len(rollout_survivor_plan["survivors"]),
                len(rollout_survivor_plan["to_kill_old_ranks"]),
                len(rollout_survivor_plan["to_build_new_ranks"]),
            )

        # Symmetrically tear down the old weight-sync collective group BEFORE
        # killing any actor worker. The shared group's TCPStore couples every
        # member process: if ray.kill(actor) runs first, survivor rollout workers
        # still attached to the old group see a broken pipe and their comm state
        # collapses (recvBytes/sendBytes Broken pipe -> "actor died unexpectedly"
        # -> ActorDiedError). Destroying both sides via the unified cleanup path
        # first lets rollout detach cleanly, so killing the actor no longer takes
        # the survivor rollout's comm down with it. Applies to every resize
        # (survivor and full-rebuild); the full-rebuild branch below also reaches
        # _cleanup_pending_weight_sync_groups but registering here keeps the
        # ordering invariant "tear down comm before killing workers" in one place.
        old_group_name = getattr(self, "_active_weight_sync_group_name", None)
        if old_group_name is not None and old_actor_wg is not None and old_rollout_wg is not None:
            self._register_weight_sync_group_for_cleanup(old_group_name, old_actor_wg, old_rollout_wg)
            self._cleanup_pending_weight_sync_groups()

        self.remove_role_group(Role.Actor, old_actor_wg)

        # Harvest deadline (common upper bound for shrink AND expand branches):
        # the old actor is gone, the shared weight-sync group is already torn
        # down symmetrically above, but NO rollout engine has been destroyed yet.
        # Await any in-flight raw generation here so it overlapped export +
        # group teardown + actor kill, and so no engine it runs on is killed out
        # from under it. Off / no-future: returns None, path unchanged.
        harvested_raw = await self._harvest_pending_generation(pending_gen_future)
        if harvested_raw is not None:
            self._last_harvested_raw = harvested_raw
            # Generation is now complete; clearing rollout KV cache can no longer
            # race a decoding request. Preclear here (moved from the export block)
            # to reclaim transient memory before the rebuild reclaims the slots.
            if pending_gen_future is not None and self._handoff_staging_config.preclear_rollout_kv_cache:
                kv_cache_preclear_s = await self._clear_rollout_kv_cache_before_resize_async()
                self._update_resize_execution_metrics(**{"resize/kv_cache_preclear_s": kv_cache_preclear_s})

        # Sleep-pool path (axis 2). Deliberately placed HERE: after the harvest
        # barrier above, so no engine is slept while a request is in flight (G3),
        # and before the survivor/full-rebuild decision below, which it replaces
        # for the elastic zone. Disabled -> returns None and everything downstream
        # behaves exactly as before.
        sleep_pool_reuse_replicas = None
        sleep_pool_keep_old_ranks = set()
        sleep_pool_topology = await self._apply_sleep_pool_schedule(rollout_spec)
        if sleep_pool_topology is not None:
            expected_bundles = tuple(self._rollout_spec_bundles(rollout_spec))
            if sleep_pool_topology.awake_rollout_bundles != expected_bundles:
                raise ValueError(
                    "sleep-pool topology does not match rollout spec: "
                    f"topology={sleep_pool_topology.awake_rollout_bundles} "
                    f"spec={expected_bundles}"
                )

            manager = self.async_rollout_manager
            sleep_pool_reuse_replicas, sleep_pool_keep_old_ranks = self._elastic_pool.build_reuse_plan(
                sleep_pool_topology,
                rollout_survivor_plan or {"survivors": []},
                manager.rollout_replicas,
            )
            actor_release_started_at = time.monotonic()
            sleep_pool_rollout_wg = RayWorkerGroup.from_detached(
                worker_handles=list(sleep_pool_topology.awake_rollout_workers),
                worker_names=[None] * len(sleep_pool_topology.awake_rollout_workers),
                ray_cls_with_init=old_rollout_wg.ray_cls_with_init,
            )
            # from_detached binds role-prefixed names but skips spawn()'s
            # _rebind_actor_methods, so the unprefixed aliases the trainer calls
            # (e.g. sync_rollout_weights) would be missing. Same fix as the
            # survivor path.
            _rollout_prefix = f"{str(Role.Rollout)}_"
            for _method_name in list(dir(sleep_pool_rollout_wg)):
                if _method_name.startswith(_rollout_prefix):
                    _unprefixed = _method_name.removeprefix(_rollout_prefix)
                    if not hasattr(sleep_pool_rollout_wg, _unprefixed):
                        setattr(
                            sleep_pool_rollout_wg,
                            _unprefixed,
                            getattr(sleep_pool_rollout_wg, _method_name),
                        )

            pool_worker_set = {
                handle
                for index in self._elastic_pool.all_registered
                for handle in self._elastic_pool.handles(index)
            }
            keep_worker_set = pool_worker_set | set(sleep_pool_topology.permanent_rollout_workers)
            self.remove_role_group(Role.Rollout, old_rollout_wg, keep_workers=keep_worker_set)
            self.rollout_wg = sleep_pool_rollout_wg
            self._active_role_world_sizes[Role.Rollout] = len(sleep_pool_rollout_wg.workers)
            self.role_groups[Role.Rollout] = {"primary": sleep_pool_rollout_wg}
            self._update_resize_execution_metrics(
                **{"resize/actor_release_s": time.monotonic() - actor_release_started_at}
            )
            with resize_trace_span(
                self._resize_trace_config,
                "resize_sleep_pool_plan",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={
                    "schedule_k": sleep_pool_topology.schedule_k,
                    "num_replicas": sleep_pool_topology.num_replicas,
                    "awake_workers": len(sleep_pool_topology.awake_rollout_workers),
                    "sleeping_workers": len(sleep_pool_topology.sleeping_workers),
                    "awake_bundles": list(sleep_pool_topology.awake_rollout_bundles),
                    "reuse_ranks": sorted(sleep_pool_reuse_replicas),
                    "keep_old_ranks": sorted(sleep_pool_keep_old_ranks),
                    "reuse_complete": len(sleep_pool_reuse_replicas) == sleep_pool_topology.num_replicas,
                },
            ):
                pass

        # Decide rollout path: survivor-reuse vs full rebuild.
        #
        # Pure shrink (to_build == 0) is the legacy condition. The MIXED case
        # (survivors + engines still to build, i.e. expansion) additionally needs
        # the (a)/(b)/(c) handling below and is therefore gated behind
        # worker_reuse.allow_mixed so the default path stays byte-for-byte.
        _survivor_base = (
            rollout_survivor_plan is not None
            and rollout_survivor_plan["has_plan"]
            and len(rollout_survivor_plan["survivors"]) > 0
        )
        _to_build_new_ranks = rollout_survivor_plan["to_build_new_ranks"] if rollout_survivor_plan else []
        use_rollout_survivor_path = sleep_pool_topology is None and _survivor_base and (
            len(_to_build_new_ranks) == 0 or self._dynamic_resize_worker_reuse_mixed_enabled
        )
        # Mixed = survivors kept AND new engines to build in the same resize.
        use_rollout_mixed_path = use_rollout_survivor_path and len(_to_build_new_ranks) > 0
        logger.info(
            "[one-step-off][resize][worker-reuse] DEBUG: use_survivor_path=%s mixed=%s "
            "(plan_not_none=%s has_plan=%s survivors>0=%s to_build=%s allow_mixed=%s)",
            use_rollout_survivor_path,
            use_rollout_mixed_path,
            rollout_survivor_plan is not None,
            rollout_survivor_plan["has_plan"] if rollout_survivor_plan else "N/A",
            len(rollout_survivor_plan["survivors"]) > 0 if rollout_survivor_plan else "N/A",
            _to_build_new_ranks,
            self._dynamic_resize_worker_reuse_mixed_enabled,
        )
        # Emit the plan + path decision as a machine-readable span. Without this
        # the survivor triple lives only in stdout, so a run that silently fell
        # back to a full rebuild (has_plan=False on a non-divisible topology, or
        # the mixed gate closed) is indistinguishable from a real survivor run
        # in the trace -- i.e. a false PASS. Zero-duration span: it records a
        # decision, not a duration.
        with resize_trace_span(
            self._resize_trace_config,
            "resize_survivor_plan",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={
                "has_plan": bool(rollout_survivor_plan["has_plan"]) if rollout_survivor_plan else False,
                "survivors": [
                    {"old_rank": s["old_rank"], "new_rank": s["new_rank"], "bundles": s["bundles"]}
                    for s in (rollout_survivor_plan["survivors"] if rollout_survivor_plan else [])
                ],
                "to_build_new_ranks": list(_to_build_new_ranks),
                "to_kill_old_ranks": list(rollout_survivor_plan["to_kill_old_ranks"])
                if rollout_survivor_plan
                else [],
                "use_survivor_path": bool(use_rollout_survivor_path),
                "use_mixed_path": bool(use_rollout_mixed_path),
                "allow_mixed_flag": bool(self._dynamic_resize_worker_reuse_mixed_enabled),
            },
        ):
            pass

        if sleep_pool_topology is not None:
            pass
        elif use_rollout_survivor_path:
            # Pure shrink: survivors cover 100% of new topology, no new engines.
            # Keep survivor workers alive and adopt them.
            logger.info(
                "[one-step-off][resize][worker-reuse] *** ENTERING SURVIVOR PATH ***"
            )
            survivors = rollout_survivor_plan["survivors"]
            rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size

            # Each survivor entry represents one replica (engine). A replica occupies
            # world_size consecutive workers: workers[world_size*old_rank : world_size*(old_rank+1)].
            # Collect all survivor worker handles in new_rank order.
            survivor_worker_handles = []
            keep_worker_set = set()
            # new_rank -> that replica's TP-ordered worker handles. Survivors fill
            # their entry here; the mixed path below fills the to_build ranks.
            handles_by_new_rank: dict[int, list] = {}
            for s in sorted(survivors, key=lambda x: x["new_rank"]):
                old_rank = s["old_rank"]
                start_idx = rollout_world_size * old_rank
                end_idx = rollout_world_size * (old_rank + 1)
                if end_idx > len(old_rollout_wg.workers):
                    logger.error(
                        "[one-step-off][resize][worker-reuse] old_rank %d out of bounds: world_size=%d, total_workers=%d",
                        old_rank,
                        rollout_world_size,
                        len(old_rollout_wg.workers),
                    )
                    raise IndexError(f"survivor old_rank {old_rank} out of bounds")
                replica_handles = old_rollout_wg.workers[start_idx:end_idx]
                survivor_worker_handles.extend(replica_handles)
                keep_worker_set.update(replica_handles)
                handles_by_new_rank[s["new_rank"]] = list(replica_handles)

            logger.info(
                "[one-step-off][resize][worker-reuse] %s: keeping %d survivor replicas (%d workers), killing %d replicas",
                "mixed" if use_rollout_mixed_path else "pure shrink",
                len(survivors),
                len(survivor_worker_handles),
                len(rollout_survivor_plan["to_kill_old_ranks"]),
            )
            self.remove_role_group(Role.Rollout, old_rollout_wg, keep_workers=keep_worker_set)

            # (c) Number of replicas in the NEW topology, derived from the new
            # rollout spec -- NOT from the survivor worker count. agent_loop
            # computes num_replicas = wg.world_size // TP, so the merged group
            # must span the whole new topology or survivors adopted at a new rank
            # >= num_replicas get silently dropped.
            #
            # Resolve the pool ONCE and derive both the card count and the
            # per-replica sub-pools from it, so "how many cards does the new
            # rollout topology have" has a single source of truth here (deriving
            # the count via _pool_world_size_from_spec while slicing a separately
            # resolved pool would re-introduce the algorithm fork that causes the
            # dropped-survivor bug).
            rollout_pool_for_delta = self._resolve_role_pool(Role.Rollout, rollout_spec)
            if rollout_pool_for_delta is None:
                raise ValueError(
                    f"survivor path requires a resolvable rollout pool spec, got rollout_spec={rollout_spec}"
                )
            expected_rollout_cards = int(rollout_pool_for_delta.world_size)
            if expected_rollout_cards % rollout_world_size != 0:
                raise ValueError(
                    f"new rollout cards {expected_rollout_cards} not divisible by TP {rollout_world_size}"
                )
            num_new_replicas = expected_rollout_cards // rollout_world_size

            # (b) Build the delta workers for the to_build ranks. The legacy path
            # skipped _build_commit_rollout entirely on the survivor branch, so on
            # expansion the new cards' workers were never created. Build one group
            # per missing replica: each gets its own TP-sized SubRayResourcePool
            # (correct start_bundle_index -> correct GPUs) and, via
            # _get_resize_worker_group_kwargs, its own rendezvous endpoint, so its
            # torch world is self-contained and RANK runs 0..TP-1 as vLLM expects.
            delta_new_ranks = sorted(r for r in _to_build_new_ranks if r < num_new_replicas)
            if delta_new_ranks:
                # split_resource_pool accepts an already-split SubRayResourcePool and
                # accumulates start_bundle_index (base.py:219), so replica_pools[i]
                # lands on the GPUs of new_rank i within the rollout segment.
                replica_pools = split_resource_pool(rollout_pool_for_delta, rollout_world_size)
                logger.info(
                    "[one-step-off][resize][worker-reuse] mixed: building delta workers for new_ranks=%s "
                    "(num_new_replicas=%d, TP=%d)",
                    delta_new_ranks,
                    num_new_replicas,
                    rollout_world_size,
                )
                for _new_rank in delta_new_ranks:
                    # Ray actor names are f"{name_prefix}{cls}_{pg_idx}:{local_rank}"
                    # (single_controller/ray/base.py _create_worker). Every delta group
                    # here spans its own TP-sized sub-pool, so pg_idx/local_rank repeat
                    # across groups: sharing one name_prefix would collide. The
                    # full-rebuild path can pass a shared prefix because it builds
                    # exactly one rollout group per topology.
                    _delta_prefix = f"{name_prefix or ''}dr{self.global_steps}_r{_new_rank}_"
                    delta_wg = await asyncio.to_thread(
                        self._build_role_group,
                        Role.Rollout,
                        resource_pool=replica_pools[_new_rank],
                        name_prefix=_delta_prefix,
                        detached=detached,
                    )
                    await asyncio.to_thread(self._prepare_role_group_init, delta_wg, role=Role.Rollout)
                    await asyncio.to_thread(self._commit_role_group_init, delta_wg, role=Role.Rollout)
                    if len(delta_wg.workers) != rollout_world_size:
                        raise ValueError(
                            f"delta rollout group for new_rank {_new_rank} has "
                            f"{len(delta_wg.workers)} workers, expected TP={rollout_world_size}"
                        )
                    handles_by_new_rank[_new_rank] = list(delta_wg.workers)

            # (a) Merge survivors + freshly built delta workers into ONE group,
            # ordered strictly by new_rank. agent_loop slices this list per replica
            # (workers[rank*TP:(rank+1)*TP]) and ExternalZeroMQDistributedExecutor
            # slices the ZMQ addresses the same way, so position == new_rank is a
            # hard contract, not a convenience.
            missing_ranks = [r for r in range(num_new_replicas) if r not in handles_by_new_rank]
            if missing_ranks:
                raise ValueError(
                    f"merged rollout group is missing workers for new_ranks {missing_ranks} "
                    f"(survivors={[s['new_rank'] for s in survivors]}, to_build={_to_build_new_ranks})"
                )
            merged_worker_handles = []
            for _new_rank in range(num_new_replicas):
                merged_worker_handles.extend(handles_by_new_rank[_new_rank])

            # Adopt survivors into a new worker group without spawning.
            # Note: from_detached needs worker_names for _world_size calculation,
            # even when using worker_handles. Pass a dummy list of same length.
            new_rollout_wg = RayWorkerGroup.from_detached(
                worker_handles=merged_worker_handles,
                worker_names=[None] * len(merged_worker_handles),
                ray_cls_with_init=old_rollout_wg.ray_cls_with_init,
            )
            # from_detached binds worker methods under their role-prefixed names
            # (e.g. rollout_sync_rollout_weights) via _bind_worker_method, but
            # skips the spawn() path that _rebind_actor_methods uses to also
            # expose the unprefixed aliases (e.g. sync_rollout_weights). The
            # trainer calls the unprefixed name (e.g. self.rollout_wg.sync_rollout_weights()),
            # so without this rebind an adopted survivor group raises
            # AttributeError: 'RayWorkerGroup' object has no attribute 'sync_rollout_weights'.
            # Mirror spawn()'s _rebind_actor_methods (single_controller/ray/base.py) to
            # restore the unprefixed aliases. Zero effect on the full-rebuild path.
            _rollout_prefix = f"{str(Role.Rollout)}_"
            for _method_name in list(dir(new_rollout_wg)):
                if _method_name.startswith(_rollout_prefix):
                    _unprefixed = _method_name.removeprefix(_rollout_prefix)
                    if not hasattr(new_rollout_wg, _unprefixed):
                        setattr(new_rollout_wg, _unprefixed, getattr(new_rollout_wg, _method_name))
            self.rollout_wg = new_rollout_wg
            self._active_role_world_sizes[Role.Rollout] = len(new_rollout_wg.workers)
            self.role_groups[Role.Rollout] = {"primary": new_rollout_wg}
            # Old collective group state was already torn down symmetrically above
            # (before remove_role_group(Actor)) via _cleanup_pending_weight_sync_groups,
            # which clears the survivor processes' group metadata just as well via
            # old_rollout_wg as via new_rollout_wg (same processes). No redundant
            # rollout-only destroy here.
            logger.info(
                "[one-step-off][resize][worker-reuse] adopted %d workers into new rollout group "
                "(%d survivor + %d newly built, num_replicas=%d)",
                len(merged_worker_handles),
                len(survivor_worker_handles),
                len(merged_worker_handles) - len(survivor_worker_handles),
                num_new_replicas,
            )
        else:
            # Default path: full rebuild (either no plan, or expansion/mixed topology).
            self.remove_role_group(Role.Rollout, old_rollout_wg)

        actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
        rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

        actor_resume_load_path = self._normalize_actor_resume_load_path(actor_resume_path)
        defer_optimizer_restore = bool(
            runtime_staging_cfg.get("stage_optimizer", False)
            and str(runtime_staging_cfg.get("optimizer_restore_policy", "deferred")) == "deferred"
        )
        actor_restore_task = None
        actor_restore_wait_task = None
        comm_prewarm_task = None

        async def _start_actor_restore_after_commit(actor_wg: RayWorkerGroup):
            restore_task = None
            if actor_resume_kind == "host_staging" and actor_resume_path is not None:
                logger.info("[one-step-off][resize][host-stage] starting async actor state import into new actor")
                with resize_trace_span(
                    self._resize_trace_config,
                    "resize_import_launch",
                    step=self._trace_step(),
                    lane="trainer_main",
                    metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
                ):
                    restore_task = self._start_async_actor_host_restore(
                        actor_wg,
                        actor_resume_path,
                        runtime_staging_cfg,
                    )
            elif actor_resume_kind == "handoff" and actor_resume_path is not None:
                logger.info("[one-step-off][resize][handoff] importing actor state into new actor")
                with resize_trace_span(
                    self._resize_trace_config,
                    "resize_import",
                    step=self._trace_step(),
                    lane="trainer_main",
                    metadata={"backend": "handoff"},
                ):
                    import_started_at = time.monotonic()
                    try:
                        actor_wg.load_actor_handoff_state(actor_resume_path)
                    except Exception as exc:
                        self._update_resize_execution_metrics(
                            **{
                                "resize/restore_failed": 1.0,
                                "resize/partial_restore_cleanup_count": 1.0,
                            }
                        )
                        logger.exception(
                            "[one-step-off][resize][handoff] restore failed: path=%s "
                            "resize/restore_failed=1.0 resize/partial_restore_cleanup_count=1.0 error=%r",
                            actor_resume_path,
                            exc,
                        )
                        try:
                            actor_wg.cleanup_actor_handoff_restore_session(actor_resume_path)
                        finally:
                            raise
                    import_duration = time.monotonic() - import_started_at
                self._update_resize_execution_metrics(**{"resize/host_stage_import_s": import_duration})
                if should_cleanup_actor_resume:
                    try:
                        shutil.rmtree(actor_resume_path)
                        self._update_resize_execution_metrics(**{"resize/host_stage_cleanup": 1.0})
                        logger.info("[one-step-off][resize][handoff] cleaned temporary actor handoff dir: %s", actor_resume_path)
                    except FileNotFoundError:
                        pass
                    except Exception as exc:  # pragma: no cover - best effort cleanup
                        logger.warning(
                            "[one-step-off][resize][handoff] failed to remove temporary handoff %s: %s",
                            actor_resume_path,
                            exc,
                        )
            elif actor_resume_load_path is not None:
                logger.info("[one-step-off][resize][resume] loading actor checkpoint into new actor: %s", actor_resume_load_path)
                actor_wg.load_checkpoint(
                    actor_resume_load_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )
                if should_cleanup_actor_resume:
                    try:
                        shutil.rmtree(actor_resume_path)
                        logger.info("[one-step-off][resize][resume] cleaned temporary checkpoint dir: %s", actor_resume_path)
                    except FileNotFoundError:
                        pass
                    except Exception as exc:  # pragma: no cover - best effort cleanup
                        logger.warning(
                            "[one-step-off][resize][resume] failed to remove temporary checkpoint %s: %s",
                            actor_resume_path,
                            exc,
                        )
            else:
                logger.info(
                    "[one-step-off][resize][staged] new actor keeps its freshly initialized weights; "
                    "rollout weight metadata will be attached after rollout group commit"
                )
            return restore_task

        async def _build_commit_actor():
            actor_wg = await asyncio.to_thread(
                self.add_role_group,
                Role.Actor,
                name=item.get("actor_group_name"),
                resource_pool=actor_pool,
                detached=detached,
                name_prefix=name_prefix,
                prepare_only=True,
            )
            await asyncio.to_thread(self._commit_role_group_init, actor_wg, role=Role.Actor)
            return actor_wg

        async def _build_commit_rollout():
            rollout_wg = await asyncio.to_thread(
                self.add_role_group,
                Role.Rollout,
                name=item.get("rollout_group_name"),
                resource_pool=rollout_pool,
                detached=detached,
                name_prefix=name_prefix,
                prepare_only=True,
            )
            await asyncio.to_thread(self._commit_role_group_init, rollout_wg, role=Role.Rollout)
            return rollout_wg

        async def _wait_for_actor_restore_async(actor_wg: RayWorkerGroup):
            logger.info("[one-step-off][resize][host-stage] waiting for async actor import before reading weights")

            def _wait_restore():
                with resize_trace_span(
                    self._resize_trace_config,
                    "resize_import_wait",
                    step=self._trace_step(),
                    lane="trainer_main",
                    metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
                ):
                    return self._finish_async_actor_host_restore(
                        actor_restore_task,
                        actor_wg=actor_wg,
                        actor_resume_path=actor_resume_path,
                        runtime_staging_cfg=runtime_staging_cfg,
                    )

            return await asyncio.to_thread(_wait_restore)

        actor_group_task = asyncio.create_task(_build_commit_actor())
        if sleep_pool_topology is None and not use_rollout_survivor_path:
            rollout_group_task = asyncio.create_task(_build_commit_rollout())
        new_actor_wg = await actor_group_task
        actor_restore_task = await _start_actor_restore_after_commit(new_actor_wg)
        if actor_restore_task is not None:
            actor_restore_wait_task = asyncio.create_task(_wait_for_actor_restore_async(new_actor_wg))
        if sleep_pool_topology is None and not use_rollout_survivor_path:
            new_rollout_wg = await rollout_group_task
            self._update_resize_execution_metrics(**{"resize/parallel_actor_rollout_reinit": 1.0})
        else:
            # Reuse path: new_rollout_wg already set in Phase 1.
            # Old collective group was cleaned up there; _prepare_weight_sync_group below
            # will create a fresh group for the new topology.
            new_rollout_wg = self.rollout_wg
            if sleep_pool_topology is not None:
                self._update_resize_execution_metrics(**{"resize/rollout_sleep_pool_reuse": 1.0})
            else:
                self._update_resize_execution_metrics(**{"resize/rollout_survivor_reuse": 1.0})

        if self._dynamic_resize_async_comm_prewarm_enabled:
            comm_prewarm_task = asyncio.create_task(
                asyncio.to_thread(
                    self._prepare_weight_sync_group,
                    new_actor_wg,
                    new_rollout_wg,
                    topology_key=self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec),
                    actor_spec=actor_spec,
                    rollout_spec=rollout_spec,
                )
            )

        if actor_restore_wait_task is not None:
            await actor_restore_wait_task
            if defer_optimizer_restore and has_restore_session_manifest(actor_resume_path):
                session_manifest = read_restore_session_manifest(actor_resume_path)
                self._update_resize_execution_metrics(
                    **{
                        "resize/optimizer_deferred_restore": 1.0,
                        "resize/optimizer_pending_pages": float(session_manifest.get("optimizer_page_count", 0)),
                    }
                )
            new_actor_wg.release_host_staging_buffer(
                actor_resume_path,
                staging_config=runtime_staging_cfg,
            )
            if should_cleanup_actor_resume and not defer_optimizer_restore:
                try:
                    shutil.rmtree(actor_resume_path)
                    self._update_resize_execution_metrics(**{"resize/host_stage_cleanup": 1.0})
                    logger.info("[one-step-off][resize][host-stage] cleaned temporary actor handoff dir: %s", actor_resume_path)
                except FileNotFoundError:
                    pass
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    logger.warning(
                        "[one-step-off][resize][host-stage] failed to remove temporary handoff %s: %s",
                        actor_resume_path,
                        exc,
                    )

        with resize_trace_span(
            self._resize_trace_config,
            "resize_postprocess",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"backend": runtime_staging_cfg.get("backend", "disk_fallback")},
        ):
            weights_info = self._get_actor_weights_info(new_actor_wg)[0]
            self._set_actor_weights_info(new_rollout_wg, weights_info)

            if comm_prewarm_task is not None:
                await comm_prewarm_task

            # Publish the new pair together.
            self._switch_weight_sync_group(
                new_actor_wg,
                new_rollout_wg,
                actor_spec=actor_spec,
                rollout_spec=rollout_spec,
            )
            self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)
            if defer_optimizer_restore:
                # Default pinned optimizer restore uses the original full-state
                # distribution path. Start it asynchronously right after publishing
                # the new topology so it can overlap with post-switch rollout work.
                # The experimental chunked optimizer path remains available behind
                # optimizer_full_state_restore=false.
                use_lazy_pinned_optimizer = (
                    runtime_staging_cfg.get("backend") == "pinned_cpu"
                    and bool(runtime_staging_cfg.get("async_optimizer_preload", True))
                    and not bool(runtime_staging_cfg.get("optimizer_full_state_restore", True))
                )
                if not use_lazy_pinned_optimizer:
                    self._start_deferred_optimizer_materialize(new_actor_wg)

            with resize_trace_span(
                self._resize_trace_config,
                "resize_rollout_shutdown",
                step=self._trace_step(),
                lane="trainer_main",
            ):
                if sleep_pool_topology is not None:
                    await self._shutdown_async_rollout_manager_async(
                        keep_replica_ranks=sleep_pool_keep_old_ranks
                    )
                elif use_rollout_survivor_path:
                    # Keep survivor replicas alive (servers + workers).
                    # shutdown_async() iterates the OLD manager's replicas, whose
                    # replica_rank is still in OLD-topology space -- survivors only
                    # adopt their new rank later, inside _init_async_rollout_manager_async
                    # (agent_loop.py: "survivor.replica_rank = replica_rank  # adopt new rank").
                    # So keep_replica_ranks must be the survivors' OLD ranks. Passing
                    # new ranks kills the real survivor (e.g. vllm_server_1_0) and keeps
                    # a doomed replica, which later crashes clear_kv_cache_async with
                    # ActorDiedError on the dead server during resize_postprocess.
                    survivor_old_ranks = {s["old_rank"] for s in rollout_survivor_plan["survivors"]}
                    logger.info(
                        "[one-step-off][resize][worker-reuse] shutdown: keeping %d survivor replicas",
                        len(survivor_old_ranks),
                    )
                    await self._shutdown_async_rollout_manager_async(keep_replica_ranks=survivor_old_ranks)
                else:
                    await self._shutdown_async_rollout_manager_async()
            with resize_trace_span(
                self._resize_trace_config,
                "resize_rollout_init",
                step=self._trace_step(),
                lane="trainer_main",
            ):
                if sleep_pool_topology is not None:
                    await self._init_async_rollout_manager_async(
                        reuse_replicas=sleep_pool_reuse_replicas
                    )
                elif use_rollout_survivor_path:
                    # Reuse survivor replica objects (with live servers).
                    manager = getattr(self, "async_rollout_manager", None)
                    if manager is not None and hasattr(manager, "rollout_replicas"):
                        # Validate indices before mapping.
                        max_old_rank = max(s["old_rank"] for s in rollout_survivor_plan["survivors"])
                        if max_old_rank >= len(manager.rollout_replicas):
                            logger.warning(
                                "[one-step-off][resize][worker-reuse] survivor old_rank %d >= manager replica count %d, falling back to full init",
                                max_old_rank,
                                len(manager.rollout_replicas),
                            )
                            await self._init_async_rollout_manager_async()
                        else:
                            survivor_replicas = {
                                s["new_rank"]: manager.rollout_replicas[s["old_rank"]]
                                for s in rollout_survivor_plan["survivors"]
                            }
                            logger.info(
                                "[one-step-off][resize][worker-reuse] init: reusing %d survivor replicas",
                                len(survivor_replicas),
                            )
                            await self._init_async_rollout_manager_async(reuse_replicas=survivor_replicas)
                    else:
                        logger.warning(
                            "[one-step-off][resize][worker-reuse] manager missing or no rollout_replicas, falling back to full init"
                        )
                        await self._init_async_rollout_manager_async()
                else:
                    await self._init_async_rollout_manager_async()
            await self.async_rollout_manager.wake_up_async()
            await self._sync_rollout_weights_after_resize_async(handoff_path=actor_resume_path)

        # Outside resize_postprocess on purpose: this probe's RPC round trip would
        # otherwise be counted as A-line postprocess time, inflating the very span
        # the startup attribution work reads.
        pending = self._pending_memory_transition
        self._pending_memory_transition = None
        if pending is not None:
            after_sync = self._sample_pool_memory(list(pending["groups"]))
            with resize_trace_span(
                self._resize_trace_config,
                "memory_probe_wake_after_sync",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={
                    "transition_id": pending["transition_id"],
                    "groups": pending["groups"],
                    "schedule_k": pending["schedule_k"],
                    # wake_after_sync - wake_mapped is NOT the weight broadcast:
                    # a manager-level wake_up_async runs in that interval and can
                    # itself allocate. Naming the steps stops the delta being read
                    # as "the weight sync". Kept in code order and pinned to the
                    # real source by test_post_sync_span_records_what_actually_
                    # runs_between_the_two_wake_samples.
                    "between": [
                        "resize_survivor_plan",
                        "resize_rollout_shutdown",
                        "resize_rollout_init",
                        "wake_up_async",
                        "_sync_rollout_weights_after_resize_async",
                    ],
                    "wake_after_sync": after_sync,
                    "config_fingerprint": self._get_memory_config_fingerprint(),
                },
            ):
                pass

        self._cleanup_pending_weight_sync_groups()
        logger.info("[one-step-off][resize][staged] done")
        return True

    async def _hard_resize_actor_rollout_from_checkpoint(self, item: dict, *, actor_resume_path: str | None = None) -> bool:
        """Baseline resize: destroy old actor/rollout workers, then rebuild from checkpoint.

        This path intentionally avoids the staged shared-pool handoff, pinned
        pages, direct rollout page load, phased prewarm, and communicator cache
        optimizations used by the main resize path. It keeps critic untouched so
        the experiment isolates actor/rollout topology rebuild cost.
        """
        actor_spec = item.get("actor_pool")
        rollout_spec = item.get("rollout_pool")
        detached = item.get("detached", False)
        name_prefix = item.get("name_prefix")
        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg

        handoff_path = actor_resume_path or os.path.join(
            self.config.trainer.default_local_dir,
            f"hard_resize_actor_handoff_step_{self.global_steps}",
        )
        if actor_resume_path is None:
            shutil.rmtree(handoff_path, ignore_errors=True)
        hard_staging_cfg = self._make_runtime_handoff_staging_dict(effective_backend="disk_fallback")
        hard_staging_cfg.update(
            {
                "backend": "disk_fallback",
                "stage_optimizer": True,
                "optimizer_restore_policy": "immediate",
                "progressive_swap": False,
                "async_optimizer_preload": False,
                "cleanup_after_load": False,
                "preclear_rollout_kv_cache": False,
            }
        )
        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_export_handoff",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"backend": "disk_fallback"},
        ):
            export_started_at = time.monotonic()
            export_results = old_actor_wg.stage_actor_handoff_state_to_host(
                handoff_path,
                staging_config=hard_staging_cfg,
            )
            export_duration = time.monotonic() - export_started_at
        export_metrics = {"resize/hard_resize_export_handoff_s": export_duration}
        export_metrics.update(self._merge_resize_numeric_metrics(export_results))
        self._update_resize_execution_metrics(**export_metrics)

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_shutdown_rollout_manager",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            await self._shutdown_async_rollout_manager_async()

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_kill_old_workers",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)

        actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
        rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

        new_actor_wg = await self.add_role_group_async(
            Role.Actor,
            name=item.get("actor_group_name"),
            resource_pool=actor_pool,
            detached=detached,
            name_prefix=name_prefix,
            prepare_only=False,
        )
        new_rollout_wg = await self.add_role_group_async(
            Role.Rollout,
            name=item.get("rollout_group_name"),
            resource_pool=rollout_pool,
            detached=detached,
            name_prefix=name_prefix,
            prepare_only=False,
        )

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_actor_load_handoff",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"path": handoff_path, "backend": "disk_fallback"},
        ):
            new_actor_wg.load_actor_handoff_state_from_host(
                handoff_path,
                staging_config=hard_staging_cfg,
            )

        self.actor_wg = new_actor_wg
        self.actor_rollout_wg = new_actor_wg
        self.rollout_wg = new_rollout_wg
        weights_info = self._get_actor_weights_info(new_actor_wg)[0]
        self._set_actor_weights_info(new_rollout_wg, weights_info)

        self._switch_weight_sync_group(
            new_actor_wg,
            new_rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )
        self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)

        with resize_trace_span(
            self._resize_trace_config,
            "hard_resize_init_rollout_manager",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={},
        ):
            await self._init_async_rollout_manager_async()
        await self._sync_rollout_weights_after_resize_async(handoff_path=None)
        self._cleanup_pending_weight_sync_groups()
        self._update_resize_execution_metrics(
            **{
                "resize/hard_resize_destroy_all": 1.0,
                "resize/parallel_actor_rollout_reinit": 0.0,
            }
        )
        return True

    async def _maybe_dynamic_resize(self, pending_gen_future=None) -> tuple | None:
        # Execute scheduled resize steps driven by config.trainer.dynamic_resize.
        # Accept both of the config shapes currently seen in experiments:
        #   1) a list of schedule items
        #   2) a named mapping such as {stage0: {...}, stage1: {...}}
        # The latter is convenient in YAML, but iterating over it directly
        # yields string keys, so normalize it before processing.
        # These two early returns predate barrier-sink and do not harvest
        # pending_gen_future. They are safe only because a live future is never
        # threaded here: fit() gates it behind `resize_scheduled_this_step and
        # overlap_prefetch`, and both `_has_dynamic_resize_scheduled_at_step`
        # and `_is_overlap_prefetch_enabled` return False when cfg is None or
        # enable is False. So whenever we return here, pending_gen_future is
        # None. If a future caller ever forwards a live future with resize
        # disabled, add a `_harvest_pending_generation` before these returns.
        cfg = OmegaConf.select(self.config.trainer, "dynamic_resize")
        if cfg is None:
            return
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if not cfg.get("enable", True):
            return
        schedule = cfg.get("schedule", []) or []
        if isinstance(schedule, dict):
            schedule = list(schedule.values())
        elif not isinstance(schedule, list):
            raise TypeError(f"trainer.dynamic_resize.schedule must be a list or dict, got {type(schedule).__name__}")

        harvested_raw = None

        # FIXME: this looped check seems heavy and unnecessary to do at every step
        for item in schedule:
            if not isinstance(item, dict):
                raise TypeError(
                    "Each trainer.dynamic_resize.schedule item must be a dict, "
                    f"got {type(item).__name__}: {item!r}"
                )
            if item.get("step") != self.global_steps:
                continue

            logger.info(
                "[one-step-off][resize] trigger: "
                f"global_step={self.global_steps}, item={item}"
            )

            gate_pass, required_action, gate_metrics = self._gate_dynamic_resize_schedule_item(item)
            self._latest_resize_control_metrics = gate_metrics
            if not gate_pass:
                logger.info(
                    "[one-step-off][resize][gate] skip scheduled resize: step=%s required_action=%s signal=%s "
                    "decision=%s ratio=%.4f cooldown_remaining=%s dwell_remaining=%s",
                    self.global_steps,
                    required_action,
                    gate_metrics.get("resize/hysteresis_signal"),
                    gate_metrics.get("resize/hysteresis_decision"),
                    float(gate_metrics.get("resize/rollout_train_ratio", 0.0)),
                    gate_metrics.get("resize/cooldown_remaining"),
                    gate_metrics.get("resize/dwell_remaining"),
                )
                continue

            actor_spec = item.get("actor_pool")
            rollout_spec = item.get("rollout_pool")
            release_old = item.get("release_old", True)
            detached = item.get("detached", False)
            actor_resume_path = item.get("actor_resume_from_path")

            if bool(item.get("hard_resize_destroy_all", False)):
                logger.info("[one-step-off][resize][hard] using destroy-all hard resize baseline path")
                # Hard-resize is not overlap-optimized this iteration. If a future
                # is in flight, await it BEFORE hard-resize tears engines down so
                # we never cross its shutdown/kill with a live generation task.
                if pending_gen_future is not None:
                    harvested_raw = await self._harvest_pending_generation(pending_gen_future)
                    pending_gen_future = None
                resize_applied = await self._hard_resize_actor_rollout_from_checkpoint(item, actor_resume_path=actor_resume_path)
                if resize_applied:
                    if self._resize_controller is not None:
                        snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
                        self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
                    logger.info("[one-step-off][resize][hard] switch complete")
                continue

            staged_enabled = self._should_use_staged_shared_pool_resize(item)
            if staged_enabled:
                if release_old:
                    logger.info(
                        "[one-step-off][resize] normalize release_old=true shared-pool split resize to staged path"
                    )
                logger.info("[one-step-off][resize] using staged shared-pool resize path")
                try:
                    resize_applied = await self._staged_resize_shared_pool(
                        item, actor_resume_path=actor_resume_path, pending_gen_future=pending_gen_future
                    )
                finally:
                    # Instrumentation state only. On the success path the post-sync
                    # probe already consumed the slot, so this is a no-op; on a raise
                    # it is what keeps a half-finished transition from outliving its
                    # own resize. The next schedule's entry clear is the backstop,
                    # not the guarantee.
                    self._pending_memory_transition = None
                harvested_raw = self._last_harvested_raw
                if resize_applied:
                    if self._resize_controller is not None:
                        snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
                        self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
                    logger.info("[one-step-off][resize] staged switch complete")
                continue

            actor_pool = self._resolve_role_pool(Role.Actor, actor_spec)
            rollout_pool = self._resolve_role_pool(Role.Rollout, rollout_spec)

            if pending_gen_future is not None:
                # Legacy non-staged switch path: not overlap-optimized. Await the
                # in-flight generation before switching so no engine is killed
                # under it.
                harvested_raw = await self._harvest_pending_generation(pending_gen_future)
                pending_gen_future = None

            # 先只做新 group 的 prepare，给“旧 worker 仍存活时提前准备”预留时序。
            new_actor = await self.add_role_group_async(
                Role.Actor,
                name=item.get("actor_group_name"),
                resource_pool=actor_pool,
                detached=detached,
                name_prefix=item.get("name_prefix"),
                prepare_only=True,
            )
            new_rollout = await self.add_role_group_async(
                Role.Rollout,
                name=item.get("rollout_group_name"),
                resource_pool=rollout_pool,
                detached=detached,
                name_prefix=item.get("name_prefix"),
                prepare_only=True,
            )
            await self._switch_actor_rollout_groups(
                new_actor,
                new_rollout,
                resume_from_path=actor_resume_path,
                release_old=release_old,
                actor_spec=actor_spec,
                rollout_spec=rollout_spec,
            )
            if self._resize_controller is not None:
                snapshot = self._resize_controller.mark_resize_applied(step=self.global_steps, action=required_action)
                self._latest_resize_control_metrics = self._build_resize_control_metrics(snapshot)
            logger.info(
                "[one-step-off][resize] switch complete: actor=%s, rollout=%s",
                len(getattr(new_actor, "workers", [])),
                len(getattr(new_rollout, "workers", [])),
            )

        # Catch-all harvest for decline paths. A scheduled resize can be
        # declined without ever awaiting the in-flight generation: the gate
        # rejects it (continue), the budget blocks the staged path (returns
        # False before its internal harvest), or no schedule item matches this
        # step. In every such case harvested_raw is still None here while a real
        # generation task is running. fit() nulls batch_data_future right after
        # this call, so without harvesting now the task is orphaned and its
        # batch discarded (and a duplicate generation is later launched on the
        # same engine). Harvest it here; branches that already harvested leave
        # harvested_raw non-None (or nulled the future), so this is a no-op for
        # them. No engine teardown happens on any decline path, so the await is
        # safely before any engine the generation depends on is destroyed.
        if pending_gen_future is not None and harvested_raw is None:
            harvested_raw = await self._harvest_pending_generation(pending_gen_future)
        return harvested_raw

    async def _switch_actor_rollout_groups(
        self,
        new_actor_wg: RayWorkerGroup,
        new_rollout_wg: RayWorkerGroup,
        *,
        resume_from_path: str | None = None,
        release_old: bool = False,
        actor_spec: dict | None = None,
        rollout_spec: dict | None = None,
    ) -> None:
        # Dynamic resize must switch actor and rollout as one pair. Updating them
        # one by one temporarily creates a mixed topology (new actor + old rollout
        # or the reverse), and that intermediate state can hang weight sync /
        # collective group setup. Stage both groups first, then publish them
        # together and rebuild the dependent managers once.
        old_actor_wg = self.actor_wg
        old_rollout_wg = self.rollout_wg

        # Phase 1: 收缩旧拓扑。
        # prepare 已经在旧 worker 存活时提前完成；如果当前策略允许释放旧 group，
        # 则优先在切换窗口开始时回收旧资源，再执行新 group 的 commit。
        if release_old:
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)

        # Phase 2: 提交新拓扑。
        # 这一阶段执行真正依赖最终 worker/runtime 状态的初始化。
        await self._commit_role_group_init_async(new_actor_wg, role=Role.Actor)
        await self._commit_role_group_init_async(new_rollout_wg, role=Role.Rollout)

        # Phase 3: 恢复/传递权重状态。
        # actor 先恢复可训练状态，再把最新参数视图传给 rollout。
        if resume_from_path is not None:
            logger.info("[one-step-off][resize] loading checkpoint into new actor")
            new_actor_wg.load_checkpoint(
                resume_from_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        else:
            logger.info(
                "[one-step-off][resize] new actor keeps its freshly initialized weights; "
                "rollout weight metadata will be attached afterwards"
            )

        weights_info = self._get_actor_weights_info(new_actor_wg)[0]
        self._set_actor_weights_info(new_rollout_wg, weights_info)

        # Prewarm the inter-role communicator on the new worker lifecycle before
        # publishing the new topology. The final switch can then be reduced to a
        # lightweight activation step instead of an in-window rendezvous.
        if self._dynamic_resize_async_comm_prewarm_enabled:
            self._prepare_weight_sync_group(
                new_actor_wg,
                new_rollout_wg,
                topology_key=self._candidate_topology_key(actor_spec=actor_spec, rollout_spec=rollout_spec),
                actor_spec=actor_spec,
                rollout_spec=rollout_spec,
            )

        # Phase 4: 发布新拓扑。
        # actor/rollout 必须作为一对同时对外可见，避免混合拓扑。
        self._switch_weight_sync_group(
            new_actor_wg,
            new_rollout_wg,
            actor_spec=actor_spec,
            rollout_spec=rollout_spec,
        )
        self._publish_active_topology_state(actor_spec=actor_spec, rollout_spec=rollout_spec)

        # Phase 5: 重建依赖最终拓扑的管理器和同步链路。
        # Rollout manager 和 sync group 都依赖最终 actor/rollout 对，因此必须在
        # publish 之后统一重建。
        with resize_trace_span(
            self._resize_trace_config,
            "resize_rollout_shutdown",
            step=self._trace_step(),
            lane="trainer_main",
        ):
            await self._shutdown_async_rollout_manager_async()
        with resize_trace_span(
            self._resize_trace_config,
            "resize_rollout_init",
            step=self._trace_step(),
            lane="trainer_main",
        ):
            await self._init_async_rollout_manager_async()
        await self._sync_rollout_weights_after_resize_async(handoff_path=resume_from_path)

        # Phase 6: 兼容旧策略的延后清理。
        # 当 release_old=False 时，保留旧语义：新拓扑发布完后再释放旧 group。
        if not release_old:
            self.remove_role_group(Role.Actor, old_actor_wg)
            self.remove_role_group(Role.Rollout, old_rollout_wg)
            self._cleanup_pending_weight_sync_groups()



    @staticmethod
    def _collect_rollout_direct_weight_load_metrics(results):
        metrics = {}
        if not results:
            return metrics
        numeric_fields = (
            "total_s",
            "prepare_inference_model_s",
            "load_pages_s",
            "h2d_s",
            "rollout_load_weights_s",
            "empty_cache_s",
            "page_count",
            "manifest_page_count",
            "empty_page_count",
            "loaded_weight_count",
            "loaded_bytes",
            "host_file_page_count",
            "disk_page_count",
            "host_page_count",
            "device_free_before_bytes",
            "device_total_before_bytes",
            "device_free_after_bytes",
            "device_total_after_bytes",
        )
        rollout_results = [item for item in results if isinstance(item, dict) and item.get("role") == "rollout"]
        if not rollout_results:
            return metrics
        metrics["rollout_direct_weight_load_worker_count"] = float(len(rollout_results))
        for field in numeric_fields:
            values = []
            for item in rollout_results:
                value = item.get(field)
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                metrics[f"rollout_direct_weight_load_{field}_max"] = max(values)
                metrics[f"rollout_direct_weight_load_{field}_mean"] = sum(values) / len(values)
        first = rollout_results[0]
        source = first.get("source")
        if source is not None:
            metrics["rollout_direct_weight_load_source"] = str(source)
        return metrics

    def _load_rollout_weights_from_handoff_model_pages(self, handoff_path: str):
        launch_started_at = time.monotonic()
        refs = self.rollout_wg.execute_all_async(
            f"{str(Role.Rollout)}_load_rollout_weights_from_handoff_model_pages",
            handoff_path,
            staging_config=self._make_runtime_handoff_staging_dict(
                effective_backend=self._handoff_staging_config.effective_backend()
            ),
        )
        wait_started_at = time.monotonic()
        results = ray.get(refs) if refs else []
        diagnostics = {
            "rollout_direct_weight_load_launch_s": float(wait_started_at - launch_started_at),
            "rollout_direct_weight_load_wait_s": float(time.monotonic() - wait_started_at),
        }
        diagnostics.update(self._collect_rollout_direct_weight_load_metrics(results))
        self._last_sync_rollout_weight_diagnostics = diagnostics
        self._update_resize_execution_metrics(**diagnostics)
        return results

    async def _sync_rollout_weights_after_resize_async(
        self,
        *,
        clear_kv_cache: bool = True,
        handoff_path: str | None = None,
    ) -> None:
        group_name = getattr(self, "_active_weight_sync_group_name", "actor_rollout")
        use_direct_staged_rollout_load = (
            handoff_path is not None
            and self._handoff_staging_config.effective_backend() == "pinned_cpu"
            and bool(getattr(self._handoff_staging_config, "stage_optimizer", False))
        )
        use_alt_src = (
            not use_direct_staged_rollout_load
            and self._handoff_staging_config.effective_backend() == "pinned_cpu"
            and not bool(getattr(self._handoff_staging_config, "stage_optimizer", False))
            and len(getattr(self.actor_wg, "workers", [])) > 1
        )
        src_rank = 1 if use_alt_src else 0
        sync_started_at = time.monotonic()
        if use_direct_staged_rollout_load:
            self._load_rollout_weights_from_handoff_model_pages(handoff_path)
        else:
            self.sync_rollout_weights(src_rank=src_rank if use_alt_src else None)
        sync_duration = time.monotonic() - sync_started_at
        if not bool(getattr(self._handoff_staging_config, "optimizer_full_state_restore", True)):
            self._start_optimizer_preload_after_weight_sync()
        kv_clear_duration = 0.0
        if clear_kv_cache and self.async_rollout_manager is not None:
            kv_started_at = time.monotonic()
            await self.async_rollout_manager.clear_kv_cache_async()
            kv_clear_duration = time.monotonic() - kv_started_at
        metrics = {
            "resize/post_switch_weight_sync_s": sync_duration,
            "resize/post_switch_kv_clear_s": kv_clear_duration,
            "resize/post_switch_weight_sync_src_rank": float(src_rank),
            "resize/post_switch_weight_sync_alt_src": 1.0 if use_alt_src else 0.0,
            "resize/post_switch_weight_sync_direct_staged_load": 1.0 if use_direct_staged_rollout_load else 0.0,
            "resize/post_switch_weight_sync_path": "handoff_model_pages" if use_direct_staged_rollout_load else "collective_broadcast",
        }
        for key, value in getattr(self, "_last_sync_rollout_weight_diagnostics", {}).items():
            metrics[f"resize/post_switch_{key}"] = value
        self._update_resize_execution_metrics(**metrics)
    @staticmethod
    def _collect_optimizer_preload_start_metrics(results):
        metrics = {}
        if not results:
            return metrics
        numeric_fields = ("started", "full_state_restore", "page_count", "queue_depth", "total_s")
        for field in numeric_fields:
            values = []
            for item in results:
                if isinstance(item, dict) and isinstance(item.get(field), (int, float)):
                    values.append(float(item[field]))
            if values:
                metrics[f"resize/optimizer_preload_start_{field}_max"] = max(values)
                metrics[f"resize/optimizer_preload_start_{field}_mean"] = sum(values) / len(values)
        return metrics

    def _start_optimizer_preload_after_weight_sync(self) -> None:
        if self._handoff_staging_config.effective_backend() != "pinned_cpu":
            return
        started_at = time.monotonic()
        try:
            with resize_trace_span(
                self._resize_trace_config,
                "optimizer_preload_start",
                step=self._trace_step(),
                lane="trainer_main",
                metadata={},
            ):
                refs = self.actor_wg.execute_all_async(f"{str(Role.Actor)}_start_pending_optimizer_preload")
                results = ray.get(refs) if refs else []
        except Exception as exc:
            self._update_resize_execution_metrics(
                **{
                    "resize/optimizer_preload_start_s": float(time.monotonic() - started_at),
                    "resize/optimizer_preload_start_failed": 1.0,
                    "resize/optimizer_preload_start_error": repr(exc),
                }
            )
            raise
        metrics = self._collect_optimizer_preload_start_metrics(results)
        metrics["resize/optimizer_preload_start_s"] = float(time.monotonic() - started_at)
        metrics["resize/optimizer_preload_start_failed"] = 0.0
        self._update_resize_execution_metrics(**metrics)


    def _post_load_checkpoint_for_switch(self) -> None:
        self.sync_rollout_weights()
        if self.async_rollout_manager is not None:
            self.async_rollout_manager.clear_kv_cache()

    async def _post_load_checkpoint_for_switch_async(self) -> None:
        self.sync_rollout_weights()
        if self.async_rollout_manager is not None:
            await self.async_rollout_manager.clear_kv_cache_async()


    @staticmethod
    def _collect_sync_rollout_weight_metrics(results):
        metrics = {}
        if not results:
            return metrics
        numeric_fields = (
            "total_s",
            "load_actor_gpu_s",
            "get_params_s",
            "prepare_inference_model_s",
            "init_collective_s",
            "alloc_s",
            "actor_copy_s",
            "actor_source_direct_count",
            "actor_source_direct_bytes",
            "broadcast_s",
            "rollout_load_weights_s",
            "empty_cache_s",
            "src_rank",
            "weight_bytes",
            "weight_count",
            "device_free_before_bytes",
            "device_total_before_bytes",
            "device_free_after_bytes",
            "device_total_after_bytes",
        )
        for role in ("actor", "rollout"):
            role_results = [item for item in results if isinstance(item, dict) and item.get("role") == role]
            if not role_results:
                continue
            metrics[f"sync_rollout_weights_{role}_worker_count"] = float(len(role_results))
            for field in numeric_fields:
                values = []
                for item in role_results:
                    value = item.get(field)
                    if isinstance(value, bool):
                        value = float(value)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                if values:
                    metrics[f"sync_rollout_weights_{role}_{field}_max"] = max(values)
                    metrics[f"sync_rollout_weights_{role}_{field}_mean"] = sum(values) / len(values)
            first = role_results[0]
            for field in ("group_name", "collective_world_size", "collective_initialized", "torch_world_size"):
                value = first.get(field)
                if isinstance(value, (int, float, bool)):
                    metrics[f"sync_rollout_weights_{role}_{field}"] = float(value)
                elif value is not None:
                    metrics[f"sync_rollout_weights_{role}_{field}"] = str(value)
        return metrics

    def sync_rollout_weights(self, src_rank: int | None = None, *, actor_first: bool = False):
        actor_workers = getattr(self.actor_wg, "workers", [])
        rollout_workers = getattr(self.rollout_wg, "workers", [])

        # Safety guard: actor and rollout must NOT share the same underlying Ray Actor.
        # Otherwise, broadcast will deadlock (rollout enters first and blocks the actor call).
        try:
            actor_ids = {getattr(w, "_actor_id", None) for w in actor_workers}
            rollout_ids = {getattr(w, "_actor_id", None) for w in rollout_workers}
            actor_ids.discard(None)
            rollout_ids.discard(None)
            overlap = actor_ids.intersection(rollout_ids)
        except Exception:  # pragma: no cover - best effort
            overlap = set()

        if overlap:
            raise RuntimeError(
                "Actor/Rollout worker groups overlap on the same Ray ActorID(s): "
                f"{list(overlap)[:6]} (showing up to 6). "
                "This will deadlock sync_rollout_weights. "
                "Please create actor/rollout as separate Ray Actors (no shared spawn/from_detached)."
            )

        launch_started_at = time.monotonic()
        rollout_refs = None
        actor_refs = None
        rollout_launch_s = 0.0
        actor_launch_s = 0.0

        def _launch_actor_sync():
            if src_rank is None:
                return self.actor_wg.sync_rollout_weights()
            return self.actor_wg.sync_rollout_weights(int(src_rank))

        def _launch_rollout_sync():
            if src_rank is None:
                return self.rollout_wg.sync_rollout_weights()
            return self.rollout_wg.sync_rollout_weights(int(src_rank))

        # NOTE: launch only fires async sync_rollout_weights RPCs and returns
        # ObjectRefs immediately; the real cross-worker weight transfer
        # blocks in the ray.get() waits below. The "sync_rollout_weights" trace
        # span therefore wraps those waits, NOT this launch phase. Wrapping the
        # launch alone would report only the launch overhead and hide the actual
        # sync cost.
        if actor_first:
            actor_launch_started_at = time.monotonic()
            actor_refs = _launch_actor_sync()
            actor_launch_s = time.monotonic() - actor_launch_started_at
            rollout_launch_started_at = time.monotonic()
            rollout_refs = _launch_rollout_sync()
            rollout_launch_s = time.monotonic() - rollout_launch_started_at
        else:
            rollout_launch_started_at = time.monotonic()
            rollout_refs = _launch_rollout_sync()
            rollout_launch_s = time.monotonic() - rollout_launch_started_at
            actor_launch_started_at = time.monotonic()
            actor_refs = _launch_actor_sync()
            actor_launch_s = time.monotonic() - actor_launch_started_at
        if rollout_refs is None and actor_refs is None:
            self._last_sync_rollout_weight_diagnostics = {
                "sync_rollout_weights_launch_s": time.monotonic() - launch_started_at,
                "sync_rollout_weights_rollout_launch_s": rollout_launch_s,
                "sync_rollout_weights_actor_launch_s": actor_launch_s,
                "sync_rollout_weights_wait_s": 0.0,
            }
            self._update_resize_execution_metrics(**self._last_sync_rollout_weight_diagnostics)
            return []
        rollout_ref_list = []
        actor_ref_list = []
        if rollout_refs is not None:
            rollout_ref_list.extend(rollout_refs if isinstance(rollout_refs, list) else [rollout_refs])
        if actor_refs is not None:
            actor_ref_list.extend(actor_refs if isinstance(actor_refs, list) else [actor_refs])

        with resize_trace_span(
            self._resize_trace_config,
            "sync_rollout_weights",
            step=self._trace_step(),
            lane="trainer_main",
            metadata={"actor_workers": len(actor_workers), "rollout_workers": len(rollout_workers)},
        ):
            rollout_wait_started_at = time.monotonic()
            rollout_results = ray.get(rollout_ref_list) if rollout_ref_list else []
            rollout_wait_s = time.monotonic() - rollout_wait_started_at
            actor_wait_started_at = time.monotonic()
            actor_results = ray.get(actor_ref_list) if actor_ref_list else []
            actor_wait_s = time.monotonic() - actor_wait_started_at
        diagnostics = {
            "sync_rollout_weights_launch_s": time.monotonic() - launch_started_at - rollout_wait_s - actor_wait_s,
            "sync_rollout_weights_rollout_launch_s": rollout_launch_s,
            "sync_rollout_weights_actor_launch_s": actor_launch_s,
            "sync_rollout_weights_rollout_wait_s": rollout_wait_s,
            "sync_rollout_weights_actor_wait_s": actor_wait_s,
            "sync_rollout_weights_wait_s": rollout_wait_s + actor_wait_s,
        }
        diagnostics.update(self._collect_sync_rollout_weight_metrics(rollout_results + actor_results))
        self._last_sync_rollout_weight_diagnostics = diagnostics
        self._update_resize_execution_metrics(**diagnostics)
        return rollout_results + actor_results

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def _take_pending_raw(self, pre_generated_raw: tuple | None, batch_data_future) -> tuple[tuple | None, object]:
        # Consume-side selection: a raw batch harvested before a resize takes
        # priority over awaiting the in-flight future. Returns (raw_or_None,
        # batch_data_future_unchanged). Caller clears pre_generated_raw after use.
        if pre_generated_raw is not None:
            return pre_generated_raw, batch_data_future
        return None, batch_data_future

    async def _async_gen_raw_batch(self, continuous_iterator) -> tuple | None:
        # Topology-agnostic generation half of `_async_gen_next_batch`. Produces
        # a fully-generated batch (prompts repeated by rollout.n, unioned with
        # rollout output) WITHOUT applying the topology-bound runtime batch plan,
        # balance, or reward launch. Those are deferred to
        # `_finalize_batch_for_current_topology` on the consuming step.
        self._ensure_actor_training_group_binding()
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in _async_gen_raw_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # The async agent/reward loop reuses non-tensor fields from the original
        # training batch. Some one-step-off datasets only provide prompt/answer
        # style fields and omit the reward-routing metadata expected by the
        # downstream reward loop worker. Patch the minimal fields here so the
        # compatibility fix stays local to the experimental trainer.
        self._maybe_patch_reward_metadata(batch.non_tensor_batch)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            with resize_trace_span(
                self._resize_trace_config,
                "generate_async_background",
                step=self._trace_step(),
                lane="trainer_async",
                metadata={"batch_size": len(batch.batch)},
            ):
                gen_batch_output = await self.async_rollout_manager.generate_sequences_async(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        return metrics, timing_raw, epoch, batch

    async def _async_gen_next_batch(self, continuous_iterator):
        """Call parameter synchronization and asynchronous sequence generation."""
        raw = await self._async_gen_raw_batch(continuous_iterator)
        if raw is None:
            return None
        metrics, timing_raw, epoch, batch = raw
        batch, future_reward = self._finalize_batch_for_current_topology(batch, metrics, timing_raw)
        return metrics, timing_raw, epoch, batch, future_reward

    def _finalize_batch_for_current_topology(
        self, batch: DataProto, metrics: dict, timing_raw: dict
    ) -> tuple[DataProto, object | None]:
        # Topology-bound tail of batch preparation. Split out of
        # `_async_gen_next_batch` so it can run on the CONSUMING step, after any
        # dynamic resize has updated `self._active_role_world_sizes` /
        # `self.actor_wg`. Running it here (not at generation time) is what keeps
        # a batch generated under the old topology correctly aligned to the new
        # one.

        # Dynamic-resize can change the active actor/critic topology between
        # steps. Any downstream path that reasons about the current DP shape
        # (for example `_balance_batch`) must therefore see a batch that has
        # already been aligned to the active runtime topology. Keep this local
        # to the one-step-off trainer so the shared PPO path remains unchanged.
        batch = self._apply_runtime_batch_plan(batch)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Launch individual reward computations as each generation completes
        future_reward = None
        if self.config.reward_model.launch_reward_fn_async:
            # Store the object reference and set up callback
            future_reward = self._launch_individual_rewards.remote(batch, self.config, self.tokenizer)

        return batch, future_reward

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        # Get generation results
        gen_batch_result = batch
        original_non_tensor_batch = batch.non_tensor_batch

        # Repeat non_tensor_batch to match the number of responses
        n = config.actor_rollout_ref.rollout.n
        repeated_non_tensor_batch = {}
        for key, value in original_non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(value, n, axis=0)

        # Split into individual responses with preserved non_tensor_batch
        responses_split = []
        for i in range(len(gen_batch_result)):
            response_data = gen_batch_result[i : i + 1]  # Get single response
            # Add repeated non_tensor_batch values
            for key in repeated_non_tensor_batch:
                response_data.non_tensor_batch[key] = repeated_non_tensor_batch[key][i : i + 1]
            responses_split.append(response_data)

        # Launch async reward computation
        reward_futures = [
            compute_reward_async.remote(response_data, config, tokenizer) for response_data in responses_split
        ]

        # Wait for results and combine
        results = ray.get(reward_futures)
        rewards_list = [r[0] for r in results]
        extras_list = [r[1] for r in results]

        combined_reward_tensor = torch.cat(rewards_list, dim=0)
        combined_extras_dict = {}
        if extras_list and extras_list[0]:
            for key in extras_list[0].keys():
                combined_extras_dict[key] = [d[key] for d in extras_list if key in d]

        return combined_reward_tensor, combined_extras_dict

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._ensure_actor_training_group_binding()

        # load checkpoint before doing anything
        self._load_checkpoint()

        # after load checkpoint sync rollout weights
        await self.async_rollout_manager.wake_up_async()
        self.sync_rollout_weights()
        await self.async_rollout_manager.clear_kv_cache_async()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()

        # Start the first asynchronous generation task.
        overlap_prefetch = self._is_overlap_prefetch_enabled()
        pre_generated_raw = None
        if overlap_prefetch:
            batch_data_future = asyncio.create_task(self._async_gen_raw_batch(continuous_iterator))
        else:
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))

        while batch_data_future is not None or pre_generated_raw is not None:
            self._ensure_actor_training_group_binding()
            self._reset_resize_execution_metrics()
            self._reset_communicator_cache_metrics()
            self._reset_resize_budget_metrics()
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps
            resize_scheduled_this_step = self._has_dynamic_resize_scheduled_at_step(self.global_steps)
            overlap_prefetch = self._is_overlap_prefetch_enabled()

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            with marked_timer("step", timing_raw):
                # wait for the previous batch
                with marked_timer("gen", timing_raw, color="red"):
                    with resize_trace_span(
                        self._resize_trace_config,
                        "generate_async_wait",
                        step=self.global_steps,
                        lane="trainer_main",
                        metadata={},
                    ):
                        if overlap_prefetch:
                            raw, batch_data_future = self._take_pending_raw(pre_generated_raw, batch_data_future)
                            if raw is not None:
                                pre_generated_raw = None
                            else:
                                raw = await batch_data_future
                            _metrics, _timing_raw, epoch, batch_raw = raw
                            batch, future_reward = self._finalize_batch_for_current_topology(
                                batch_raw, _metrics, _timing_raw
                            )
                        else:
                            _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
                        timing_raw.update(batch.meta_info["timing"])
                        timing_raw.update(_timing_raw)
                        metrics.update(_metrics)
                        batch.meta_info.pop("timing", None)

                # sync weights from actor to rollout only when we are about to
                # launch generation on the current rollout group. On a scheduled
                # resize step the old rollout group will be replaced before the
                # next generation, so this pre-train sync is pure overhead and
                # can also contend with resize staging/preload work.
                if not resize_scheduled_this_step or overlap_prefetch:
                    with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                        self.sync_rollout_weights()
                        await self.async_rollout_manager.clear_kv_cache_async()
                else:
                    timing_raw["sync_rollout_weights"] = 0.0

                # async next generation
                if not is_last_step and (not resize_scheduled_this_step or overlap_prefetch):
                    if overlap_prefetch:
                        batch_data_future = asyncio.create_task(self._async_gen_raw_batch(continuous_iterator))
                    else:
                        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
                    await asyncio.sleep(0)
                elif not is_last_step:
                    logger.info(
                        "[one-step-off][resize][prefetch] defer next async batch until resize completes: step=%s",
                        self.global_steps,
                    )
                    batch_data_future = None

                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(
                            data=batch, config=self.config, tokenizer=self.tokenizer
                        )
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # await asyncio.sleep(0) ensures:
                # Asynchronous tasks can start executing immediately
                # The event loop can handle other pending coroutines
                # Prevents computations in a certain phase from blocking the entire asynchronous workflow
                #
                # The purpose here is to ensure that after triggering
                # `self.async_rollout_manager.generate_sequences_async(gen_batch_output)`,
                # the subsequent relevant logic can proceed in a timely manner
                await asyncio.sleep(0)

                # Operating Mode Selection:
                # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                else:  # Recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                await asyncio.sleep(0)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                await asyncio.sleep(0)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                await asyncio.sleep(0)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Compute rollout correction: IS weights, rejection sampling, and metrics
                    # Only runs in decoupled mode (computes once per batch using stable π_old)
                    # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in batch.batch
                        and not bypass_recomputing_logprobs  # Only in decoupled mode
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                        # Compute IS weights, apply rejection sampling, compute metrics
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    self._maybe_record_resize_history_batch(batch)

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                await asyncio.sleep(0)

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        with resize_trace_span(
                            self._resize_trace_config,
                            "train_update_critic",
                            step=self.global_steps,
                            lane="trainer_train",
                            metadata={},
                        ):
                            critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                await asyncio.sleep(0)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with resize_trace_span(
                            self._resize_trace_config,
                            "train_update_actor",
                            step=self.global_steps,
                            lane="trainer_train",
                            metadata={},
                        ):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    resize_execution_actor_metrics = {
                        key: value
                        for key, value in actor_output_metrics.items()
                        if key in self._latest_resize_execution_metrics
                    }
                    if resize_execution_actor_metrics:
                        self._update_resize_execution_metrics(**resize_execution_actor_metrics)
                    async_optimizer_metrics = self._collect_deferred_optimizer_materialize_metrics(block=False)
                    if async_optimizer_metrics:
                        self._update_resize_execution_metrics(**async_optimizer_metrics)
                        metrics.update(async_optimizer_metrics)
                    metrics.update(actor_output_metrics)
                await asyncio.sleep(0)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            await asyncio.sleep(0)
            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
            await asyncio.sleep(0)

            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
            esi_close_to_expiration = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )
            metrics.update(self._record_resize_control_observation(timing_raw=timing_raw, batch=batch))
            # Check if the conditions for saving a checkpoint are met.
            # The conditions include a mandatory condition (1) and
            # one of the following optional conditions (2/3/4):
            # 1. The save frequency is set to a positive value.
            # 2. It's the last training step.
            # 3. The current step number is a multiple of the save frequency.
            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
            ):
                if esi_close_to_expiration:
                    print("Force saving checkpoint: ESI instance expiration approaching.")
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

            if overlap_prefetch and resize_scheduled_this_step and batch_data_future is not None:
                # Barrier sink: instead of draining generation BEFORE resize, hand
                # the in-flight raw-generation task to the resize path. It keeps
                # running through export + group teardown + actor kill and is
                # awaited at the common deadline inside _staged_resize_shared_pool
                # (after the actor is gone, before any rollout engine is killed).
                # The harvested topology-agnostic raw batch is finalized next step.
                logger.info(
                    "[one-step-off][resize][prefetch] sink harvest into resize (overlap export/teardown): step=%s",
                    self.global_steps,
                )
                pre_generated_raw = await self._maybe_dynamic_resize(pending_gen_future=batch_data_future)
                batch_data_future = None
            else:
                # dynamic resize of actor/rollout pool
                # NOTE: this may be executed asynchronously with _save_checkpoint()
                await self._maybe_dynamic_resize()
            metrics.update(self._latest_resize_control_metrics)
            metrics.update(self._latest_resize_execution_metrics)
            metrics.update(self._latest_communicator_cache_metrics)
            metrics.update(self._latest_resize_budget_metrics)

            if not is_last_step and batch_data_future is None and pre_generated_raw is None:
                logger.info(
                    "[one-step-off][resize][prefetch] start deferred async batch after resize: step=%s",
                    self.global_steps,
                )
                if overlap_prefetch:
                    batch_data_future = asyncio.create_task(self._async_gen_raw_batch(continuous_iterator))
                else:
                    batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
                await asyncio.sleep(0)

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(self._build_step_path_debug_metrics(timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            tracking_logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.actor_rollout_ref.actor, "profiler")
                and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_rollout_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                )

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            # this is experimental and may be changed/removed in the future
            # in favor of a general-purpose data buffer pool
            if hasattr(self.train_dataset, "on_batch_end"):
                # The dataset may be changed after each training batch
                self.train_dataset.on_batch_end(batch=batch)
