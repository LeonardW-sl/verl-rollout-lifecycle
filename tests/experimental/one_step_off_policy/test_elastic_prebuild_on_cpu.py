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
"""Elastic-zone prebuild: the spawn recipe, asserted structurally.

These are recipe guards: getting any element wrong does not fail loudly, it
silently puts two engines on one card or deadlocks the actor's later claim on
the same bundles.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_RAY_TRAINER = (
    pathlib.Path(__file__).resolve().parents[3]
    / "verl"
    / "experimental"
    / "one_step_off_policy"
    / "ray_trainer.py"
)


def _method_src(name: str) -> str:
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OneStepOffRayTrainer":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return ast.unparse(item)
    raise RuntimeError(f"OneStepOffRayTrainer has no method {name}")


def _top_level_func_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"top-level function {name} not found")


def _class_method_src(class_name: str, method_name: str) -> str:
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.unparse(item)
            raise RuntimeError(f"{class_name} has no method {method_name}")
    raise RuntimeError(f"class {class_name} not found")


def _load_scope_helper():
    node = _top_level_func_node("_scope_elastic_worker_env")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(_RAY_TRAINER), "exec"), namespace)
    return namespace["_scope_elastic_worker_env"]


def _load_trainer_method(name: str, namespace: dict):
    node = ast.parse(_method_src(name)).body[0]
    host = ast.ClassDef(name="_Trainer", bases=[], keywords=[], body=[node], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[host], type_ignores=[]))
    exec(compile(module, str(_RAY_TRAINER), "exec"), namespace)
    return getattr(namespace["_Trainer"], name)


@pytest.mark.parametrize(("rank", "selected"), [(0, "GPU-a"), (1, "GPU-b"), (3, "GPU-b")])
def test_elastic_worker_env_selects_one_physical_token(rank: int, selected: str):
    original = {
        "CUDA_VISIBLE_DEVICES": " GPU-a, GPU-b ",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    }

    scoped = _load_scope_helper()(original, rank)

    assert scoped is not original
    assert original["CUDA_VISIBLE_DEVICES"] == " GPU-a, GPU-b "
    assert scoped["CUDA_VISIBLE_DEVICES"] == selected
    assert scoped["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"
    assert "LOCAL_RANK" not in scoped


def test_single_card_worker_env_is_unchanged():
    original = {"CUDA_VISIBLE_DEVICES": "GPU-only"}
    assert _load_scope_helper()(original, 1) is original


def test_missing_worker_env_is_unchanged():
    assert _load_scope_helper()(None, 0) is None


def test_elastic_worker_group_delegates_actor_creation_to_core():
    src = _class_method_src("ElasticPoolRayWorkerGroup", "_create_worker")
    assert "_scope_elastic_worker_env" in src
    assert "super()._create_worker" in src
    for forbidden in ("ray_cls_with_init(", "update_options(", "placement_group="):
        assert forbidden not in src


def test_only_missing_elastic_groups_use_scoped_worker_group(prepare_src: str):
    assert "worker_group_cls=ElasticPoolRayWorkerGroup" in prepare_src
    build_src = _method_src("_build_role_group")
    assert "worker_group_cls" in build_src
    assert "worker_group_cls or self.ray_worker_group_cls" in build_src


def test_build_role_group_merges_trace_and_elastic_worker_env():
    trace_env = {
        "VERL_RESIZE_TRACE_PATH": "/tmp/resize.jsonl",
        "VERL_RESIZE_TRACE_RUN_NAME": "trace-run",
        "SHARED_KEY": "trace-value",
    }
    elastic_env = {
        "CUDA_VISIBLE_DEVICES": "4,5",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "SHARED_KEY": "elastic-value",
    }
    original_trace_env = dict(trace_env)
    original_elastic_env = dict(elastic_env)
    captured = {}

    class CapturingWorkerGroup:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def spawn(self, prefix_set):
            captured["prefix_set"] = prefix_set
            return {"rollout": "spawned-rollout"}

    namespace = {
        "Role": object,
        "RayWorkerGroup": object,
        "RayClassWithInitArgs": lambda **kwargs: kwargs,
        "create_colocated_worker_cls": lambda **kwargs: kwargs,
    }
    method = _load_trainer_method("_build_role_group", namespace)
    trainer = namespace["_Trainer"]()
    trainer.role_worker_mapping = {"rollout": object()}
    trainer.config = type("Config", (), {"actor_rollout_ref": object()})()
    trainer._get_resize_worker_group_kwargs = lambda: {"worker_env": trace_env}

    result = method(
        trainer,
        "rollout",
        resource_pool="elastic-pool",
        worker_env=elastic_env,
        worker_group_cls=CapturingWorkerGroup,
    )

    assert result == "spawned-rollout"
    assert captured["worker_env"] == {
        "VERL_RESIZE_TRACE_PATH": "/tmp/resize.jsonl",
        "VERL_RESIZE_TRACE_RUN_NAME": "trace-run",
        "CUDA_VISIBLE_DEVICES": "4,5",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "SHARED_KEY": "elastic-value",
    }
    assert captured["worker_env"] is not trace_env
    assert captured["worker_env"] is not elastic_env
    assert trace_env == original_trace_env
    assert elastic_env == original_elastic_env


def _code_only(src: str) -> str:
    """Strip the docstring so prose that *names* a forbidden token is not
    mistaken for code that *uses* it.

    Same failure mode as the prose above: the method's own docstring deliberately
    explains why split_resource_pool and RAY_EXPERIMENTAL_NOSET must not be
    used, so a naive substring match fails the correct implementation.
    """
    tree = ast.parse(src)
    fn = tree.body[0]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        fn.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


@pytest.fixture(scope="module")
def prepare_src() -> str:
    return _code_only(_method_src("_prepare_elastic_pool_workers"))


@pytest.fixture(scope="module")
def finalize_src() -> str:
    return _code_only(_method_src("_finalize_elastic_pool_replicas"))


def test_pool_is_built_with_zero_ray_gpu_quota(prepare_src):
    """Zero Ray GPU quota depends on this. With normal quota the cards stay booked
    while the engines sleep, so the actor's later expansion never gets them."""
    assert "use_gpu=False" in prepare_src


def test_pool_reuses_the_parent_placement_groups(prepare_src):
    """use_gpu=False alone drops the GPU:1 bundle entry (ray/base.py:128-131).
    The bundle survives only because the parent's pgs are passed in, which makes
    get_placement_groups early-return (:116-117)."""
    assert "placement_groups=" in prepare_src


def test_pool_does_not_use_split_resource_pool(prepare_src):
    """split_resource_pool copies use_gpu from the parent (ray/base.py:229), so
    it cannot produce a zero-quota child."""
    assert "split_resource_pool" not in prepare_src


def test_ray_zero_quota_clobber_is_disabled(prepare_src):
    """Ray 2.53 rewrites CUDA_VISIBLE_DEVICES="" at zero-GPU actor creation, so
    the injected value never arrives; this flag is what lets it reach the
    process. Precedent: pinned_cpu_staging.py:87."""
    assert "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO" in prepare_src


def test_noset_visible_devices_is_not_used(prepare_src):
    """Under zero quota worker.py:288 indexes an empty accelerator_ids list and
    raises IndexError at actor init."""
    assert "RAY_EXPERIMENTAL_NOSET" not in prepare_src


def test_each_group_gets_its_own_visible_devices(prepare_src):
    """The group-wide CVD is bootstrap input derived from that group's bundles.

    ElasticPoolRayWorkerGroup narrows it per worker before startup. LOCAL_RANK is
    unavailable: _create_worker never sets it (ray/base.py:541-549), worker_env is
    one dict per group, and both Worker.__init__ (base/worker.py:212) and vLLM's
    _init_worker (vllm_rollout.py:339-343) then compute local_rank=0, so both
    workers report local_rank_env="0". A hardcoded shared CVD would collide.
    """
    assert "CUDA_VISIBLE_DEVICES" in prepare_src
    assert "worker_env" in prepare_src
    # Must join the group's own bundles, not a literal card list.
    assert "join" in prepare_src and "bundles" in prepare_src
    # A hardcoded multi-card literal like "0,1,2,3" must not appear as the value.
    tree = ast.parse(prepare_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "," in node.value:
            # env flag values like "0" for RAY_ACCEL are fine; multi-card CVD literals are not
            parts = node.value.split(",")
            if all(p.strip().isdigit() for p in parts) and len(parts) > 1:
                raise AssertionError(
                    f"hardcoded multi-card CVD literal {node.value!r} found; "
                    "must derive from the group's own bundles"
                )


def test_prebuild_duration_is_recorded(finalize_src):
    """pool/prebuild_s is mandatory, not optional instrumentation: it is where the
    cold-start cost went. A resize-window win reported without it is a baseline
    misread rather than a result."""
    assert "pool/prebuild_s" in finalize_src


def test_missing_replicas_are_slept_after_launch(finalize_src):
    assert "replica.sleep" in finalize_src


def test_init_workers_finalizes_pool_after_manager_exists():
    src = _method_src("init_workers")
    assert src.index("self._init_models()") < src.index("self._init_async_rollout_manager()")
    assert src.index("self._init_async_rollout_manager()") < src.index(
        "self._finalize_elastic_pool_replicas()"
    )


def test_fit_wakes_rollout_before_initial_weight_sync():
    src = _method_src("fit")
    assert src.index("await self.async_rollout_manager.wake_up_async()") < src.index(
        "self.sync_rollout_weights()"
    )


def test_worker_prepare_does_not_sleep_or_launch_replicas(prepare_src):
    for forbidden in ("sleep_groups", "init_hybrid", "rollout_replica_class"):
        assert forbidden not in prepare_src


def test_prepare_adopts_active_workers_and_spawns_missing_groups(prepare_src):
    assert "awake_elastic_group_indices" in prepare_src
    assert "self.rollout_wg.workers" in prepare_src
    assert "if group_index in initial_awake" in prepare_src


def test_finalize_adopts_and_builds_replicas(finalize_src):
    assert "if self._elastic_pool is None" in finalize_src
    assert "self.async_rollout_manager.rollout_replicas" in finalize_src
    assert "self.async_rollout_manager.rollout_replica_class" in finalize_src
    assert "init_hybrid" in finalize_src
    assert "register_replica" in finalize_src


def test_prebuild_is_gated_on_the_master_switch():
    """Default-off must stay byte-for-byte: no pool, no spawn, no metric."""
    src = _RAY_TRAINER.read_text(encoding="utf-8")
    assert "_prepare_elastic_pool_workers" in src
    assert "_finalize_elastic_pool_replicas" in src
    prepare = _method_src("_prepare_elastic_pool_workers")
    finalize = _method_src("_finalize_elastic_pool_replicas")
    assert "_sleep_pool_layout" in prepare or "_sleep_pool_enabled" in prepare
    assert "return" in prepare
    assert "if self._elastic_pool is None" in finalize
