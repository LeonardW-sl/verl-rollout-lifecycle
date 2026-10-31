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
"""Sleep-pool resize wiring: ordering guards, not behaviour tests.

Everything here is about WHERE the sleep/wake calls sit relative to the existing
resize steps. Each ordering constraint maps to a named guardrail, and violating
one produces a hang or a silently-wrong result rather than an exception:

  G3  sleeping an engine with in-flight generation is as fatal as killing it,
      so sleep must come AFTER _harvest_pending_generation
  D6  sleep -> actor expand -> wake must be strictly serial, both to avoid an
      OOM at larger model sizes and because measuring actor_release_s (V4) is a goal
  G4  rollout_wg must equal the awake set, since sync_rollout_weights
      broadcasts to the whole group
  G1  post-resize weights come only from the driver's single sync entry point

ray_trainer.py cannot be imported on CPU (ray/torch), hence AST.
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


def _method_node(name: str):
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OneStepOffRayTrainer":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return item
    raise RuntimeError(f"OneStepOffRayTrainer has no method {name}")


def _code_only(node) -> str:
    """Body without the docstring, so prose naming a token is not read as code."""
    clone = ast.parse(ast.unparse(node)).body[0]
    body = clone.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        clone.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(clone)
    return ast.unparse(clone)


def _line_of(src: str, needle: str) -> int:
    """First line index containing needle, or -1."""
    for i, line in enumerate(src.splitlines()):
        if needle in line:
            return i
    return -1


@pytest.fixture(scope="module")
def resize_src() -> str:
    return _code_only(_method_node("_staged_resize_shared_pool"))


@pytest.fixture(scope="module")
def sleep_branch_src() -> str:
    return _code_only(_method_node("_apply_sleep_pool_schedule"))


# --- the branch exists and is gated ------------------------------------------


def test_sleep_pool_branch_is_gated_on_the_master_switch(resize_src, sleep_branch_src):
    """Default-off must leave the survivor / full-rebuild paths byte-for-byte.

    The gate lives in the helper (it returns None when the pool is absent) and
    the resize path branches on that result, so assert both halves: the resize
    path must guard on the returned value rather than calling through
    unconditionally.
    """
    assert "_elastic_pool" in sleep_branch_src, "helper does not consult the pool"
    assert "return None" in sleep_branch_src, "helper must opt out when the pool is absent"
    assert "sleep_pool_topology is not None" in resize_src, (
        "resize path must skip the rebuild when the helper opts out"
    )


def test_schedule_helper_exists_and_is_async(sleep_branch_src):
    node = _method_node("_apply_sleep_pool_schedule")
    assert isinstance(node, ast.AsyncFunctionDef), "sleep/wake are awaited, so this must be async"


def test_helper_returns_early_when_pool_absent(sleep_branch_src):
    assert "return" in sleep_branch_src
    assert "_elastic_pool" in sleep_branch_src


# --- G3: sleep strictly after the harvest barrier ----------------------------


def test_sleep_happens_after_the_harvest_barrier(resize_src):
    """G3. Sleeping an engine mid-generation is as fatal as killing it,
    and the harvest is the barrier that guarantees no request is in flight.
    """
    harvest = _line_of(resize_src, "_harvest_pending_generation")
    sleep_call = _line_of(resize_src, "_apply_sleep_pool_schedule")
    assert harvest >= 0, "harvest barrier vanished from the resize path"
    assert sleep_call >= 0, "sleep-pool schedule is not wired into the resize path"
    assert sleep_call > harvest, (
        f"sleep-pool schedule at line {sleep_call} runs BEFORE the harvest barrier at {harvest} (G3 violation)"
    )


# --- G1: weights only through the single entry point -------------------------


def test_weight_sync_still_goes_through_the_single_entry_point(resize_src):
    """G1. wake_up maps empty memory under level=2, so the driver's broadcast is
    what fills it. A second weight path would race it."""
    assert "_sync_rollout_weights_after_resize_async" in resize_src


def test_helper_does_not_push_weights_itself(sleep_branch_src):
    """G1 again, from the other side: the helper wakes engines and stops there."""
    for forbidden in ("sync_rollout_weights", "update_weights", "_get_actor_params"):
        assert forbidden not in sleep_branch_src, (
            f"sleep-pool helper must not touch weights ({forbidden}); that is G1's job"
        )


# --- D6: strict serialization, and the spans that prove it -------------------


def test_sleep_precedes_wake_in_the_helper(sleep_branch_src):
    sleep_at = _line_of(sleep_branch_src, "sleep_groups")
    wake_at = _line_of(sleep_branch_src, "wake_groups")
    assert sleep_at >= 0 and wake_at >= 0
    assert sleep_at < wake_at, "D6 requires sleep to complete before wake begins"


def test_both_transitions_are_awaited(sleep_branch_src):
    """Fire-and-forget would defeat D6: the point is to WAIT so the actor's
    release latency (V4) is measurable instead of hidden under overlap."""
    assert "await" in sleep_branch_src
    tree = ast.parse(sleep_branch_src)
    awaited = {
        ast.unparse(node.value.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
    }
    assert any("sleep_groups" in name for name in awaited), f"sleep_groups not awaited: {awaited}"
    assert any("wake_groups" in name for name in awaited), f"wake_groups not awaited: {awaited}"


@pytest.mark.parametrize("span", ["pool/sleep_s", "pool/wake_s"])
def test_transition_spans_are_emitted(sleep_branch_src, span):
    """These spans are the evidence; a missing span means an unmeasured
    phase, which is how a moved cost gets reported as a saving."""
    assert span in sleep_branch_src


def test_reconcile_drives_the_transition(sleep_branch_src):
    """The pool must be reconciled against an AwakeTopology rather than the
    helper recomputing which groups should be awake -- that fork is exactly what
    awake_topology exists to prevent."""
    assert "reconcile" in sleep_branch_src


# --- G4: rollout_wg equals the awake set ------------------------------------


def test_group_is_rebuilt_from_the_awake_set(resize_src):
    """G4. sync_rollout_weights broadcasts to the whole group, so a sleeping
    member hangs it rather than merely missing an update."""
    assert "awake_rollout_workers" in resize_src or "awake_handles" in resize_src


def test_sleep_pool_builds_complete_replica_plan(resize_src):
    assert "build_reuse_plan" in resize_src
    assert "sleep_pool_reuse_replicas" in resize_src
    assert "sleep_pool_keep_old_ranks" in resize_src


def test_sleep_pool_preserves_leaving_replicas(resize_src):
    assert "keep_replica_ranks=sleep_pool_keep_old_ranks" in resize_src


def test_sleep_pool_reuses_manager_replicas(resize_src):
    assert "reuse_replicas=sleep_pool_reuse_replicas" in resize_src


def test_new_rollout_manager_is_awake_before_post_resize_weight_sync(resize_src):
    wake_at = _line_of(resize_src, "await self.async_rollout_manager.wake_up_async()")
    sync_at = _line_of(resize_src, "_sync_rollout_weights_after_resize_async")
    assert wake_at >= 0, "new rollout managers remain level-2 asleep after resize"
    assert sync_at >= 0, "post-resize weight sync vanished"
    assert wake_at < sync_at, "rollout weights are synced before the new manager is awake"


def test_sleep_pool_has_priority_over_axis1_survivor_branch(resize_src):
    assert "sleep_pool_topology is None and _survivor_base" in resize_src


def test_sleep_pool_computes_survivor_plan_even_without_axis1(resize_src):
    assert "self._dynamic_resize_worker_reuse_enabled or self._sleep_pool_enabled" in resize_src


def test_sleep_pool_emits_complete_reuse_metadata(resize_src):
    for field in ("awake_bundles", "reuse_ranks", "keep_old_ranks", "reuse_complete"):
        assert field in resize_src


def test_sleeping_handles_never_enter_the_worker_group(resize_src):
    """G4, the failure direction that matters.

    A presence check on awake_rollout_workers passes even if sleeping_workers is
    concatenated onto it, and that mutation is exactly the hang: the broadcast
    waits forever on an engine whose memory is released. So inspect the actual
    from_detached call and require its handle list to mention only the awake set.
    """
    tree = ast.parse(resize_src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("from_detached")
    ]
    sleep_pool_calls = [
        c
        for c in calls
        if any(
            kw.arg == "worker_handles" and "sleep_pool_topology" in ast.unparse(kw.value)
            for kw in c.keywords
        )
    ]
    assert sleep_pool_calls, "no from_detached call builds the group from the sleep-pool topology"
    for call in sleep_pool_calls:
        for kw in call.keywords:
            if kw.arg not in {"worker_handles", "worker_names"}:
                continue
            rendered = ast.unparse(kw.value)
            assert "sleeping_workers" not in rendered, (
                f"from_detached({kw.arg}=...) includes sleeping_workers, which hangs "
                f"the weight-sync broadcast (G4): {rendered}"
            )


def test_source_keeps_the_guardrail_anchors():
    """Fail loudly if the wiring is deleted while these tests remain."""
    src = _RAY_TRAINER.read_text(encoding="utf-8")
    for needle in (
        "_apply_sleep_pool_schedule",
        "_elastic_pool",
        "pool/sleep_s",
        "pool/wake_s",
        "resize/actor_release_s",
    ):
        assert needle in src, f"missing production anchor {needle}"
