import asyncio
import inspect
import types

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.one_step_off_policy.trace_utils import ResizeTraceConfig
from verl.trainer.ppo.utils import Role


def _fake_self_with_trainer_cfg(trainer_cfg: dict):
    # We exercise the unbound method against a lightweight stand-in so we do not
    # need Ray/GPU to construct a real trainer. Only `self.config.trainer` is read.
    fake = types.SimpleNamespace()
    fake.config = OmegaConf.create({"trainer": trainer_cfg})
    return fake


def test_overlap_prefetch_disabled_by_default_when_no_dynamic_resize():
    fake = _fake_self_with_trainer_cfg({})
    assert OneStepOffRayTrainer._is_overlap_prefetch_enabled(fake) is False


def test_overlap_prefetch_disabled_when_flag_absent():
    fake = _fake_self_with_trainer_cfg({"dynamic_resize": {"enable": True, "schedule": []}})
    assert OneStepOffRayTrainer._is_overlap_prefetch_enabled(fake) is False


def test_overlap_prefetch_enabled_when_flag_true():
    fake = _fake_self_with_trainer_cfg({"dynamic_resize": {"enable": True, "overlap_prefetch": True, "schedule": []}})
    assert OneStepOffRayTrainer._is_overlap_prefetch_enabled(fake) is True


def test_overlap_prefetch_false_when_flag_false():
    fake = _fake_self_with_trainer_cfg({"dynamic_resize": {"overlap_prefetch": False}})
    assert OneStepOffRayTrainer._is_overlap_prefetch_enabled(fake) is False


def test_overlap_prefetch_disabled_when_dynamic_resize_disabled():
    # overlap-prefetch is a dynamic-resize optimization; a disabled resize
    # feature disables it even if the flag is explicitly True. Mirrors
    # _has_dynamic_resize_scheduled_at_step's `enable` gate.
    fake = _fake_self_with_trainer_cfg({"dynamic_resize": {"enable": False, "overlap_prefetch": True}})
    assert OneStepOffRayTrainer._is_overlap_prefetch_enabled(fake) is False


def _make_raw_batch(num_samples: int, seqlen: int = 4) -> DataProto:
    # A minimal topology-agnostic batch as produced by the generate-only half:
    # prompts already repeated by rollout.n and unioned with gen output.
    input_ids = torch.arange(num_samples * seqlen).reshape(num_samples, seqlen)
    attention_mask = torch.ones(num_samples, seqlen, dtype=torch.long)
    responses = torch.arange(num_samples * seqlen).reshape(num_samples, seqlen)
    batch = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "responses": responses,
        }
    )
    batch.non_tensor_batch["uid"] = np.array([f"u{i}" for i in range(num_samples)], dtype=object)
    return batch


def _fake_self_for_finalize(actor_world_size: int):
    fake = types.SimpleNamespace()
    fake.config = OmegaConf.create(
        {
            "trainer": {"balance_batch": False},
            "reward_model": {"launch_reward_fn_async": False},
            "actor_rollout_ref": {"actor": {}},
            "critic": {},
        }
    )
    fake.use_critic = False
    fake.global_steps = 0
    fake._active_role_world_sizes = {Role.Actor: actor_world_size}
    fake._resize_padding_history = []
    fake._last_runtime_batch_plan = None
    # Bind the real methods we depend on onto the stand-in.
    fake._build_runtime_batch_plan = types.MethodType(OneStepOffRayTrainer._build_runtime_batch_plan, fake)
    fake._apply_runtime_batch_plan = types.MethodType(OneStepOffRayTrainer._apply_runtime_batch_plan, fake)
    fake._sample_padding_from_history = types.MethodType(OneStepOffRayTrainer._sample_padding_from_history, fake)
    return fake


def test_finalize_alignment_follows_current_actor_topology():
    # Same raw batch, two different active topologies -> different alignment.
    raw_a = _make_raw_batch(num_samples=12)
    raw_b = _make_raw_batch(num_samples=12)

    fake4 = _fake_self_for_finalize(actor_world_size=4)
    fake6 = _fake_self_for_finalize(actor_world_size=6)

    batch4, reward4 = OneStepOffRayTrainer._finalize_batch_for_current_topology(fake4, raw_a, metrics={}, timing_raw={})
    batch6, reward6 = OneStepOffRayTrainer._finalize_batch_for_current_topology(fake6, raw_b, metrics={}, timing_raw={})

    plan4 = batch4.meta_info["runtime_batch_plan"]
    plan6 = batch6.meta_info["runtime_batch_plan"]

    assert plan4["actor_world_size"] == 4
    assert plan6["actor_world_size"] == 6
    # This is the whole point: finalizing under topology 6 must NOT reuse a
    # plan computed under topology 4. Here world_size / micro-batch differ by
    # topology (mini-batch is 12 for both).
    assert plan6 != plan4
    # No async reward path in this config.
    assert reward4 is None
    assert reward6 is None


def test_finalize_sets_global_token_num_and_response_mask():
    raw = _make_raw_batch(num_samples=8)
    fake = _fake_self_for_finalize(actor_world_size=4)
    batch, _ = OneStepOffRayTrainer._finalize_batch_for_current_topology(fake, raw, metrics={}, timing_raw={})
    assert "global_token_num" in batch.meta_info
    assert "response_mask" in batch.batch.keys()


def test_gen_raw_batch_method_exists():
    assert hasattr(OneStepOffRayTrainer, "_async_gen_raw_batch")
    assert inspect.iscoroutinefunction(OneStepOffRayTrainer._async_gen_raw_batch)


def test_async_gen_next_batch_still_coroutine():
    # Off-path contract: still a coroutine returning the 5-tuple (asserted by
    # remote e2e; here we only guard the signature did not regress to sync).
    assert inspect.iscoroutinefunction(OneStepOffRayTrainer._async_gen_next_batch)


def test_take_pending_raw_prefers_cached_over_future():
    # pre_generated_raw present -> return it, signal no await needed.
    fake = types.SimpleNamespace()
    sentinel = ("m", "t", 0, "batch_raw")
    took, still_future = OneStepOffRayTrainer._take_pending_raw(
        fake, pre_generated_raw=sentinel, batch_data_future="FUTURE_OBJ"
    )
    assert took is sentinel
    assert still_future == "FUTURE_OBJ"  # future untouched, caller clears cache


def test_take_pending_raw_none_when_only_future():
    fake = types.SimpleNamespace()
    took, still_future = OneStepOffRayTrainer._take_pending_raw(
        fake, pre_generated_raw=None, batch_data_future="FUTURE_OBJ"
    )
    assert took is None
    assert still_future == "FUTURE_OBJ"


def _fake_self_for_harvest():
    fake = types.SimpleNamespace()
    fake._resize_trace_config = ResizeTraceConfig()  # disabled -> span is passthrough
    fake.global_steps = 3
    fake._trace_step = types.MethodType(OneStepOffRayTrainer._trace_step, fake)
    return fake


def test_harvest_returns_none_when_no_future():
    fake = _fake_self_for_harvest()
    result = asyncio.run(OneStepOffRayTrainer._harvest_pending_generation(fake, None))
    assert result is None


def test_harvest_awaits_and_returns_raw_tuple():
    fake = _fake_self_for_harvest()
    sentinel = ({"m": 1}, {"t": 2}, 0, "batch_raw")

    async def _gen():
        return sentinel

    async def _drive():
        task = asyncio.create_task(_gen())
        return await OneStepOffRayTrainer._harvest_pending_generation(fake, task)

    result = asyncio.run(_drive())
    assert result is sentinel


def test_harvest_propagates_generation_error():
    fake = _fake_self_for_harvest()

    async def _gen_boom():
        raise RuntimeError("gen failed")

    async def _drive():
        task = asyncio.create_task(_gen_boom())
        return await OneStepOffRayTrainer._harvest_pending_generation(fake, task)

    import pytest

    with pytest.raises(RuntimeError, match="gen failed"):
        asyncio.run(_drive())


def test_staged_resize_accepts_pending_gen_future_kwarg():
    sig = inspect.signature(OneStepOffRayTrainer._staged_resize_shared_pool)
    assert "pending_gen_future" in sig.parameters
    param = sig.parameters["pending_gen_future"]
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_staged_resize_awaits_before_rollout_teardown_in_source():
    # Source-order guard (CPU cannot run the full Ray method): the harvest await
    # MUST appear after remove_role_group(Role.Actor) and before BOTH
    # remove_role_group(Role.Rollout ...) call sites. This is the whole safety
    # invariant -- no engine generation depends on is killed while in flight.
    src = inspect.getsource(OneStepOffRayTrainer._staged_resize_shared_pool)
    harvest_idx = src.index("_harvest_pending_generation")
    actor_removed_idx = src.index("remove_role_group(Role.Actor")
    first_rollout_removed_idx = src.index("remove_role_group(Role.Rollout")
    assert actor_removed_idx < harvest_idx < first_rollout_removed_idx


def test_staged_resize_preclear_deferred_after_harvest_in_source():
    # KV-cache preclear must not clear an engine still decoding in-flight
    # generation. For the future-present path it is deferred to AFTER the
    # harvest await. NOTE: there are TWO preclear call sites (a flag-off one in
    # the export block that preserves original timing, and the deferred one).
    # Assert on the LAST occurrence (the deferred site) with rindex, since the
    # first occurrence is the guarded flag-off site that legitimately precedes
    # the harvest await.
    src = inspect.getsource(OneStepOffRayTrainer._staged_resize_shared_pool)
    harvest_idx = src.index("_harvest_pending_generation")
    last_preclear_idx = src.rindex("_clear_rollout_kv_cache_before_resize_async")
    assert harvest_idx < last_preclear_idx
    # the flag-off preclear stays guarded so its original pre-export timing is
    # preserved byte-for-byte when there is no in-flight future. Assert the FULL
    # guard binds the correct site (not just that the substring exists somewhere)
    # so a mis-wired guard cannot silently pass this test.
    assert "if pending_gen_future is None and self._handoff_staging_config.preclear_rollout_kv_cache:" in src
    # and the deferred site is guarded for the future-present case.
    assert "if pending_gen_future is not None and self._handoff_staging_config.preclear_rollout_kv_cache:" in src


def test_maybe_dynamic_resize_accepts_pending_gen_future_kwarg():
    sig = inspect.signature(OneStepOffRayTrainer._maybe_dynamic_resize)
    assert "pending_gen_future" in sig.parameters
    param = sig.parameters["pending_gen_future"]
    assert param.default is None


def test_maybe_dynamic_resize_forwards_future_and_returns_harvest_in_source():
    src = inspect.getsource(OneStepOffRayTrainer._maybe_dynamic_resize)
    # forwards the future into the staged path
    assert "pending_gen_future=pending_gen_future" in src
    # reads the stashed harvest and returns it
    assert "self._last_harvested_raw" in src


def test_hard_resize_awaits_future_before_crossing_in_source():
    # Out-of-scope for overlap, but must not cross hard-resize teardown with a
    # live future: the defensive await must appear BEFORE the hard-resize call.
    src = inspect.getsource(OneStepOffRayTrainer._maybe_dynamic_resize)
    assert "hard_resize_destroy_all" in src  # branch still present
    harvest_idx = src.index("_harvest_pending_generation(pending_gen_future)")
    hard_resize_idx = src.index("_hard_resize_actor_rollout_from_checkpoint")
    assert harvest_idx < hard_resize_idx


def test_fit_forwards_future_into_resize_not_pre_await():
    src = inspect.getsource(OneStepOffRayTrainer.fit)
    # fit assigns the resize return (harvested raw) into pre_generated_raw,
    # forwarding the in-flight future into resize -- the single rewired call.
    assert "pre_generated_raw = await self._maybe_dynamic_resize(pending_gen_future=batch_data_future)" in src
    # the OLD pre-resize drain (await batch_data_future then call resize bare)
    # must be gone: pre_generated_raw is no longer assigned by a bare future await.
    assert "pre_generated_raw = await batch_data_future" not in src


def test_fit_still_calls_resize_bare_when_no_overlap():
    # Non-overlap / non-resize steps still call resize with no future.
    src = inspect.getsource(OneStepOffRayTrainer.fit)
    assert "await self._maybe_dynamic_resize()" in src


def _fake_self_for_maybe_resize(*, schedule, gate_pass, global_steps=2):
    # Exercise the REAL _maybe_dynamic_resize + _harvest_pending_generation
    # coroutines (not string-matching). Only the gate decision, config, and
    # trace are stubbed; the catch-all harvest path runs for real.
    fake = types.SimpleNamespace()
    fake.config = OmegaConf.create({"trainer": {"dynamic_resize": {"enable": True, "schedule": schedule}}})
    fake.global_steps = global_steps
    fake._resize_trace_config = ResizeTraceConfig()  # disabled -> span passthrough
    fake._latest_resize_control_metrics = {}
    fake._last_harvested_raw = None
    fake._trace_step = types.MethodType(OneStepOffRayTrainer._trace_step, fake)
    fake._harvest_pending_generation = types.MethodType(OneStepOffRayTrainer._harvest_pending_generation, fake)
    fake._gate_dynamic_resize_schedule_item = lambda item: (gate_pass, "shrink", {})
    return fake


def test_maybe_dynamic_resize_catchall_harvests_on_gate_skip():
    # Gate declines the scheduled resize (normal hysteresis condition). The
    # in-flight future must still be harvested by the catch-all, not orphaned.
    fake = _fake_self_for_maybe_resize(schedule=[{"step": 2}], gate_pass=False)
    sentinel = ({"m": 1}, {"t": 2}, 0, "batch_raw")

    async def _gen():
        return sentinel

    async def _drive():
        task = asyncio.create_task(_gen())
        return await OneStepOffRayTrainer._maybe_dynamic_resize(fake, pending_gen_future=task)

    assert asyncio.run(_drive()) is sentinel


def test_maybe_dynamic_resize_catchall_harvests_on_empty_schedule():
    # No matching schedule item -> loop body never runs -> catch-all harvests.
    fake = _fake_self_for_maybe_resize(schedule=[], gate_pass=True)
    sentinel = ({"m": 1}, {"t": 2}, 0, "batch_raw")

    async def _gen():
        return sentinel

    async def _drive():
        task = asyncio.create_task(_gen())
        return await OneStepOffRayTrainer._maybe_dynamic_resize(fake, pending_gen_future=task)

    assert asyncio.run(_drive()) is sentinel


def test_maybe_dynamic_resize_none_future_still_returns_none():
    # Flag-off regression guard: no future in flight -> catch-all is a no-op,
    # method returns None (byte-for-byte the pre-barrier-sink behavior).
    fake = _fake_self_for_maybe_resize(schedule=[{"step": 2}], gate_pass=False)
    assert asyncio.run(OneStepOffRayTrainer._maybe_dynamic_resize(fake, pending_gen_future=None)) is None


def test_maybe_dynamic_resize_catchall_harvests_on_budget_block():
    # Gate passes and the staged path is selected, but the budget controller
    # blocks it: _staged_resize_shared_pool returns False BEFORE its internal
    # harvest (so _last_harvested_raw stays None). The catch-all must still
    # harvest the in-flight future rather than orphan it.
    fake = _fake_self_for_maybe_resize(schedule=[{"step": 2}], gate_pass=True)
    fake._should_use_staged_shared_pool_resize = lambda item: True

    async def _blocked_staged(item, *, actor_resume_path=None, pending_gen_future=None):
        # Budget-block: return False without touching _last_harvested_raw,
        # mirroring the early `return False` at the export/restore budget gates.
        return False

    fake._staged_resize_shared_pool = _blocked_staged
    fake._resize_controller = None
    sentinel = ({"m": 1}, {"t": 2}, 0, "batch_raw")

    async def _gen():
        return sentinel

    async def _drive():
        task = asyncio.create_task(_gen())
        return await OneStepOffRayTrainer._maybe_dynamic_resize(fake, pending_gen_future=task)

    assert asyncio.run(_drive()) is sentinel
