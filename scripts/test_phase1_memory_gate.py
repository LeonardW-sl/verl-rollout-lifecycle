"""The gate must reject the exact failure mode that shipped last round: spans
present, status ok, memory payload empty."""

from __future__ import annotations

import importlib.util
import json
import pathlib

_GATE = pathlib.Path(__file__).resolve().parent / "phase1_memory_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase1_memory_gate", _GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(group_index: int, tp_rank: int, **overrides) -> dict:
    sample = {
        "group_index": group_index,
        "tp_rank": tp_rank,
        "worker_rank": tp_rank,
        "memory_allocated_bytes": 1_000,
        "memory_reserved_bytes": 2_000,
        "memory_free_bytes": 10_000,
        "memory_total_bytes": 20_000,
        "actual_sleep_level": 2,
        "sampling_error": None,
    }
    sample.update(overrides)
    return sample


def _result(groups: list[int], tp: int, **overrides) -> dict:
    samples = [_sample(g, r) for g in groups for r in range(tp)]
    result = {
        "samples": samples,
        "expected_samples": len(samples),
        "observed_samples": len(samples),
        "error_count": 0,
        "sampling_rpc_ms": 5.0,
    }
    result.update(overrides)
    return result


def _fingerprint() -> dict:
    """Mirrors _get_memory_config_fingerprint() in ray_trainer.py."""
    return {
        "model_path": "/models/qwen2.5-0.5b-instruct",
        "gpu_memory_utilization": 0.6,
        "tensor_model_parallel_size": 2,
        "max_num_seqs": 64,
        "enforce_eager": False,
        "free_cache_engine": False,
    }


def _good_trace(tp: int = 2) -> list[dict]:
    # Metadata mirrors what ray_trainer.py really emits, including the
    # config_fingerprint / groups keys the deferred mutations drop.
    return [
        {
            "event": "memory_probe_startup",
            "status": "ok",
            "metadata": {
                "phase": "startup_baseline",
                "initial_k": 1,
                "awake_groups": [0],
                "sleeping_groups": [1],
                "awake": _result([0], tp),
                "sleeping": _result([1], tp),
                "config_fingerprint": _fingerprint(),
            },
        },
        {
            "event": "memory_probe_sleep",
            "status": "ok",
            "metadata": {
                "transition_id": "step2_k1_t1",
                "groups": [0],
                "schedule_k": 1,
                "sleep_before": _result([0], tp),
                "sleep_after": _result([0], tp),
                "config_fingerprint": _fingerprint(),
            },
        },
        {
            "event": "memory_probe_wake_mapped",
            "status": "ok",
            "metadata": {
                "transition_id": "step4_k3_t2",
                "groups": [1],
                "schedule_k": 3,
                "wake_mapped_after": _result([1], tp),
                "config_fingerprint": _fingerprint(),
            },
        },
        {
            "event": "memory_probe_wake_after_sync",
            "status": "ok",
            "metadata": {
                "transition_id": "step4_k3_t2",
                "groups": [1],
                "schedule_k": 3,
                "wake_after_sync": _result([1], tp),
                "config_fingerprint": _fingerprint(),
            },
        },
    ]


def _a_line_spans() -> list[dict]:
    """The startup-attribution spans. The gate must ignore them."""
    return [
        {"event": "elastic_group_build_wg", "status": "ok", "metadata": {"group_index": 0}},
        {"event": "elastic_group_prepare", "status": "ok", "metadata": {"group_index": 0}},
        {"event": "elastic_group_commit", "status": "ok", "metadata": {"group_index": 0}},
        {"event": "warmup_communicator_cache_keys", "status": "ok", "metadata": {}},
        {"event": "create_weight_sync_group_full", "status": "ok", "metadata": {}},
        {"event": "elastic_pool_launch_engines", "status": "ok", "metadata": {"groups": [0, 1]}},
        {"event": "elastic_pool_initial_sleep", "status": "ok", "metadata": {"groups": [1]}},
        {"event": "finalize_elastic_pool_replicas", "status": "ok", "metadata": {"initial_k": 1}},
    ]


def _mentions(reasons: list[str], needle: str) -> bool:
    return any(needle in reason for reason in reasons)


def test_accepts_a_complete_trace():
    gate = _load()
    assert gate.check_trace(_good_trace(), tp=2) == []


def test_rejects_empty_samples_even_when_status_is_ok():
    gate = _load()
    trace = _good_trace()
    trace[0]["metadata"]["awake"] = {
        "samples": [],
        "expected_samples": 0,
        "observed_samples": 0,
        "error_count": 0,
        "sampling_rpc_ms": 0.0,
    }
    reasons = gate.check_trace(trace, tp=2)
    assert reasons, "an empty payload must fail the gate"
    assert _mentions(reasons, "memory_probe_startup.awake"), reasons
    assert _mentions(reasons, "no samples recorded"), reasons


def test_rejects_missing_ranks():
    gate = _load()
    trace = _good_trace()
    trace[1]["metadata"]["sleep_after"]["samples"].pop()
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_sleep.sleep_after"), reasons
    assert _mentions(reasons, "at TP=2"), reasons


def test_rejects_sampling_errors():
    gate = _load()
    trace = _good_trace()
    trace[1]["metadata"]["sleep_after"]["samples"][0]["sampling_error"] = "RuntimeError: boom"
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_sleep.sleep_after[group=0,tp_rank=0]"), reasons
    assert _mentions(reasons, "RuntimeError: boom"), reasons


def test_rejects_impossible_driver_numbers():
    gate = _load()
    trace = _good_trace()
    trace[0]["metadata"]["awake"]["samples"][0]["memory_free_bytes"] = 99_999
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_startup.awake[group=0,tp_rank=0]"), reasons
    assert _mentions(reasons, "impossible free/total 99999/20000"), reasons


def test_rejects_missing_actual_sleep_level():
    gate = _load()
    trace = _good_trace()
    trace[2]["metadata"]["wake_mapped_after"]["samples"][0]["actual_sleep_level"] = None
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_wake_mapped.wake_mapped_after[group=1,tp_rank=0]"), reasons
    assert _mentions(reasons, "actual_sleep_level"), reasons


def test_rejects_unpaired_wake_records():
    gate = _load()
    trace = _good_trace()
    trace[3]["metadata"]["transition_id"] = "step9_k9_t9"
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "wake transition step4_k3_t2 has no post-sync sample"), reasons
    assert _mentions(reasons, "post-sync sample step9_k9_t9 has no matching wake_mapped record"), reasons


def test_requires_the_wake_after_sync_point_to_exist():
    gate = _load()
    trace = [span for span in _good_trace() if span["event"] != "memory_probe_wake_after_sync"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "missing required event memory_probe_wake_after_sync"), reasons


# --- deferred mutation escapes: only a real trace exposes these ---


def test_rejects_missing_config_fingerprint():
    """The post-sync span drops its config_fingerprint metadata key."""
    gate = _load()
    trace = _good_trace()
    del trace[3]["metadata"]["config_fingerprint"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_wake_after_sync"), reasons
    assert _mentions(reasons, "config_fingerprint"), reasons

    empty = _good_trace()
    empty[3]["metadata"]["config_fingerprint"] = {}
    empty_reasons = gate.check_trace(empty, tp=2)
    assert _mentions(empty_reasons, "config_fingerprint"), empty_reasons


def test_rejects_missing_groups_metadata():
    """The post-sync span drops its groups metadata key."""
    gate = _load()
    trace = _good_trace()
    del trace[3]["metadata"]["groups"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_wake_after_sync"), reasons
    assert _mentions(reasons, "groups"), reasons


def test_rejects_missing_startup_group_metadata():
    """Startup declares its groups under different keys."""
    gate = _load()
    trace = _good_trace()
    del trace[0]["metadata"]["sleeping_groups"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_startup"), reasons
    assert _mentions(reasons, "sleeping_groups"), reasons


def test_rejects_implausible_sampling_rpc_duration():
    """The sampler timeout is widened to 1e9, so a hung RPC never bounds."""
    gate = _load()
    trace = _good_trace()
    trace[3]["metadata"]["wake_after_sync"]["sampling_rpc_ms"] = 1e9
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_wake_after_sync.wake_after_sync"), reasons
    assert _mentions(reasons, "sampling_rpc_ms"), reasons

    nan = _good_trace()
    nan[3]["metadata"]["wake_after_sync"]["sampling_rpc_ms"] = float("nan")
    assert _mentions(gate.check_trace(nan, tp=2), "sampling_rpc_ms")


def test_ignores_a_line_spans():
    """The A-line spans share the trace file; they must not fail the gate."""
    gate = _load()
    trace = _a_line_spans() + _good_trace()
    assert gate.check_trace(trace, tp=2) == []


# --- the declared group list must agree with the groups actually sampled ---


def test_rejects_startup_awake_groups_that_the_samples_do_not_cover():
    """The docstring promised this and the code never checked it: a sample set
    covering the wrong groups read as complete."""
    gate = _load()
    trace = _good_trace()
    trace[0]["metadata"]["awake_groups"] = [0, 5]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons, "declared [0, 5] while only group 0 was sampled must fail"
    assert _mentions(reasons, "memory_probe_startup.awake"), reasons
    assert _mentions(reasons, "declared groups"), reasons


def test_rejects_post_sync_groups_that_the_samples_do_not_cover():
    gate = _load()
    trace = _good_trace()
    trace[3]["metadata"]["groups"] = [0, 1, 2, 3]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons, "declared [0, 1, 2, 3] while only group 1 was sampled must fail"
    assert _mentions(reasons, "memory_probe_wake_after_sync.wake_after_sync"), reasons
    assert _mentions(reasons, "declared groups"), reasons


def test_rejects_startup_sleeping_groups_declared_empty_while_samples_exist():
    gate = _load()
    trace = _good_trace()
    trace[0]["metadata"]["sleeping_groups"] = []
    reasons = gate.check_trace(trace, tp=2)
    assert reasons, "samples for group 1 with nothing declared must fail"
    assert _mentions(reasons, "memory_probe_startup.sleeping"), reasons
    assert _mentions(reasons, "declared groups"), reasons


def test_rejects_a_sleep_probe_whose_two_sides_sample_different_groups():
    """before/after must describe the same engines or the delta is meaningless."""
    gate = _load()
    trace = _good_trace()
    trace[1]["metadata"]["sleep_after"] = _result([3], 2)
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "memory_probe_sleep.sleep_after"), reasons
    assert _mentions(reasons, "declared groups"), reasons


def test_accepts_a_startup_where_nothing_is_sleeping():
    """No false-fail: k0 == num_groups means the sleeping side is legitimately empty."""
    gate = _load()
    trace = _good_trace()
    trace[0]["metadata"]["sleeping_groups"] = []
    trace[0]["metadata"]["sleeping"] = {
        "samples": [],
        "expected_samples": 0,
        "observed_samples": 0,
        "error_count": 0,
        "sampling_rpc_ms": 0.0,
    }
    assert gate.check_trace(trace, tp=2) == []


# --- probe spans must be correlated with their A-line twins ---


def _pool_span(event: str, transition_id: str, groups: list[int]) -> dict:
    """The A-line span emitted from the same branch as its memory probe."""
    return {
        "event": event,
        "status": "ok",
        "metadata": {"groups": groups, "schedule_k": 1, "transition_id": transition_id},
    }


def _paired_trace(tp: int = 2) -> list[dict]:
    return [
        _pool_span("pool_sleep", "step2_k1_t1", [0]),
        _pool_span("pool_wake", "step4_k3_t2", [1]),
        *_good_trace(tp),
    ]


def test_accepts_a_trace_whose_probes_pair_with_their_a_line_twins():
    gate = _load()
    assert gate.check_trace(_paired_trace(), tp=2) == []


def test_rejects_a_pool_sleep_with_no_memory_probe_sleep():
    """Sleep side: the headline B-line number silently disappears."""
    gate = _load()
    trace = [span for span in _paired_trace() if span["event"] != "memory_probe_sleep"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons, "a pool_sleep with no memory_probe_sleep must fail"
    assert _mentions(reasons, "step2_k1_t1"), reasons
    assert _mentions(reasons, "memory_probe_sleep"), reasons


def test_rejects_a_pool_wake_with_no_memory_probe_wake_mapped():
    gate = _load()
    trace = [span for span in _paired_trace() if span["event"] != "memory_probe_wake_mapped"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "step4_k3_t2"), reasons
    assert _mentions(reasons, "memory_probe_wake_mapped"), reasons


def test_accepts_a_schedule_that_never_sleeps():
    """No false-fail: requiring memory_probe_sleep unconditionally would break a
    schedule that only ever grows."""
    gate = _load()
    trace = [span for span in _paired_trace() if span["event"] not in {"pool_sleep", "memory_probe_sleep"}]
    assert gate.check_trace(trace, tp=2) == []


def test_a_line_twin_correlation_matches_on_the_transition_id_not_the_count():
    """One pool_sleep and one memory_probe_sleep that describe different transitions
    must not cancel each other out."""
    gate = _load()
    trace = _paired_trace()
    for span in trace:
        if span["event"] == "memory_probe_sleep":
            span["metadata"]["transition_id"] = "step7_k1_t7"
    reasons = gate.check_trace(trace, tp=2)
    assert reasons
    assert _mentions(reasons, "step2_k1_t1"), reasons


# --- transition_id is the pairing mechanism, so it cannot be optional ---


def test_rejects_probe_spans_without_a_transition_id():
    gate = _load()
    cases = (
        (1, "memory_probe_sleep"),
        (2, "memory_probe_wake_mapped"),
        (3, "memory_probe_wake_after_sync"),
    )
    for index, event in cases:
        trace = _good_trace()
        del trace[index]["metadata"]["transition_id"]
        reasons = gate.check_trace(trace, tp=2)
        assert reasons, f"{event} without a transition_id must fail"
        assert _mentions(reasons, event), reasons
        assert _mentions(reasons, "transition_id"), reasons


def test_dropping_transition_id_from_both_wake_spans_still_fails():
    """The pairing loops iterate over empty sets when both ids vanish, so the
    gate used to return no reasons at all."""
    gate = _load()
    trace = _good_trace()
    del trace[2]["metadata"]["transition_id"]
    del trace[3]["metadata"]["transition_id"]
    reasons = gate.check_trace(trace, tp=2)
    assert reasons, "losing the pairing mechanism entirely must not read as a clean run"
    assert _mentions(reasons, "transition_id"), reasons


def test_rejects_a_non_string_or_empty_transition_id():
    gate = _load()
    for bad in ("", 17, None):
        trace = _good_trace()
        trace[3]["metadata"]["transition_id"] = bad
        reasons = gate.check_trace(trace, tp=2)
        assert _mentions(reasons, "transition_id"), (bad, reasons)


# --- M-3: the gate must not depend on a human retyping the run's TP correctly ---


def test_tp_is_derived_from_the_traces_own_fingerprint():
    """ray_trainer.py builds ZoneLayout(tp=rollout.tensor_model_parallel_size) and
    records that same field in every config_fingerprint, so the trace knows its own
    TP and nobody has to remember it."""
    gate = _load()
    tp, reasons = gate.resolve_tp(_good_trace())
    assert reasons == [], reasons
    assert tp == 2


def test_tp_derivation_reports_disagreeing_fingerprints():
    """A TP that changed mid-run means the records are not comparable."""
    gate = _load()
    trace = _good_trace()
    trace[3]["metadata"]["config_fingerprint"] = dict(_fingerprint(), tensor_model_parallel_size=4)
    tp, reasons = gate.resolve_tp(trace)
    assert tp is None
    assert _mentions(reasons, "tensor_model_parallel_size"), reasons


def test_tp_derivation_reports_a_trace_that_carries_no_tp_at_all():
    gate = _load()
    trace = _good_trace()
    for span in trace:
        span["metadata"]["config_fingerprint"] = {"model_path": "/models/x"}
    tp, reasons = gate.resolve_tp(trace)
    assert tp is None
    assert reasons, "a trace with no TP anywhere cannot be gated silently"


def test_tp_derivation_rejects_an_implausible_tp():
    gate = _load()
    for bad in (0, -1, "2", 2.5):
        trace = _good_trace()
        for span in trace:
            span["metadata"]["config_fingerprint"] = dict(_fingerprint(), tensor_model_parallel_size=bad)
        tp, reasons = gate.resolve_tp(trace)
        assert tp is None and reasons, (bad, tp, reasons)


def test_cli_gates_a_trace_without_being_told_the_tp(tmp_path):
    """End to end: the documented invocation is `phase1_memory_gate.py <trace>`."""
    gate = _load()
    trace_path = tmp_path / "good.jsonl"
    trace_path.write_text("\n".join(json.dumps(span) for span in _good_trace()), encoding="utf-8")
    assert gate.main([str(trace_path)]) == 0

    bad = _good_trace()
    bad[0]["metadata"]["awake_groups"] = [0, 5]
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text("\n".join(json.dumps(span) for span in bad), encoding="utf-8")
    assert gate.main([str(bad_path)]) == 1


def test_cli_reports_an_explicit_tp_that_contradicts_the_trace(tmp_path):
    """Passing --tp 4 against a TP=2 run silently changed what the gate checked."""
    gate = _load()
    trace_path = tmp_path / "good.jsonl"
    trace_path.write_text("\n".join(json.dumps(span) for span in _good_trace()), encoding="utf-8")
    assert gate.main([str(trace_path), "--tp", "4"]) == 1
