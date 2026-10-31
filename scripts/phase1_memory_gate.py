#!/usr/bin/env python3
"""Reject Phase 1 traces whose memory probe produced no usable data.

The failure this guards against is specific: the run completes, the spans exist,
every status is "ok", and every memory payload is empty because the sampling RPC
never reached a worker. Checking exit codes or span names cannot catch that.

Three further checks exist because a mutation audit of the CPU test suite found
they are the only remaining line of defence: dropping a probe span's
``config_fingerprint`` or ``groups`` metadata, and widening the sampler timeout so
a hung RPC is never bounded, all pass every source-level test and are only visible
in a real trace.

Usage: ``phase1_memory_gate.py <trace.jsonl>``. The TP is read from the trace's own
``config_fingerprint``; ``--tp`` only needs passing for a trace that carries none,
and is reported as a failure when it contradicts what the run recorded.
"""

from __future__ import annotations

import argparse
import json
import sys

_RESULT_KEYS = {
    "memory_probe_startup": (("awake", "awake_groups"), ("sleeping", "sleeping_groups")),
    "memory_probe_sleep": (("sleep_before", "groups"), ("sleep_after", "groups")),
    "memory_probe_wake_mapped": (("wake_mapped_after", "groups"),),
    "memory_probe_wake_after_sync": (("wake_after_sync", "groups"),),
}

# Which metadata key declares the groups a probe span covers. Startup names its
# two sets separately because it samples awake and sleeping groups in one span.
# Derived from _RESULT_KEYS so the declared list and the payload it guards cannot
# drift apart.
_GROUP_KEYS = {
    event: tuple(dict.fromkeys(group_key for _result_key, group_key in pairs))
    for event, pairs in _RESULT_KEYS.items()
}

# transition_id is the only thing that ties the records of one transition
# together, so it is required rather than optional on every span that has one.
_TRANSITION_EVENTS = ("memory_probe_sleep", "memory_probe_wake_mapped", "memory_probe_wake_after_sync")

# A-line span -> the B-line probe that must exist for the same transition_id.
# Correlating instead of hard-requiring the probe is what keeps a schedule that
# never shrinks from failing: no pool_sleep means no memory_probe_sleep is owed.
_A_LINE_TWINS = {
    "pool_sleep": "memory_probe_sleep",
    "pool_wake": "memory_probe_wake_mapped",
}

# The sampler's own ray.get timeout is 30s (ray_trainer.py _sample_pool_memory's
# timeout_s default). Anything past double that means the timeout did not bound
# the wait, so the reported duration is not trustworthy. test_sampler_timeout_
# default_is_pinned_and_the_gate_ceiling_tracks_it holds the two together.
MAX_SAMPLING_RPC_MS = 60_000.0

# ray_trainer.py builds the sleep pool's ZoneLayout with
# tp=config.actor_rollout_ref.rollout.tensor_model_parallel_size and records that
# same field in every probe span's config_fingerprint, so the trace already knows
# its own geometry.
_TP_KEY = "tensor_model_parallel_size"


def _is_finite_number(value: object) -> bool:
    """True for a real int/float that is neither NaN nor +/-inf.

    Written without ``math`` so the gate stays on the small stdlib set the
    analysis scripts are allowed: NaN fails every comparison, infinities fail the
    bounds, and bool is excluded because it is an int subclass.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float("-inf") < float(value) < float("inf")


def _sorted_groups(groups: set) -> list:
    """Stable ordering even when a malformed sample carries a None group_index."""
    return sorted(groups, key=repr)


def _check_result(label: str, result: object, tp: int, declared: object) -> list[str]:
    if not isinstance(result, dict):
        return [f"{label}: payload is not a dict"]
    samples = result.get("samples")
    if not isinstance(samples, list):
        return [f"{label}: no samples recorded"]

    declared_groups = set(declared) if isinstance(declared, list) else None
    if not samples:
        if declared_groups == set():
            # Nothing was meant to be sampled (k0 == num_groups leaves no
            # sleeping side); an empty payload is the correct answer.
            return []
        return [f"{label}: no samples recorded"]

    reasons: list[str] = []
    if result.get("error_count"):
        reasons.append(f"{label}: error_count={result['error_count']}")
    if result.get("observed_samples") != result.get("expected_samples"):
        reasons.append(
            f"{label}: observed={result.get('observed_samples')} != expected={result.get('expected_samples')}"
        )

    rpc_ms = result.get("sampling_rpc_ms")
    if rpc_ms is not None:
        if not _is_finite_number(rpc_ms):
            reasons.append(f"{label}: sampling_rpc_ms is not a finite number ({rpc_ms!r})")
        elif float(rpc_ms) > MAX_SAMPLING_RPC_MS:
            reasons.append(f"{label}: sampling_rpc_ms={rpc_ms} exceeds the {MAX_SAMPLING_RPC_MS} ms ceiling")

    groups = {sample.get("group_index") for sample in samples}
    if len(samples) != len(groups) * tp:
        reasons.append(
            f"{label}: expected {len(groups) * tp} records for {len(groups)} group(s) at TP={tp}, got {len(samples)}"
        )
    # The whole point of recording the group list: a sample set covering the
    # wrong groups is otherwise indistinguishable from a complete one.
    if declared_groups is not None and declared_groups != groups:
        reasons.append(
            f"{label}: declared groups {_sorted_groups(declared_groups)} "
            f"!= sampled groups {_sorted_groups(groups)}"
        )

    for sample in samples:
        where = f"{label}[group={sample.get('group_index')},tp_rank={sample.get('tp_rank')}]"
        if sample.get("sampling_error"):
            reasons.append(f"{where}: {sample['sampling_error']}")
            continue
        free_bytes = sample.get("memory_free_bytes")
        total_bytes = sample.get("memory_total_bytes")
        if free_bytes is None or total_bytes is None:
            reasons.append(f"{where}: driver view missing")
        elif not 0 <= free_bytes <= total_bytes:
            reasons.append(f"{where}: impossible free/total {free_bytes}/{total_bytes}")
        if sample.get("memory_allocated_bytes") is None or sample.get("memory_reserved_bytes") is None:
            reasons.append(f"{where}: torch view missing")
        if sample.get("actual_sleep_level") is None:
            reasons.append(f"{where}: actual_sleep_level not reported")
    return reasons


def _check_metadata(event: str, metadata: dict) -> list[str]:
    """Check the span-level keys that make a payload interpretable after the run.

    Without the fingerprint the numbers cannot be compared across runs, and
    without the group list there is no record of which groups were meant to be
    sampled -- so a sample set covering the wrong groups would read as complete.
    """
    reasons: list[str] = []
    fingerprint = metadata.get("config_fingerprint")
    if not fingerprint:
        reasons.append(f"{event}: config_fingerprint missing or empty")

    if event in _TRANSITION_EVENTS:
        transition_id = metadata.get("transition_id")
        if not isinstance(transition_id, str) or not transition_id:
            reasons.append(f"{event}: transition_id missing or not a non-empty string ({transition_id!r})")

    for key in _GROUP_KEYS.get(event, ()):
        if key not in metadata:
            reasons.append(f"{event}: {key} metadata missing")
        elif not isinstance(metadata[key], list):
            reasons.append(f"{event}: {key} is not a list ({type(metadata[key]).__name__})")
    return reasons


def check_trace(spans: list[dict], tp: int) -> list[str]:
    reasons: list[str] = []
    seen: dict[str, int] = {}
    probe_ids: dict[str, set[str]] = {event: set() for event in _TRANSITION_EVENTS}
    a_line_ids: dict[str, set[str]] = {event: set() for event in _A_LINE_TWINS}

    for span in spans:
        event = span.get("event")
        metadata = span.get("metadata") or {}
        if event in _A_LINE_TWINS:
            transition_id = metadata.get("transition_id")
            if isinstance(transition_id, str) and transition_id:
                a_line_ids[event].add(transition_id)
            continue
        if event not in _RESULT_KEYS:
            continue
        seen[event] = seen.get(event, 0) + 1
        reasons.extend(_check_metadata(event, metadata))
        for result_key, group_key in _RESULT_KEYS[event]:
            reasons.extend(
                _check_result(f"{event}.{result_key}", metadata.get(result_key), tp, metadata.get(group_key))
            )
        transition_id = metadata.get("transition_id")
        if event in probe_ids and isinstance(transition_id, str) and transition_id:
            probe_ids[event].add(transition_id)

    for required in ("memory_probe_startup", "memory_probe_wake_after_sync"):
        if required not in seen:
            reasons.append(f"missing required event {required}")

    # Every A-line transition owes a B-line record. This is what catches a sleep
    # probe that was skipped or dead-coded, without false-failing a schedule that
    # legitimately never sleeps (no pool_sleep span means nothing is owed).
    for a_line_event, probe_event in _A_LINE_TWINS.items():
        for missing in sorted(a_line_ids[a_line_event] - probe_ids[probe_event]):
            reasons.append(f"{a_line_event} transition {missing} has no {probe_event} sample")

    mapped_ids = probe_ids["memory_probe_wake_mapped"]
    synced_ids = probe_ids["memory_probe_wake_after_sync"]
    for unpaired in sorted(mapped_ids - synced_ids):
        reasons.append(f"wake transition {unpaired} has no post-sync sample")
    for orphan in sorted(synced_ids - mapped_ids):
        reasons.append(f"post-sync sample {orphan} has no matching wake_mapped record")
    return reasons


def resolve_tp(spans: list[dict]) -> tuple[int | None, list[str]]:
    """Read the TP the run actually used out of the trace's own fingerprints.

    A wrong ``--tp`` does not error, it changes what "complete" means: at TP=2 a
    gate told TP=1 accepts half the ranks as a full sample set. Deriving the value
    from the record removes the only gate input a human could get wrong.
    """
    seen: dict[str, object] = {}
    for span in spans:
        if span.get("event") not in _RESULT_KEYS:
            continue
        fingerprint = (span.get("metadata") or {}).get("config_fingerprint")
        if isinstance(fingerprint, dict) and _TP_KEY in fingerprint:
            # Keyed by repr so an unhashable (malformed) value cannot raise here.
            seen.setdefault(repr(fingerprint[_TP_KEY]), fingerprint[_TP_KEY])

    if not seen:
        return None, [f"no probe span carries config_fingerprint.{_TP_KEY}; pass --tp explicitly"]
    if len(seen) > 1:
        return None, [f"config_fingerprint.{_TP_KEY} disagrees across spans: {sorted(seen)}"]

    (value,) = seen.values()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None, [f"config_fingerprint.{_TP_KEY} is not a positive int ({value!r})"]
    return value, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="path to the resize trace JSONL")
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="tensor parallel size per elastic group; derived from the trace's own config_fingerprint when omitted",
    )
    args = parser.parse_args(argv)

    spans = []
    with open(args.trace, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                spans.append(json.loads(line))

    if args.tp is None:
        tp, reasons = resolve_tp(spans)
        if tp is None:
            print(f"FAIL: cannot determine TP for {args.trace}")
            for reason in reasons:
                print(f"  - {reason}")
            return 1
    else:
        tp = args.tp
        derived, _ = resolve_tp(spans)
        reasons = []
        if derived is not None and derived != tp:
            # Silently gating at the operator's TP instead of the run's is how a
            # half-sampled trace gets a PASS.
            reasons.append(f"--tp {tp} contradicts the trace's own config_fingerprint ({_TP_KEY}={derived})")

    reasons.extend(check_trace(spans, tp))
    if reasons:
        print(f"FAIL: {len(reasons)} problem(s) in {args.trace} (TP={tp})")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print(f"PASS: memory probe data complete in {args.trace} (TP={tp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
