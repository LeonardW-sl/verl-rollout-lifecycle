#!/usr/bin/env python3
"""resize_trace_span must record the exception MESSAGE, not just its type.

Why this exists: a run died with `optimizer_materialize` -> OutOfMemoryError
and the trace recorded only `error_type: "OutOfMemoryError"`. Torch's CUDA OOM
message is the only place that carries "GPU N has a total capacity of X / this
process has Y memory in use" -- the two numbers whose difference tells you
whether a foreign process was on the card. Without them the post-mortem could
not distinguish "our own budget was too thin" from "someone else took the
card", and the run was unattributable.

Stdlib-only / CPU-only (omegaconf is lazy in trace_utils), matching
test_trace_utils_m1.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TU_PATH = REPO / "verl" / "experimental" / "one_step_off_policy" / "trace_utils.py"


def _load():
    spec = importlib.util.spec_from_file_location("trace_utils_errmsg_under_test", TU_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_span_and_read(tu, body, *, event="probe", flush=True):
    """Run `body()` inside a span against a temp trace file; return the record."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "trace.jsonl")  # sub/ also covers makedirs
        cfg = tu.ResizeTraceConfig(enabled=True, path=path, run_name="errmsg_test", flush=flush)
        raised = None
        try:
            with tu.resize_trace_span(cfg, event, lane="actor_rank1", metadata={"rank": 1}):
                body()
        except BaseException as exc:  # noqa: BLE001 - test asserts on the passthrough
            raised = exc
        lines = Path(path).read_text(encoding="utf-8").splitlines() if os.path.exists(path) else []
        assert len(lines) == 1, f"expected exactly one span record, got {len(lines)}"
        return json.loads(lines[0]), raised


def test_oom_message_is_recorded_with_the_decisive_numbers():
    """The real regression: a CUDA-OOM-shaped message must survive into the trace."""
    tu = _load()

    # Shape of torch's CUDA OOM text (this is what got lost). Values are synthetic.
    oom_text = (
        "CUDA out of memory. Tried to allocate 1.00 GiB. GPU 0 has a total capacity of "
        "10.00 GiB of which 256.00 MiB is free. Including non-PyTorch memory, this process "
        "has 4.00 GiB memory in use. Of the allocated memory 3.00 GiB is allocated by PyTorch"
    )

    class OutOfMemoryError(RuntimeError):
        pass

    def body():
        raise OutOfMemoryError(oom_text)

    record, raised = _run_span_and_read(tu, body, event="optimizer_materialize")

    assert isinstance(raised, OutOfMemoryError), "span must re-raise, never swallow"
    assert record["status"] == "error"
    assert record["error_type"] == "OutOfMemoryError"

    message = record["error_message"]
    # The whole point: the attribution numbers must be present.
    for fragment in ("total capacity of 10.00 GiB", "this process has 4.00 GiB memory in use"):
        assert fragment in message, f"missing {fragment!r} -- attribution still impossible"


def test_ok_span_has_empty_error_message():
    tu = _load()
    record, raised = _run_span_and_read(tu, lambda: None)
    assert raised is None
    assert record["status"] == "ok"
    assert record["error_type"] == ""
    assert record["error_message"] == ""


def test_long_message_is_truncated_but_keeps_the_head():
    """A NCCL/Gloo failure carries a multi-KB C++ stack; it must not bloat the JSONL.

    The decisive numbers are always at the head, so truncate the tail and mark it.
    """
    tu = _load()
    head = "CUDA out of memory. GPU 0 has a total capacity of 10.00 GiB"
    body_text = head + " " + ("x" * 50_000)

    def body():
        raise RuntimeError(body_text)

    record, _ = _run_span_and_read(tu, body)
    message = record["error_message"]

    assert message.startswith(head), "head of the message must be preserved verbatim"
    assert len(message) <= tu.TRACE_ERROR_MESSAGE_MAX_CHARS + 32, "message was not truncated"
    assert "truncated" in message, "truncation must be visible, not silent"


def test_broken_str_does_not_crash_the_traced_path():
    """An exception whose __str__ raises must not turn into a tracer crash.

    The tracer sits on the resize/cold-start critical path; it must degrade, and
    it must not mask the original exception (same contract as append_resize_trace).
    """
    tu = _load()

    class Hostile(RuntimeError):
        def __str__(self):
            raise ValueError("cannot stringify me")

    def body():
        raise Hostile()

    record, raised = _run_span_and_read(tu, body)

    assert isinstance(raised, Hostile), "original exception must still propagate"
    assert record["error_type"] == "Hostile", "type must still be recorded"
    assert isinstance(record["error_message"], str), "field must exist and stay a string"


def test_keyboard_interrupt_is_recorded_and_reraised():
    """BaseException (Ctrl-C / SystemExit) previously bypassed the error branch.

    A run killed mid-resize should show which span was open, not a bare `ok`.
    """
    tu = _load()

    def body():
        raise KeyboardInterrupt()

    record, raised = _run_span_and_read(tu, body)

    assert isinstance(raised, KeyboardInterrupt), "must re-raise"
    assert record["status"] == "error", "an interrupted span must not be logged as ok"
    assert record["error_type"] == "KeyboardInterrupt"


def _run_all():
    test_oom_message_is_recorded_with_the_decisive_numbers()
    test_ok_span_has_empty_error_message()
    test_long_message_is_truncated_but_keeps_the_head()
    test_broken_str_does_not_crash_the_traced_path()
    test_keyboard_interrupt_is_recorded_and_reraised()
    print("test_trace_span_error_message_on_cpu: all passed")


if __name__ == "__main__":
    _run_all()
