from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ResizeTraceConfig:
    enabled: bool = False
    path: str = ""
    run_name: str = ""
    flush: bool = False
    replica_rank: str | None = None


def build_resize_trace_config(config: Any) -> ResizeTraceConfig:
    from omegaconf import OmegaConf

    trace_cfg = None
    try:
        trace_cfg = OmegaConf.select(config, "trainer.dynamic_resize.trace")
    except Exception:
        trace_cfg = None
    if trace_cfg is None:
        try:
            trace_cfg = OmegaConf.select(config, "dynamic_resize.trace")
        except Exception:
            trace_cfg = None
    if trace_cfg is None:
        return ResizeTraceConfig()
    try:
        data = OmegaConf.to_container(trace_cfg, resolve=True) or {}
    except Exception:
        data = dict(trace_cfg) if isinstance(trace_cfg, dict) else {}
    path = os.path.abspath(os.path.expanduser(str(data.get("path", "")).strip())) if data.get("path") else ""
    enabled = bool(data.get("enable", False) and path)
    return ResizeTraceConfig(
        enabled=enabled,
        path=path,
        run_name=str(data.get("run_name", "")).strip(),
        flush=bool(data.get("flush", False)),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    if dataclass_is_instance(value):
        return _json_safe(asdict(value))
    return repr(value)


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def append_resize_trace(config: ResizeTraceConfig, payload: dict[str, Any]) -> None:
    # Best-effort by design: this runs from resize_trace_span's finally, which
    # sits on the vLLM engine cold-start critical path. A trace-write failure
    # (mkdir/open/write raising OSError, e.g. disk full or a bad path) must NOT
    # propagate -- otherwise the tracer would (a) abort engine rebuild and
    # (b) mask the original exception a traced block was already raising. An
    # instrumentation sink swallowing its own I/O errors is the correct
    # contract; we log once at warning level so silent data loss is visible.
    if not config.enabled or not config.path:
        return
    try:
        os.makedirs(os.path.dirname(config.path), exist_ok=True)
        record = {
            "kind": "span",
            "run_name": config.run_name,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **{str(k): _json_safe(v) for k, v in payload.items()},
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        fd = os.open(config.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            if config.flush:
                os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 - tracer must never crash the traced path
        import logging

        logging.getLogger(__name__).warning(
            "append_resize_trace failed to write span (event=%s); dropping record",
            payload.get("event"),
            exc_info=True,
        )


# Cap on the recorded exception message. A CUDA OOM message is ~300 chars, but a
# NCCL/Gloo failure carries a multi-KB C++ stack that would bloat every JSONL
# line. The attribution numbers ("GPU N has a total capacity of X / this process
# has Y memory in use") always sit at the head, so keep the head and mark the cut.
TRACE_ERROR_MESSAGE_MAX_CHARS = 2000


def _exception_message(exc: BaseException) -> str:
    """`str(exc)`, truncated, and never raising.

    Worth the defensiveness: this runs from resize_trace_span's except branch on
    the resize/cold-start critical path, and a third-party exception whose
    __str__ raises must not mask the original exception (same contract as
    append_resize_trace swallowing its own I/O errors).
    """
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - tracer must never crash the traced path
        return f"<unprintable {type(exc).__name__}>"
    if len(text) > TRACE_ERROR_MESSAGE_MAX_CHARS:
        return text[:TRACE_ERROR_MESSAGE_MAX_CHARS] + f"...<truncated {len(text)} chars>"
    return text


# NVTX range instrumentation for Nsight Systems.
# Independent of the JSONL trace (config.enabled): a whole-application nsys run
# may want the semantic range labels on the timeline WITHOUT writing JSONL, and
# vice versa. Gated by VERL_RESIZE_NVTX=1 so it is a byte-for-byte no-op by
# default (import + range push/pop only happen when explicitly enabled).
_NVTX_ENABLED: bool | None = None
_NVTX_FN = None


def _nvtx_available() -> bool:
    global _NVTX_ENABLED, _NVTX_FN
    if _NVTX_ENABLED is not None:
        return _NVTX_ENABLED
    if os.environ.get("VERL_RESIZE_NVTX", "") not in ("1", "true", "True"):
        _NVTX_ENABLED = False
        return False
    try:
        import torch

        # torch.cuda.nvtx.range_push/pop are CPU-side markers; they land on the
        # NVTX timeline even for the trainer driver (no active CUDA context /
        # no GPU work needed). This is exactly what lets Nsight label the
        # trainer's resize window as e.g. "resize_rollout_init".
        _NVTX_FN = torch.cuda.nvtx
        _NVTX_ENABLED = True
    except Exception:
        _NVTX_ENABLED = False
    return _NVTX_ENABLED


def _nvtx_push(event: str, step: int | None, lane: str | None) -> None:
    if _nvtx_available():
        try:
            label = event
            if step is not None:
                label = f"{event}#step{step}"
            if lane:
                label = f"{label}@{lane}"
            _NVTX_FN.range_push(label)
        except Exception:
            pass


def _nvtx_pop() -> None:
    if _nvtx_available():
        try:
            _NVTX_FN.range_pop()
        except Exception:
            pass


@contextlib.contextmanager
def resize_trace_span(
    config: ResizeTraceConfig,
    event: str,
    *,
    step: int | None = None,
    lane: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    # NVTX push happens regardless of JSONL enablement (see _nvtx_available).
    # This is the single choke point every resize span flows through, so one
    # push/pop pair here labels trainer_main + worker + replica spans on the
    # Nsight timeline at once.
    _nvtx_push(event, step, lane)
    if not config.enabled:
        try:
            yield
        finally:
            _nvtx_pop()
        return

    started_wall_ns = time.time_ns()
    started_mono_ns = time.monotonic_ns()
    exc_type = None
    exc_message = ""
    try:
        yield
    except BaseException as exc:  # pragma: no cover - passthrough with trace side effect
        # BaseException, not Exception: a Ctrl-C or SystemExit mid-resize used to
        # be logged as `ok`, hiding which span was open when the run was killed.
        exc_type = type(exc).__name__
        exc_message = _exception_message(exc)
        raise
    finally:
        _nvtx_pop()
        ended_wall_ns = time.time_ns()
        ended_mono_ns = time.monotonic_ns()
        append_resize_trace(
            config,
            {
                "event": event,
                "step": step,
                "lane": lane or "",
                "ts_start_ns": started_wall_ns,
                "ts_end_ns": ended_wall_ns,
                "dur_ms": round((ended_mono_ns - started_mono_ns) / 1_000_000.0, 3),
                "status": "error" if exc_type else "ok",
                "error_type": exc_type or "",
                "error_message": exc_message,
                "metadata": metadata or {},
            },
        )


COLD_START_RPC_ALLOWLIST: set[str] = {
    "init_worker",
    "init_device",
    "load_model",
    "get_kv_cache_spec",
    "determine_available_memory",
    "initialize_from_config",
    "compile_or_warm_up_model",
    "initialize_cache",
}


def resize_trace_config_from_env() -> ResizeTraceConfig:
    path = os.environ.get("VERL_RESIZE_TRACE_PATH", "")
    return ResizeTraceConfig(
        enabled=bool(path),
        path=path,
        run_name=os.environ.get("VERL_RESIZE_TRACE_RUN_NAME", ""),
        flush=False,
        replica_rank=os.environ.get("VERL_RESIZE_TRACE_REPLICA_RANK"),
    )
