#!/usr/bin/env python3
"""trace_utils unit tests. Stdlib-only (omegaconf is lazy in trace_utils)."""

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # repo root (test is at tests/experimental/one_step_off_policy/)
TU_PATH = REPO / "verl" / "experimental" / "one_step_off_policy" / "trace_utils.py"


def _load():
    spec = importlib.util.spec_from_file_location("trace_utils_m1_under_test", TU_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _clear_env():
    for k in ("VERL_RESIZE_TRACE_PATH", "VERL_RESIZE_TRACE_RUN_NAME", "VERL_RESIZE_TRACE_REPLICA_RANK"):
        os.environ.pop(k, None)


def test_env_helper_disabled_when_path_absent():
    _clear_env()
    tu = _load()
    cfg = tu.resize_trace_config_from_env()
    assert cfg.enabled is False
    assert cfg.path == ""
    assert cfg.replica_rank is None


def test_env_helper_reads_path_run_replica():
    os.environ["VERL_RESIZE_TRACE_PATH"] = "/tmp/x.jsonl"
    os.environ["VERL_RESIZE_TRACE_RUN_NAME"] = "m1_run"
    os.environ["VERL_RESIZE_TRACE_REPLICA_RANK"] = "2"
    try:
        tu = _load()
        cfg = tu.resize_trace_config_from_env()
        assert cfg.enabled is True
        assert cfg.path == "/tmp/x.jsonl"
        assert cfg.run_name == "m1_run"
        assert cfg.replica_rank == "2"
    finally:
        _clear_env()


def test_allowlist_exactly_cold_start_methods():
    tu = _load()
    expected = {
        "init_worker",
        "init_device",
        "load_model",
        "get_kv_cache_spec",
        "determine_available_memory",
        "initialize_from_config",
        "compile_or_warm_up_model",
        "initialize_cache",
    }
    assert tu.COLD_START_RPC_ALLOWLIST == expected
    for serving in ("execute_model", "sample_tokens", "add_lora", "profile", "sleep", "wake_up"):
        assert serving not in tu.COLD_START_RPC_ALLOWLIST, f"{serving} must not be timed"


def _run_all():
    test_env_helper_disabled_when_path_absent()
    test_env_helper_reads_path_run_replica()
    test_allowlist_exactly_cold_start_methods()
    print("test_trace_utils_m1: all passed")


if __name__ == "__main__":
    _run_all()
