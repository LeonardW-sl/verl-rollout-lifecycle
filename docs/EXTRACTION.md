# Reading and feature extraction guide

English | [中文](EXTRACTION.zh-CN.md)

## 1. How to use this repository

This repository lets a reviewer and their coding assistant inspect a historical implementation, identify its invariants, and integrate a selected feature into another codebase. It is not the current training framework or a patch bundle to overwrite a target project wholesale.

The existing documents have separate jobs: the [README](../README.md) defines scope and flags, the [architecture](ARCHITECTURE.md) explains ownership and ordering, the [review guide](REVIEW.md) gives source entry points, and [verification](../VERIFICATION.md) records observed checks. This guide adds the missing extraction layer: what can be reused, what must be rewired, and how to demonstrate that the target behavior remains correct.

Start here, select one feature, and read its source and tests. Reading the entire training framework first is unnecessary.

## 2. Select the reading scope by feature

| Feature to extract | Read first | Extraction boundary |
|---|---|---|
| Failure details and stage tracing | [Trace utility](../verl/experimental/one_step_off_policy/trace_utils.py), [error-message tests](../tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py) | The trace module can be extracted independently; the target chooses which operations to wrap. The next section walks through one actual historical commit. |
| Identify engines that can survive a resize | [Geometry and engine diff](../verl/experimental/one_step_off_policy/awake_topology.py), [algorithm tests](../tests/experimental/one_step_off_policy/test_awake_topology_on_cpu.py) | The module imports only the Python standard library. The target must provide consistent physical GPU-bundle groups and ordering. A logical rank change does not imply a physical engine change. |
| Own sleeping rollout engines | [Sleep pool](../verl/experimental/one_step_off_policy/elastic_engine_pool.py), [pool tests](../tests/experimental/one_step_off_policy/test_elastic_engine_pool_on_cpu.py), [architecture](ARCHITECTURE.md) | The module is independently loadable, but actual execution depends on the target's remote worker handles, asynchronous completion and replica ownership. Copying one file does not establish a working GPU sleep lifecycle. |
| Skip an unused training shell on rollout-only processes | [Skip predicate](../verl/experimental/one_step_off_policy/shell_skip.py), [predicate tests](../tests/experimental/one_step_off_policy/test_skip_rollout_fsdp_shell_on_cpu.py), [worker integration](../verl/experimental/one_step_off_policy/fsdp_workers.py), [config bridge](../verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer_dynamic_resize.yaml) | The predicate can be extracted separately. Recheck whether the target process trains, whether any shell consumer remains, and whether the flag reaches the worker. Training processes must always retain the shell. |
| Integrate survivor retention, the sleep pool or generation overlap | [Trainer orchestration](../verl/experimental/one_step_off_policy/ray_trainer.py), [seven review units](REVIEW.md), [architecture](ARCHITECTURE.md) | This is a manual integration boundary, not a standalone module. Reimplement against the target's existing workers, weight synchronization and replica owner; do not overwrite its trainer with this file. |

Physical zone order is **permanent actor → elastic → permanent rollout**. Sleeping workers must not remain members of the active weight-sync collective; wake completion does not replace weight synchronization.

## 3. One actual commit: preserve failure messages

**Original development commit: `4b39d5861`, 2026-08-22.** This is the August source snapshot's final commit, not a new commit in this public repository. The public packet does not contain the original development history, so this identifier cannot be checked out or cherry-picked here. The links below point to the final public source and sanitized tests, not private original diffs or experiment attachments.

### What changed, and why start with it

Previously a failed stage recorded an exception's type but lost its detailed message. This change adds failure messages to the existing stage tracer and tests the boundary cases. It improves diagnostic completeness; it does not establish a speedup or training correctness.

| File | Responsibility of this change |
|---|---|
| [Trace utility](../verl/experimental/one_step_off_policy/trace_utils.py) | Safely stringify the exception; keep the first 2000 characters of long messages and append a truncation marker; emit `error_message`; record interrupts as errors; re-raise the original exception. |
| [Error-message tests](../tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py) | Check message preservation, empty fields after successful execution, truncation, exception stringification failures that do not mask the original error, and interrupt propagation. Numerical fixtures in the public tests are synthetic. |

Key names in the source: `ResizeTraceConfig` controls enablement and output location; `resize_trace_span` wraps an operation and records its result; `error_type` identifies the exception class; `error_message` preserves its detailed text.

Do not attribute the entire module to this commit: stage timing, file writing and optional GPU timeline markers already existed. The change completes failure information along that existing path.

### What to preserve when extracting it

1. If the target has no equivalent tracing facility, extract the complete `trace_utils.py` and retain applicable license notices; do not import the entire training package.
2. If the target already has one, migrate the behavior—message extraction, truncation, fields and original-exception propagation—into its existing tracing entry point instead of adding a parallel tracer.
3. Wrap the target's own resize or initialization operations. One actual public caller is the [training-state restore operation](../verl/experimental/one_step_off_policy/fsdp_workers.py#L2723-L2730): it wraps the operation in a trace span and supplies role metadata. The minimal call shape is shown below; this does not copy the legacy trainer.
4. Direct construction of the trace config needs no third-party config library. Calling `build_resize_trace_config` requires `omegaconf`; enabling `VERL_RESIZE_NVTX` attempts to load `torch` for GPU timeline markers. The example explicitly disables that optional path.
5. Trace-write failures must not mask business exceptions. The writer is best-effort, not a guaranteed audit sink.
6. Newly generated records include the current hostname, process ID and raw exception text. Messages may contain paths or secrets. The example prints only three safe fields and deletes its temporary record automatically. Inspect real records separately before publication.

### Directly runnable example

Execute the following Python code from this repository's root. It loads the actual module by file path without launching training or requiring a GPU. After migration, use the target project's normal module import.

```python
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["VERL_RESIZE_NVTX"] = "0"
source = Path("verl/experimental/one_step_off_policy/trace_utils.py")
spec = importlib.util.spec_from_file_location("trace_example", source)
trace = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace
spec.loader.exec_module(trace)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "trace.jsonl"
    config = trace.ResizeTraceConfig(enabled=True, path=str(path))
    original = RuntimeError("example resize failed")
    try:
        with trace.resize_trace_span(config, "example_resize"):
            raise original
    except RuntimeError as caught:
        assert caught is original
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "example resize failed"
    print({key: record[key] for key in ("status", "error_type", "error_message")})
```

Expected output:

```text
{'status': 'error', 'error_type': 'RuntimeError', 'error_message': 'example resize failed'}
```

The detailed message was retained, and the original exception was neither replaced nor swallowed. This is not a training, GPU-memory or multi-node proof.

Run the five existing behavior checks for this feature:

```bash
python3 -m pytest -q tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py
```

`pytest` is needed for that test command, not for the independent example above. After integration, rerun the boundary checks in the target and trigger its actual operation's failure path.

If extracting only the trace module and this test file, preserve their current relative directory layout or update the test's source-location logic. Copying the full training dependency tree is unnecessary.

## 4. Task to hand to a coding assistant

The following is a reading task, not a claim that migration is already complete.

> Read this feature extraction guide and its Chinese counterpart. Select one feature, then read its source, existing behavior checks and the target project's actual callers. Start read-only.
>
> Return: (1) the problem solved and behavior that must stay unchanged; (2) the minimal file set, dependencies and flags; (3) the target's existing owner and call sites to rewire; (4) directly reusable versus reimplemented pieces; (5) failure, cancellation and ordering risks; (6) target-runtime verification steps and remaining limits.
>
> Start with failure-message preservation from original commit `4b39d5861` if helpful. Public links show the final implementation, not an original commit that can be cherry-picked. Do not request or copy private ledgers, server configuration or experiment attachments. Do not overwrite the target trainer or label pure-algorithm checks as training reproduction. Before implementation, propose an integration plan grounded in the target's actual call sites.

## 5. Publication and verification boundaries

- The [work timeline](WORK_HISTORY.md) distinguishes development from publication dates; development history is not fabricated.
- Original commit identities, complete diffs and experiment materials remain private. This page publishes only one verified feature example and final-source entry points.
- The example uses synthetic error text and a temporary directory, not machine records or measured figures.
- [Verification](../VERIFICATION.md) records actual exercised results. The August code is not the current team line, and compatibility with arbitrary newer frameworks is not established.
