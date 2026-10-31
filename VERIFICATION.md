# Verification of the public snapshot

This records checks executed for the sanitized August source snapshot. No server, Docker container, GPU workload, training, or multi-node experiment was launched for this delivery. No benchmark results or private experiment artifacts are published here.

## Behavior checks actually run

Python 3.12.3, pytest 9.1.1, from the repository root:

```bash
python3 -m pytest -q \
  tests/experimental/one_step_off_policy/test_awake_topology_on_cpu.py \
  tests/experimental/one_step_off_policy/test_elastic_engine_pool_on_cpu.py \
  tests/experimental/one_step_off_policy/test_skip_rollout_fsdp_shell_on_cpu.py \
  tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py
```

Result: **92 passed**, exit code 0. The README's three-file quickstart is included in this command. The additional suite exercises error-message preservation with synthetic attribution data, not actual machine/incident figures.

Scope: zone geometry, survivor mapping, sleeping-pool logic, pure-rollout shell-skip precedence, and exception-message handling. These checks do not establish integrated training behavior or numerical training equivalence.

## Direct algorithm smoke

A throwaway invocation loaded the actual portable modules by file path without importing the heavyweight verl runtime. With synthetic bundle IDs it exercised:

- expansion: an unchanged physical bundle retains its engine while moving to a new logical rank;
- shrink: an engine outside the target set is marked for removal;
- malformed TP partition: no reuse plan is returned;
- shell-skip precedence: a pure-rollout role can skip, but a role that trains always retains its shell.

All expected concrete mappings and predicate values were observed. No script or fake runtime was added to the repository.

## Source-preservation checks

All **386 production Python files** under `verl/` were compared structurally with the original snapshot after excluding docstrings. No executable syntax-tree differences or missing modules were found. The dynamic-resize YAML's non-comment content also matched.

Public-only changes are:

- removal of private experimental commentary, local-record references and infrastructure anecdotes;
- replacement of private error-message fixture figures with synthetic figures, plus two test-only descriptive strings;
- the three legacy shell launchers now require explicit `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, and `CKPT_DIR` instead of private cluster defaults;
- this README/documentation entry point and local-file ignore rules.

Original author/committer/coauthor history is preserved separately; the public repository is a new snapshot, not a rewritten original history. Existing upstream license and copyright notices are retained.

## Legacy launcher disclosure changes

The three scripts under `verl/experimental/one_step_off_policy/shell/` passed `bash -n`. Each was launched with one required location omitted, progressively supplying the preceding locations. All twelve cases stopped with the expected missing-variable error **before any directory creation or Python/training command**.

These are preflight/error-path checks only. They do not prove a successful training launch. You must provide your own model, prepared parquet data and writable output directory.

An inherited issue remains in `run_one_step_off_dynamic_resize.sh`: the trailing `+global_profiler.*` arguments are separated from the main command by a missing continuation. This legacy script is not a verified full-runtime quickstart and is not used by the portable entry point. Its unrelated profiler wiring was not changed as part of disclosure cleanup.

## What remains unverified

- Full-runtime imports and the complete legacy suite in a training environment.
- GPU sleep/wake memory lifetime, collective membership and integrated resizing.
- Loss/reward agreement, optimizer-state continuation, or performance effects.
- Multi-node use; this legacy sleep-pool geometry assumes a single node.
- Compatibility with any newer upstream or team baseline.

The original run environment has no immutable dependency lock. Package metadata is available, but it is not evidence that a modern installation reproduces the historical environment.

## Upstream context

Generic upstream examples retain their own algorithm citations, sample configurations and upstream documentation. They are baseline context, not experimental results of the August contribution. No private measured-results report, teammate identities, real server configuration or project publication-status statement is included in the public contribution documentation.
