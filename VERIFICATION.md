# Verification of the public snapshot

This records checks executed for the sanitized August source snapshot. No server, Docker container, GPU workload, training, or multi-node experiment was launched for this delivery. No benchmark results or private experiment artifacts are published here.

## Original publication behavior checks (2026-10-31)

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

## Public review and privacy update (2026-11-01)

The public packet now includes [REVIEW.md](docs/REVIEW.md) and
[WORK_HISTORY.md](docs/WORK_HISTORY.md). The root README distinguishes the
June–August development dates from the October publication date. The chronology
was checked against the 77 original contribution commits' metadata; no raw
private ledger, author details or commit messages were copied into it.

Checks exercised for this update:

- The documented three-file reviewer command completed: **87 passed**, exit
  code 0. No full training runtime was launched.
- The five main Markdown documents had **89 local links** checked, including
  linked heading anchors; all resolved. Their **three Bash blocks** passed
  syntax checks.
- The publication work set was enumerated: **928 files**, including the two
  new documents. A detailed candidate review covered credentials, personal
  identities and paths, access endpoints, private records, notebook output and
  metadata, and upstream versus private attribution.
- Before this update's commit, the published repository had **one branch, no
  tags and one reachable commit**. Its **895 unique file blobs** were decoded
  and screened; all **1,104 local Git objects** were enumerated, with **zero
  unreachable objects**. Commit identities/messages and tracked filenames were
  also inspected. The author email was a public-account privacy address.

Three types of personal-looking remnants from retained upstream examples were
removed from the current source: optional Git identity comments in two Docker
files; personal default index/corpus locations in the retrieval example; and
two editor metadata fields in a notebook. The notebook's **37 cells** and their
saved contents were unchanged. Upstream licenses, source attribution, public
maintainer/contact metadata, and container-created build directories were
retained; configuration syntax was not treated as an email address.

The retrieval example now requires explicit `--index_path` and `--corpus_path`.
Its actual argument-parser statements were exercised in isolation from the
heavy service imports: no locations or either location alone exit with code 2;
both supplied locations are accepted unchanged. Before the cleanup, the same
checks accepted missing locations and silently selected personal defaults.
No full retrieval service or model/index loading was exercised. There are no
repository launch callers relying on the removed defaults.

**History limitation:** the original publication commit already contained those
upstream example remnants. This update removes them from the current version,
not from earlier public copies or Git history. It is not a history rewrite or a
claim that previously published material can be recalled. The complete private
review repository and its internal attachments remain private.

The inspections found no confirmed user/team access data or live credentials
in the publication set. Pattern screening and human candidate review are not
a proof that every possible secret format or external link is safe.

## Bilingual extraction guide and comment correction (2026-11-02)

The [Chinese extraction guide](docs/EXTRACTION.zh-CN.md) and
[English counterpart](docs/EXTRACTION.md) now identify feature entry points,
portable-versus-integrated boundaries, one real caller, and a synthetic runnable
example. Original commit `4b39d5861` was checked against the preserved source
history: date 2026-08-22, an existing trace utility modified and its error-message
test added. This is a functional summary, not a publication of the private diff,
commit message, identities or experiment attachments. The original development
commit is not available for checkout or cherry-pick in this public snapshot.

The `ZoneLayout` class docstring now states the implemented `A | E | R` physical
order. Comparison of parsed source with docstrings excluded found no executable
syntax-tree change in `awake_topology.py`.

Checks exercised for this update:

- The four-suite behavior command documented above completed: **92 passed**,
  exit code 0.
- Both guides' identical Python examples were run separately against the actual
  trace module. Each observed the synthetic message, error type and error status,
  and asserted that the original exception object was re-raised. Each exited 0.
- The examples explicitly disabled optional GPU timeline instrumentation,
  printed only three selected fields, and automatically removed their temporary
  records. No machine metadata or raw runtime record was published.
- Local links and source-line anchors in the seven main public documents were
  checked; all resolved. Their five Bash blocks passed syntax checks.

No training, GPU workload, target-code migration or multi-node operation was
launched. The legacy launcher defect and earlier-public-history limitations
described above remain unchanged; this update does not claim to fix them.
