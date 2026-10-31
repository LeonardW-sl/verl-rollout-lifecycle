# verl rollout-lifecycle runtime — August 2026 snapshot

This repository is a preserved snapshot of an experimental
[verl](https://github.com/volcengine/verl) research runtime. It lets a reader
inspect the rollout-lifecycle logic, its deployment model, and the tests that
pin that logic without access to private experiment infrastructure.

It is a **fixed historical snapshot**, not a maintained fork. With the August
feature flags left off, the code follows the pre-August research baseline. The
new behaviour described below is reachable only through explicit flags.

## Scope: what this is and is not

- **It is** the full legacy runtime package, with its packaging, examples,
  scripts and tests, exported at the final August 2026 commit.
  The main logic is in
  [`verl/experimental/one_step_off_policy/`](verl/experimental/one_step_off_policy/),
  with tests under
  [`tests/experimental/one_step_off_policy/`](tests/experimental/one_step_off_policy/).
  Tracing and rollout-engine integration touch a small number of adjacent files.
- **It is not** current upstream verl, the current team development line, or a
  port of this work into a newer baseline. In particular, the later
  direct-reshard and full-training-state work from other lines is **not** here.
- **It is not** a benchmark report. This repository makes no performance claims.
  [`VERIFICATION.md`](VERIFICATION.md) lists the checks that were run for this
  public snapshot.
- **It is not** a compatibility promise. A newer verl baseline may have
  restructured the trainer and worker classes this snapshot depends on.

## The August contribution

The work adds an explicit lifecycle for rollout engines in a dynamic-resize
setup: engines can be retained across a resize, put to sleep instead of killed,
and woken into a topology that is computed from one source of truth. Around that
core, the contribution adds an optional overlap path and instrumentation that
makes the resize path observable.

1. **One truth of the active topology.**
   [`awake_topology.py`](verl/experimental/one_step_off_policy/awake_topology.py)
   owns the zone geometry (`ZoneLayout`), the immutable per-schedule snapshot
   (`AwakeTopology`), and the single engine-diff algorithm
   (`partition_engines`, `diff_engines`). Consumers read the snapshot; they do
   not each recompute which engines are awake. See
   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#1-deployment-geometry-and-the-one-topology-truth).

2. **Split-role asleep/awake pool.**
   [`elastic_engine_pool.py`](verl/experimental/one_step_off_policy/elastic_engine_pool.py)
   owns the elastic-zone worker handles and the live engine replicas while they
   are asleep. It reconciles the current awake set against a target topology,
   sleeps and wakes whole TP groups, and produces the target-rank reuse map.

3. **Survivor engine retention.**
   [`ray_trainer.py`](verl/experimental/one_step_off_policy/ray_trainer.py)
   keeps rollout worker processes whose GPU bundle set is unchanged across a
   resize instead of killing and respawning them. A pure-shrink path and an
   opt-in mixed path (survivors coexist with newly built engines) share the same
   engine-diff algorithm.

4. **Skip the unused pure-rollout training shell.**
   [`shell_skip.py`](verl/experimental/one_step_off_policy/shell_skip.py) decides
   whether a process should skip building the training-shaped model shell it
   never uses. The decision is a pure predicate: actor processes always keep the
   shell, and only pure-rollout processes can skip it. The call sites and the
   config bridge live in
   [`fsdp_workers.py`](verl/experimental/one_step_off_policy/fsdp_workers.py)
   and
   [`config/one_step_off_ppo_trainer_dynamic_resize.yaml`](verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer_dynamic_resize.yaml).

5. **Tracing and startup attribution.**
   [`trace_utils.py`](verl/experimental/one_step_off_policy/trace_utils.py)
   provides the structured span API (`resize_trace_span`), JSONL emission, and
   optional NVTX labels. The trainer uses spans to attribute startup and resize
   work to named phases, and records the exception message (not only its type)
   so a failed resize can be diagnosed afterwards.

6. **Optional generation/resize overlap.**
   `ray_trainer.py` contains an opt-in pipeline that splits generation into a
   raw batch plus a topology-bound finalize step, and sinks an in-flight
   generation future into the resize path instead of draining it first. This is
   an explicit flag with its own known limitation (see
   [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#8-optional-overlap-pipeline)).
   The features in this snapshot are independently switchable; they are not
   claimed to be composable by default.

## Repository map

| Path | What it contains |
|---|---|
| [`verl/experimental/one_step_off_policy/awake_topology.py`](verl/experimental/one_step_off_policy/awake_topology.py) | Zone geometry, active-topology snapshot, engine partition/diff |
| [`verl/experimental/one_step_off_policy/elastic_engine_pool.py`](verl/experimental/one_step_off_policy/elastic_engine_pool.py) | Owner of elastic-zone handles and sleeping replicas; sleep/wake/reconcile |
| [`verl/experimental/one_step_off_policy/shell_skip.py`](verl/experimental/one_step_off_policy/shell_skip.py) | Pure predicate for skipping the training shell on rollout-only processes |
| [`verl/experimental/one_step_off_policy/ray_trainer.py`](verl/experimental/one_step_off_policy/ray_trainer.py) | Trainer orchestration: flags, resize lifecycle, survivor path, pool wiring, overlap pipeline |
| [`verl/experimental/one_step_off_policy/fsdp_workers.py`](verl/experimental/one_step_off_policy/fsdp_workers.py) | Split-role worker: shell-skip call site, `sleep`/`wake_up`, memory snapshot |
| [`verl/experimental/one_step_off_policy/trace_utils.py`](verl/experimental/one_step_off_policy/trace_utils.py) | Structured resize spans, JSONL trace, optional NVTX |
| [`verl/experimental/agent_loop/agent_loop.py`](verl/experimental/agent_loop/agent_loop.py) | Agent-loop side of generation/cold-start tracing and replica wiring |
| [`verl/experimental/one_step_off_policy/agent_loop/agent_loop.py`](verl/experimental/one_step_off_policy/agent_loop/agent_loop.py) | One-step-off replica adoption and per-replica trace context |
| [`verl/workers/rollout/vllm_rollout/vllm_rollout.py`](verl/workers/rollout/vllm_rollout/vllm_rollout.py) | Engine side of per-process local-rank binding used by split rollout workers |
| [`verl/workers/rollout/vllm_rollout/vllm_async_server.py`](verl/workers/rollout/vllm_rollout/vllm_async_server.py) | Async server: replica lifecycle, cold-start spans, local-rank/trace propagation |
| [`verl/utils/profiler/profile.py`](verl/utils/profiler/profile.py) | Profiler integration for the resize spans |
| [`verl/experimental/one_step_off_policy/worker_init_plan.py`](verl/experimental/one_step_off_policy/worker_init_plan.py) | Pre-existing communicator metadata plan retained by the legacy runtime |
| [`verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer_dynamic_resize.yaml`](verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer_dynamic_resize.yaml) | Dynamic-resize config example and the worker-side flag bridge |
| [`tests/experimental/one_step_off_policy/`](tests/experimental/one_step_off_policy/) | CPU-runnable algorithm and source-wiring tests (see below) |

The surrounding legacy runtime package, packaging files, and original recipe
are retained as the integration context. Runtime execution requires its full
dependencies and is not established by the portable checks below. The vLLM
files above are the version-sensitive part of the integration: they bind each
split rollout process to the cards its worker group owns, which is what makes
the engine's device selection correct when several cards are visible. They are
not a general vLLM patch and should be reviewed against whatever engine version
a downstream baseline uses.

## Configuration surface

All new behaviour is off by default unless noted. Keys are read from the
`trainer.dynamic_resize` node; worker-side keys are bridged into
`actor_rollout_ref.dynamic_resize` by the example config.

| Key | Default | Effect |
|---|---|---|
| `trainer.dynamic_resize.enable` | `true` in the dynamic-resize config | Master switch for the resize machinery |
| `trainer.dynamic_resize.worker_reuse.enable` | `false` | Retain rollout workers whose bundle set is unchanged |
| `trainer.dynamic_resize.worker_reuse.allow_mixed` | `false` | Allow survivors to coexist with newly built engines |
| `trainer.dynamic_resize.sleep_pool.enable` | `false` | Keep elastic-zone engines alive but asleep across resizes |
| `trainer.dynamic_resize.sleep_pool.permanent_rollout_gpus` | `0` | Size of the always-awake rollout zone `R` |
| `trainer.dynamic_resize.sleep_pool.elastic_gpus` | `0` | Size of the elastic zone `E`; must be positive when the pool is enabled |
| `trainer.dynamic_resize.overlap_prefetch` | `false` | Overlap in-flight generation with actor export and resize teardown |
| `trainer.dynamic_resize.skip_rollout_fsdp_shell` | `false` | Skip the training shell on pure-rollout processes |
| `trainer.dynamic_resize.trace.enable` / `.path` / `.run_name` / `.flush` | `false` / empty | Structured resize trace output |

Additional structural requirement: the sleep pool carves its elastic zone out of
one shared pool, so `trainer.dynamic_resize.shared_pool.enable` must be enabled
when the pool is enabled. The dynamic-resize example config also bridges
`skip_rollout_fsdp_shell` into `actor_rollout_ref.dynamic_resize`; enabling it on
only one side makes the worker-side decision silently stay off.

## Topology in one paragraph

The deployment is modelled as three zones over `N` cards: `A` permanent actor
cards, `E` elastic cards that change hands, and `R` permanent rollout cards,
with physical order `A | E | R`. Only the elastic zone participates in the pool.
A schedule point `k` says how many TP-sized groups of `E` rollout currently
owns; those groups are always the high end of `E`, because rollout owns the tail
of the shared pool. `A`, `E`, and `R` are sized in TP-group units, and invalid
geometry is rejected rather than silently repaired. The full argument, including
why sleeping groups must leave the rollout worker group, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#1-deployment-geometry-and-the-one-topology-truth).

## Where to start reading

1. This README for scope and flags.
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the invariants, the resize
   lifecycle order, and the manual-port boundaries.
3. [`awake_topology.py`](verl/experimental/one_step_off_policy/awake_topology.py)
   and
   [`elastic_engine_pool.py`](verl/experimental/one_step_off_policy/elastic_engine_pool.py):
   the portable core.
4. [`ray_trainer.py`](verl/experimental/one_step_off_policy/ray_trainer.py):
   the integration, which is the part a downstream port must re-implement rather
   than copy.
5. The tests: they double as executable specifications of the invariants.

## Running the portable CPU-only behaviour checks

The following three tests are the portable behaviour suites. They load the pure
modules by file path, or parse the relevant source with `ast`, and import only
the Python standard library plus `pytest`. They do not import `verl`, `torch`,
`ray`, or any GPU runtime:

```bash
python -m pytest -q \
  tests/experimental/one_step_off_policy/test_awake_topology_on_cpu.py \
  tests/experimental/one_step_off_policy/test_elastic_engine_pool_on_cpu.py \
  tests/experimental/one_step_off_policy/test_skip_rollout_fsdp_shell_on_cpu.py
```

This is a narrow check of the pure algorithms and of the shell-skip predicate.
It is not a training reproduction, it is not proof that the integrated resize
runtime works, and it says nothing about GPU memory or throughput. The broader
legacy test suite, including the source-wiring and trainer-level tests, requires
the full runtime and is outside this public entry point; its execution is
recorded in [`VERIFICATION.md`](VERIFICATION.md).

Running the trainer itself requires a complete upstream verl environment,
including Ray and a rollout backend. The original run environment's exact
dependency pins were not preserved with this snapshot, so no full-runtime
quickstart is claimed here. Package metadata is in
[`pyproject.toml`](pyproject.toml) and [`setup.py`](setup.py); for environment
setup, use upstream verl's own installation documentation as general guidance.
That guidance describes modern upstream, not a verified compatibility statement
for this historical snapshot.

## Provenance and review boundaries

- Source baseline: a pre-August research baseline built on upstream verl, commit
  `d3ed84fd9`.
- Snapshot tip: the final August commit, `4b39d5861`.
- The August changes are concentrated in `verl/experimental/one_step_off_policy/`
  and `tests/experimental/one_step_off_policy/`, with tracing and engine-binding
  changes in `verl/experimental/agent_loop/`, `verl/utils/profiler/profile.py`,
  and the two vLLM rollout files listed above. The rest of the tree is retained
  from the baseline so the code stays in its original context.
- This snapshot is intended for review and manual integration. It performs no
  porting into another baseline, and it does not include the later
  direct-reshard/full-training-state line.

## License

The upstream Apache License 2.0 and its notices are retained in
[`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt). The August source additions
retain their existing license headers. Private experiment anecdotes and local
infrastructure references were removed from comments and documentation;
the original unedited history remains separate from this public snapshot.

---

Checks run for this snapshot: [`VERIFICATION.md`](VERIFICATION.md).
