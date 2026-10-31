# Architecture: rollout lifecycle, retention, and resize

This document explains the invariants behind the August 2026 contribution and
the order in which the resize lifecycle enforces them. It is written for a
reviewer who needs to judge the logic and then integrate it by hand into a
different verl baseline. It contains no measured results.

The code lives entirely in
[`verl/experimental/one_step_off_policy/`](../verl/experimental/one_step_off_policy/).
The tests under
[`tests/experimental/one_step_off_policy/`](../tests/experimental/one_step_off_policy/)
are the executable form of the invariants described here.

## 1. Deployment geometry and the one topology truth

### Symbols

| Symbol | Meaning |
|---|---|
| `N` | Total cards in the shared pool |
| `TP` | Tensor-parallel width of one rollout engine |
| `A` | Cards permanently assigned to the actor (`A = N - R - E`) |
| `R` | Cards permanently assigned to rollout; always awake |
| `E` | Cards that change hands between actor and rollout |
| `m` | Number of TP-sized groups in the elastic zone (`m = E / TP`) |
| `k` | Schedule point: how many elastic groups rollout currently owns (`0 <= k <= m`) |

Physical card order is `A | E | R`. That order is not arbitrary: in the existing
split plans, rollout always owns the tail of the shared pool, so the cards
rollout keeps in every plan are the last `R`, and the cards it holds only after
expanding are the middle `E`. Reversing `E` and `R` would pin the pool to cards
the actor permanently owns, and the mistake would not fail loudly — it would
silently double-book those cards.

A partial schedule takes the **high end** of `E`: rollout owns the tail of the
elastic zone, mirroring the fact that rollout owns the tail of the shared pool.
A toy example with `N = 6`, `TP = 2`, `R = 2`, `E = 2` (`m = 1`, `A = 2`):
`k = 1` means rollout owns the single elastic group, and its bundle order is
elastic group first, then the permanent rollout tail.

### `ZoneLayout`

[`awake_topology.py`](../verl/experimental/one_step_off_policy/awake_topology.py)
defines `ZoneLayout`, an immutable geometry object that validates its inputs at
construction:

- `TP` and `N` must be positive.
- `R` and `E` must be non-negative and whole multiples of `TP`.
- The zones may not oversubscribe the cluster (`A >= 0`).
- A schedule point outside `[0, m]` is rejected.

Validation raises rather than clamping or falling back. The same module derives
`k` from a split plan's rollout-card count
(`schedule_from_rollout_cards`): a rollout count below `R`, a count that is not
a whole number of `TP` groups above `R`, or a schedule beyond `E` is an error.
The intent is that a topology which does not land exactly on a schedule point
must fail loudly, never degrade into a silent full rebuild that would look like
a successful resize in the trace.

### `AwakeTopology`

`AwakeTopology` is the **single source of truth for which rollout workers are
awake at one schedule point**. It is an immutable snapshot built once per
resize via `from_schedule`, which validates the shape: exactly `m` elastic
groups of `TP` handles each, and exactly `R` permanent rollout workers.

Consumers read its properties instead of recomputing topology:

- `awake_elastic_indices` / `sleeping_elastic_indices` — schedule-derived.
- `awake_rollout_workers` — ordered handles: awake elastic groups first, then
  the permanent rollout tail.
- `awake_rollout_bundles` — the physical bundle order corresponding to those
  handles.
- `sleeping_workers` — handles that must not enter the rollout worker group.
- `num_replicas` — awake rollout workers divided by `TP`.
- `collective_members` — actor workers plus awake rollout workers.
- `validate_src_rank` — checks a proposed broadcast source against the live
  actor set.

The **ordering of `awake_rollout_workers` is a contract**, not a convenience.
List position pins the replica rank: the agent loop slices workers as
`workers[rank*TP:(rank+1)*TP]`, and the ZMQ executor slices addresses
identically. A different order would pair a worker with the wrong replica.

The module exists because this topology used to be recomputed independently at
several call sites. Two of those copies disagreeing silently dropped a retained
engine while the log still reported it as reused. Moving the algorithm out of
the trainer makes disagreement structurally impossible: there is one snapshot,
and every consumer reads it.

## 2. Engine identity and the survivor plan

Two pure functions in `awake_topology.py` define engine identity:

- `partition_engines(bundles, tp)` groups a flat ordered bundle list into
  TP-sized engines. Each engine is identified by its **sorted** bundle tuple, so
  identity does not depend on the order the caller listed cards in.
- `diff_engines(old_bundles, new_bundles, tp)` returns:
  - `survivors`: engines whose bundle set is unchanged, with old rank, new rank,
    and bundles;
  - `to_kill_old_ranks`: old engines with no matching new engine;
  - `to_build_new_ranks`: new engines with no matching old engine.

An engine whose bundle set survived is not a separate state; it is simply a
group that stayed active. Both retention axes — "not killed" and "not slept" —
read the same diff function, so they cannot drift apart.

If the bundle lists cannot be partitioned (empty list, non-positive `TP`, or a
count that is not a whole number of groups), `partition_engines` returns `None`
and `diff_engines` returns `has_plan = False`. The trainer then takes the full
rebuild path. This is the only intended fallback: malformed *inputs* cause a
rebuild; a well-formed topology that merely does not match the configured zone
geometry raises instead.

## 3. Ownership: what the pool holds

[`elastic_engine_pool.py`](../verl/experimental/one_step_off_policy/elastic_engine_pool.py)
defines `ElasticEnginePool`, the owner of the elastic zone across resizes.
It holds:

- one tuple of worker handles per elastic group (`register_group`);
- the live rollout replica that owns each group (`register_replica`);
- the set of groups currently awake.

The permanent rollout zone is **never registered**: it is always awake by
construction, and the pool exists only to handle groups that can sleep.

`reconcile(topology)` compares the current awake set with the target snapshot
and returns `(groups_to_sleep, groups_to_wake)`. `sleep_groups` and
`wake_groups` call the replicas' sleep/wake methods concurrently within a group
set, and update the awake set only after every replica has acknowledged. A group
that half-slept must not be recorded as asleep, because a live engine outside
the rollout worker group has no route back.

`build_reuse_plan(topology, survivor_plan, old_replicas)` produces the complete
target-rank map:

- physical survivors supply the permanent-rollout replica ranks;
- pool replicas supply every awake elastic rank;
- pool replicas that exist in the old manager are kept alive even if they are
  not in the target topology, so a later resize wakes the same engine instead
  of cold-building a replacement;
- a conflicting assignment for one target rank is an error, and so is an
  incomplete map.

The module imports no Ray. Handles are invoked as `handle.sleep.remote()` and
the returned object references are awaited directly. This keeps the pool
importable and drivable on a bare CPU test machine, with fake handles.

## 4. Lifecycle: startup

Startup builds the world once, then establishes the initial schedule:

1. Resource pools are created; the dynamic-resize example maps roles onto one
   shared pool, which the sleep pool carves its elastic zone out of.
2. Worker groups and models are initialized through the normal trainer path.
3. The async rollout manager is created for the initial rollout group, inside a
   trainer-lane span so the startup cost can be attributed.
4. The elastic pool is prebuilt: initially awake groups adopt the committed
   rollout workers; initially sleeping groups are built once and registered
   asleep. Each group is built through the same prepare/commit phases as any
   other worker group, with tracing around each phase.
5. `_finalize_elastic_pool_replicas` adopts live replicas where they already
   exist, launches the missing ones, and records the prebuild cost.
6. The initial schedule `k0` is derived from the committed rollout card count.

Device scoping is part of group construction: each elastic group is given a
worker environment whose visible devices are exactly that group's bundles. A
split rollout process that could see several cards would otherwise compute the
wrong local device for its engine. Scoping what each process can see makes the
engine's device choice correct by construction rather than requiring an engine
patch. This is the one place where the integration is coupled to vLLM's
per-process device discovery; see the note in
[`../README.md`](../README.md#repository-map).

From then on, an engine is built at most once per process lifetime. A resize
changes *which* engines are awake and which replica rank each one occupies; it
does not rebuild the elastic-zone engines.

## 5. Lifecycle: one resize

The resize path is ordered by invariants whose violation is silent rather than
loud. The order below is the contract; the code comments in `ray_trainer.py`
name the same guardrails.

### 5.1 Harvest, then tear down, then sleep/wake

1. **Tear down the old weight-sync collective before killing any actor
   worker.** The collective's TCP store couples every member process. If an
   actor is killed first, retained rollout workers still attached to the old
   group see a broken pipe and their communication state collapses. The cleanup
   is registered in one place so the ordering applies to every resize path.
2. **Remove the old actor group.**
3. **Harvest in-flight generation.** Await any pending raw-generation task now:
   the old actor is gone and the collective is down, but no rollout engine has
   been destroyed. This is the only barrier that guarantees quiescence before
   an engine can be slept or killed. With overlap disabled, or with no pending
   future, this is a no-op and the legacy path is unchanged.
4. **Reconcile the sleep pool** against the topology derived from the new
   rollout spec. Sleeping completes before waking begins — strict serialization.
   The tempting overlap is not worth the failure mode: a partially reclaimed
   card can fail later with no clear attribution. This step never touches
   weights; under the sleep-level semantics used here there is no CPU copy, and
   the driver's single sync entry point fills the engine afterwards.
5. **Rebuild the rollout worker group from the awake set.** The group must
   contain exactly the awake workers. Sleeping handles stay owned by the pool
   and outside the group, because weight synchronization broadcasts to every
   group member: a sleeping member would hang the broadcast, not merely miss an
   update.
6. **Build and commit the new actor group, and restore training state** through
   the handoff/staging path concurrently with rollout-side work where the
   implementation allows it.
7. **Create the new weight-sync collective** for the new topology, then perform
   the single post-resize weight sync into the awake rollout engines.

### 5.2 Invariants

| Invariant | Meaning | Enforced by |
|---|---|---|
| **G1 — one weight path** | The pool and the sleep/wake methods never move weights. Post-resize weights arrive through the driver's single sync entry point. | `elastic_engine_pool.py`, split worker `sleep`/`wake_up` docstrings, trainer sync call |
| **G2 — collective membership** | The collective contains actor workers plus **awake** rollout workers only; the broadcast source must be a live actor member. | `AwakeTopology.collective_members`, `validate_src_rank` |
| **G3 — quiescence before sleep/kill** | No engine is slept or destroyed while a generation request is in flight. | Harvest performed after actor removal and before engine destruction; pool reconcile follows it |
| **G4 — group equals awake set** | `rollout_wg.workers` is exactly the awake set. Sleeping handles are owned by the pool and excluded from the group. | `AwakeTopology`, `build_reuse_plan`, trainer group rebuild |
| **Zone geometry** | `R` and `E` are TP multiples, `E > 0` when the pool is enabled, zones do not oversubscribe, `k` is derived exactly. | `ZoneLayout` |
| **Replica ordering** | Worker list order pins replica rank. | `AwakeTopology.awake_rollout_workers` |
| **Default-off** | With a feature flag off, the corresponding legacy path is used unchanged. | Trainer flag helpers and branch gates |

## 6. What retention preserves — and what it does not

This distinction matters for a reviewer integrating the work:

**Preserved across a resize**

- Rollout worker processes whose GPU bundle set is unchanged (survivor path).
- Elastic-zone engine replicas, asleep or awake, together with their live
  servers and their identity across schedule points (sleep-pool path).

**Still rebuilt or transferred on every resize**

- The actor worker group. The actor process is removed and a new group is
  committed; training state is carried across by the handoff/staging path, not
  by keeping the actor process alive.
- The weight-sync collective. It is torn down before worker destruction and
  recreated for the new topology.
- Rollout weights. Waking an engine remaps memory; it does not restore weights.
  The post-resize sync is mandatory before generation.
- Rollout KV cache and other transient engine state are not preserved as
  training state; any pre-resize clear/cache handling remains part of the
  resize path.

In short: retention avoids **engine construction**, not the training-side
rebuild or the state-transfer work. A downstream port that keeps engines but
forgets the post-wake weight sync would produce stale generation, so G1 is not
optional.

## 7. The shell-skip boundary

A pure-rollout process in the split architecture builds a full training-shaped
model shell it never trains. `shell_skip.py` isolates the decision to skip that
shell into a pure predicate:

```python
should_skip_rollout_fsdp_shell(*, is_actor, is_rollout, flag_enabled)
```

The predicate is deliberately asymmetric:

- the flag must be on;
- `is_actor` wins unconditionally — actor and co-located actor-rollout
  processes train from the shell, so skipping it there would be corruption
  rather than a saving;
- only a pure-rollout process may skip.

The worker resolves the decision once at construction from
`dynamic_resize.skip_rollout_fsdp_shell` on its worker-side config. Because the
worker only sees `actor_rollout_ref`, the example config bridges the trainer
key into `actor_rollout_ref.dynamic_resize`; without that bridge the flag
reaches the trainer and silently no-ops in the worker.

When the shell is skipped, the worker re-sources tokenizer, processor, and
generation config from the model config that rollout construction already
created, rather than loading second copies. The standalone rollout checkpoint
manager is also not created, because with no model it would be an empty
load-only shell and the rollout group does not restore-load.

Two boundaries are intentional:

- **The shell-skip flag never changes actor-side memory or actor-side
  behaviour.** It cannot fix pressure caused by actor shrinkage, because actor
  processes always build the shell.
- **Skipping the shell is what makes a sleeping engine actually give its card
  share back.** The engine's own sleep path releases engine memory, but the
  training shell lives in the torch allocator and is not released by sleep.
  A sleeping elastic group that still carries the shell therefore continues to
  occupy the cards it was supposed to free. This is why the two changes form a
  chain rather than two independent optimizations.

The flag remains default-off so the old behaviour stays reproducible and an A/B
comparison remains possible.

## 8. Optional overlap pipeline

Flag: `trainer.dynamic_resize.overlap_prefetch`, default off, and active only
when dynamic resize itself is enabled.

In the legacy pipeline, a resize step stops generation and drains it before the
resize begins. With overlap enabled:

1. Generation is split into `_async_gen_raw_batch` (produce the raw,
   topology-agnostic batch) and a finalize step that binds the batch to the
   current topology and launches any post-generation work.
2. On a resize step, the in-flight raw-generation task is not drained before
   resize. It is handed to the resize path and continues through actor export,
   group teardown, and actor removal.
3. At the common resize deadline — after the actor is gone and before any
   rollout engine is destroyed — the resize path awaits the pending task
   (`_harvest_pending_generation`). The harvested raw batch is finalized on the
   next post-resize step.

This is an explicit, self-contained flag. It is not claimed to compose with
every other feature by default, and it has a known limitation: when enabled,
**every** step routes generation through the raw/finalize split, so the
topology-bound tail — including the asynchronous reward launch when that feature
is on — is deferred to the consuming step. That loses the original overlap
between reward launch and actor training. The configuration this work was
developed against runs rewards synchronously, so that path was not exercised.
Revisit before enabling asynchronous rewards with this flag.

## 9. Instrumentation and startup attribution

`trace_utils.py` provides one choke point, `resize_trace_span(config, event,
step=..., lane=..., metadata=...)`. Every resize phase goes through it, which
means one place emits both the structured record and the optional NVTX label.

What a span records:

- event name, step, lane;
- wall-clock start/end and monotonic duration;
- status (`ok` / `error`);
- exception type **and message** — the message was added so an out-of-memory
  failure can be attributed afterwards instead of being permanently opaque;
- caller-supplied metadata.

JSONL emission is config-gated (`trainer.dynamic_resize.trace.enable` plus a
path). NVTX labelling is attempted regardless, so a profiler timeline remains
useful even when JSONL output is off.

Startup attribution uses the same mechanism. The initialization sequence is
wrapped at phase boundaries: the async rollout manager startup has its own
trainer-lane span, elastic groups emit build/prepare/commit spans, and the
replica finalize step has its own span. Memory probes are deliberately placed
outside the attribution envelope, so RPC round trips do not inflate the phase
they are measuring.

A methodological caveat is part of the design record: **absence of a span is
not evidence that the event did not happen.** Some spans are config-gated or
environment-gated, and coverage is not uniform across old and new paths. Only
the *presence* and contents of a span are positive evidence; an empty trace must
not be used to argue that work was reused or skipped.

## 10. Manual-port boundaries

The work was written against a specific trainer and split-role worker
architecture. It was **not** ported into another baseline, and no port is
included here. For a downstream integration, separate the portable core from the
host-bound wiring.

### Portable with little or no change

- [`awake_topology.py`](../verl/experimental/one_step_off_policy/awake_topology.py)
  — dataclasses and pure functions; standard library only.
- [`elastic_engine_pool.py`](../verl/experimental/one_step_off_policy/elastic_engine_pool.py)
  — asyncio and time only; worker handles are duck-typed, so the pool does not
  depend on Ray types.
- [`shell_skip.py`](../verl/experimental/one_step_off_policy/shell_skip.py)
  — typing only; accepts dicts or config nodes.
- [`trace_utils.py`](../verl/experimental/one_step_off_policy/trace_utils.py)
  — standard library at module scope, with an optional NVTX path and a lazy
  `omegaconf` import in the config builder; takes a config object.

Their CPU tests load these files by path, which is why the tests can run without
the verl package on `sys.path`.

### Host-bound: re-implement, do not copy

The wiring in `ray_trainer.py` assumes:

- a trainer class with the existing dynamic-resize phases and a resource-pool
  manager that exposes a shared pool;
- split roles, with rollout worker groups that can be detached and rebuilt from
  retained handles (`RayWorkerGroup.from_detached`);
- an async rollout manager that exposes `rollout_replicas`, sleep/wake on
  replicas, and a post-resize weight-sync entry point.

The wiring in `fsdp_workers.py` assumes:

- a split-role rollout worker (`DetachAsyncRolloutWorker`) whose sleep/wake
  methods are dispatched directly on the Ray actor handle, not through a worker
  group;
- that the inherited unified worker's sleep path is unsuitable, because it
  assumes a co-located actor module that the split worker does not have.

A different baseline will almost certainly have different class names, a
different resize phase structure, or a different model-residency model. The
correct integration is to keep the invariants and the pure algorithms, and to
re-implement the host wiring against that baseline's lifecycle. Copying
`ray_trainer.py` wholesale is not the intended path.

### What is intentionally out of scope

- No merge or port into a newer branch.
- No direct-reshard or full-training-state implementation from other work
  lines.
- No compatibility assertion against modern upstream verl.

## 11. Known limitations

1. **The sleep pool is single-node by construction.** It reads
   `shared_pool.n_gpus_per_node` as the cluster-wide card count. That is correct
   only when the deployment is one node; with multiple nodes it under-counts
   `N`, and the zone split would land on the wrong cards instead of failing
   loudly. Multiply by the node count before trusting it on multiple nodes.
2. **Sleep-level semantics are engine-dependent.** Wake maps memory; it does not
   restore weights. Under the level-2 semantics this work used there is no CPU
   copy. Other configurations (for example, layered summon or expert parallel)
   may force a different sleep level; the worker reads the actual level from the
   live rollout rather than assuming one.
3. **The shell is a prerequisite for real sleeping.** Without shell skipping,
   a sleeping elastic group still carries the training shell in the torch
   allocator and continues to occupy card capacity.
4. **Pool geometry requirements.** The pool needs the shared pool enabled,
   positive `elastic_gpus`, and `R`/`E` in whole TP groups. Misconfiguration
   raises rather than silently degrading.
5. **Mixed survivor retention is opt-in.** The default survivor path covers a
   pure shrink (survivors fill the entire new rollout topology). Allowing
   survivors to coexist with newly built engines requires
   `worker_reuse.allow_mixed`, which is gated separately.
6. **Invalid survivor inputs fall back to a full rebuild.** A bundle list that
   cannot be partitioned yields `has_plan = False`; the trace's decision span
   exists so such a fallback is distinguishable from a real survivor run.
7. **Overlap has an asynchronous-reward limitation.** See
   [§8](#8-optional-overlap-pipeline).
8. **Trace coverage is uneven.** Some spans are gated by config or environment;
   absence of a record is not proof that the corresponding work did not occur.
9. **This snapshot contains no training-metrics verification.** Loss/reward
   agreement and full runtime behaviour are not established by these docs. See
   [`../VERIFICATION.md`](../VERIFICATION.md) for the checks that were run.
