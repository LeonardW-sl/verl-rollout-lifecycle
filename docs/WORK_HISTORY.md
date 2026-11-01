# Work history and publication dates

**The code is historical; the public repository is new.** The original contribution range represented by this snapshot contains 77 commits dated **2026-06-22 through 2026-08-22**. The sanitized public repository was created on **2026-10-31**. Its initial publication commit is not the start of the work.

This chronology was checked against the original contribution range's commit dates and changed features. It is a concise code-development timeline, not a benchmark report or a public export of the private commit ledger. Original commits retain their original dates in the private repository; no artificial commits, backdating or history rewrite were used to make this public repository look older.

## Milestones represented in the snapshot

| Original commit dates | Code-development milestone | Public source entry |
|---|---|---|
| 2026-06-22 | Resize shutdown/initialization and individual-engine tracing | [trace_utils.py](../verl/experimental/one_step_off_policy/trace_utils.py), [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py) |
| 2026-06-26 | Survivor collective ranks, detached worker methods, replica rank mapping and old weight-sync teardown ordering | [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py) |
| 2026-06-29–2026-07-24 | Rollout-server cold-start trace propagation and optional profiling labels | [vllm_async_server.py](../verl/workers/rollout/vllm_rollout/vllm_async_server.py), [trace_utils.py](../verl/experimental/one_step_off_policy/trace_utils.py) |
| 2026-07-24–2026-07-26 | Opt-in generation/resize overlap, topology-bound finalization and pending-generation harvest guards | [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py) |
| 2026-08-04 | Opt-in mixed survivor expansion | [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py) |
| 2026-08-05–2026-08-10 | Shared engine-diff algorithm, zone geometry, awake-topology snapshot, sleeping-replica ownership and sleep/wake integration | [awake_topology.py](../verl/experimental/one_step_off_policy/awake_topology.py), [elastic_engine_pool.py](../verl/experimental/one_step_off_policy/elastic_engine_pool.py) |
| 2026-08-14–2026-08-17 | Startup attribution and worker memory-sampling instrumentation; an earlier instrumentation version was reverted and replaced | [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py), [fsdp_workers.py](../verl/experimental/one_step_off_policy/fsdp_workers.py) |
| 2026-08-21–2026-08-22 | Default-off pure-rollout training-shell skipping and exception-message tracing | [shell_skip.py](../verl/experimental/one_step_off_policy/shell_skip.py), [trace_utils.py](../verl/experimental/one_step_off_policy/trace_utils.py) |
| 2026-10-31 | Sanitized source snapshot and public architecture/verification documentation published | [README](../README.md), [architecture](ARCHITECTURE.md), [verification](../VERIFICATION.md) |

## What a reader can verify publicly

The [public review guide](REVIEW.md) points to the implemented algorithms, lifecycle contracts and existing portable tests. The source baseline and final snapshot identifiers are recorded in [README provenance](../README.md#provenance-and-review-boundaries).

The private original ledger remains the evidence for the exact historical dates; it is deliberately not republished. Dates in this page are a checked summary, not a claim that all original commits are browsable in this public repository. The code links show the final state, not historical diffs.

Later work on other baselines is outside this snapshot. No private experiment outcomes, participant identities, machine details or access configuration are part of this timeline.
