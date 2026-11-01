# Public review guide

This is the public, code-only review packet for the historical rollout-lifecycle contribution. It is not the original private review request or a public copy of its attachments. Read [WORK_HISTORY.md](WORK_HISTORY.md) for the distinction between work dates and publication date.

## Review order

1. [README](../README.md): scope, independently gated features and portable checks.
2. [Architecture](ARCHITECTURE.md): ownership, ordering and migration boundaries.
3. The seven review units below, following the linked source and existing tests.
4. [Verification](../VERIFICATION.md): observed checks and runtime limits.

All source links resolve within this public snapshot. No private repository access is required for these steps.

## Seven review units

| Unit | Source entry | Consumer-visible invariant to inspect |
|---|---|---|
| Resize and cold-start tracing | [trace_utils.py](../verl/experimental/one_step_off_policy/trace_utils.py), [vllm_async_server.py](../verl/workers/rollout/vllm_rollout/vllm_async_server.py) | A span carries its phase, context and exception message; a missing span is not evidence that the phase did not run. |
| Survivor identity and rank mapping | [awake_topology.py](../verl/experimental/one_step_off_policy/awake_topology.py), [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py) | Physical bundle identity, not logical rank equality, determines survival. Each surviving engine is mapped from old rank to target rank. Mixed retention requires its own flag. |
| Zone geometry and awake membership | [awake_topology.py](../verl/experimental/one_step_off_policy/awake_topology.py) | Geometry is `A | E | R`; awake elastic groups come from the high end of `E`. Ordered awake workers define replica ranks. Sleeping workers are excluded from the live worker group and weight-sync collective. |
| Pool ownership and lifecycle | [elastic_engine_pool.py](../verl/experimental/one_step_off_policy/elastic_engine_pool.py), [fsdp_workers.py](../verl/experimental/one_step_off_policy/fsdp_workers.py) | The pool owns sleeping replicas. Sleep/wake completion precedes state bookkeeping; waking an engine never substitutes for weight synchronization. |
| Pure-rollout shell skipping | [shell_skip.py](../verl/experimental/one_step_off_policy/shell_skip.py), [dynamic-resize config](../verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer_dynamic_resize.yaml) | Actor processes always retain the training model shell. Pure-rollout skipping is opt-in and must reach the worker-side configuration. |
| Optional generation overlap | [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py), [architecture section 8](ARCHITECTURE.md#8-optional-overlap-pipeline) | Raw generation and topology-bound finalization are separate. Pending generation is harvested before rollout sleep/destruction, including resize-decline paths. The asynchronous-reward limitation remains. |
| Startup and memory instrumentation | [ray_trainer.py](../verl/experimental/one_step_off_policy/ray_trainer.py), [fsdp_workers.py](../verl/experimental/one_step_off_policy/fsdp_workers.py), [architecture section 9](ARCHITECTURE.md#9-instrumentation-and-startup-attribution) | Missing or partial worker samples must not become a complete measurement. Configuration-gated probes do not establish a performance result. |

## Ordering to trace in the trainer

Follow [architecture section 5](ARCHITECTURE.md#5-lifecycle-one-resize) against the actual trainer path:

1. Tear down the old weight-sync collective before removing old actor workers.
2. Harvest pending generation before any rollout engine is slept or destroyed.
3. Complete sleeping before waking; rebuild the rollout group from the ordered awake set.
4. Build the new actor group and carry training state through the existing handoff path.
5. Recreate the collective for the new membership and synchronize weights before generation.

The pure algorithms do not prove that these steps execute correctly on a GPU deployment. Review the host-bound call sites separately.

## Run the portable checks

Run from the repository root with Python and `pytest` installed:

```bash
python3 -m pytest -q \
  tests/experimental/one_step_off_policy/test_awake_topology_on_cpu.py \
  tests/experimental/one_step_off_policy/test_elastic_engine_pool_on_cpu.py \
  tests/experimental/one_step_off_policy/test_skip_rollout_fsdp_shell_on_cpu.py
```

These existing tests exercise geometry, engine identity, pool ownership and shell-skipping behavior without importing the full training runtime. They do not reproduce training, GPU memory reclamation, multi-node operation or throughput. Actual executed results are recorded in [VERIFICATION.md](../VERIFICATION.md).

## Integration decision checklist

- Keep one ordered awake-topology snapshot and one engine-diff algorithm.
- Decide which persistent-replica owner the target baseline already has; do not introduce a second owner.
- Reimplement trainer, worker-group and rollout-manager wiring against that baseline. Do not overwrite its trainer with this historical file.
- Review backend-specific sleep levels and device binding against the installed rollout engine version.
- Verify state continuity and real collective membership in the target runtime before treating the integration as complete.
- Treat the sleep-pool layout here as single-node. This snapshot does not implement the later multi-node/direct-reshard work.

## Publication boundary

The complete original commit ledger and original review attachments remain private. This packet publishes neither raw commit messages/author details nor server-access instructions, network topology, machine identifiers, private paths, teammate notes, raw traces or experiment measurements. It does not replace authenticated access when the reviewer needs the original complete history or private evidence.
