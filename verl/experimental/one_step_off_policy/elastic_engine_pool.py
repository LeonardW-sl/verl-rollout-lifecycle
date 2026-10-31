# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Owner of the elastic-zone rollout workers across resizes.

Guardrail G4 requires rollout_wg.workers to equal the awake set, because
sync_rollout_weights broadcasts to the whole group -- a sleeping member hangs the
broadcast rather than merely missing an update. Sleeping handles therefore have
to leave rollout_wg, and moving them out with nobody holding them loses them
for good. This pool is that owner.

No ray import: worker handles are invoked as handle.sleep.remote() and the
returned ObjectRefs are awaited directly (they are asyncio-awaitable). That keeps
the module importable, and drivable, on a bare CPU test machine.

The sleep/wake methods called here are the split-architecture overrides on
DetachAsyncRolloutWorker. They carry Dispatch.DIRECT_ROLLOUT_METHOD, which means
they must be called on the Ray actor handle directly -- routing them through a
worker group raises NotImplementedError by design.
"""

from __future__ import annotations

import asyncio
import time

# awake_topology is NOT imported. The pool only reads layout.num_elastic_groups,
# layout.tp and topology.awake_elastic_indices, so keeping those duck-typed lets
# the CPU tests load this module by file path without the verl package on
# sys.path. The concrete types are ZoneLayout and AwakeTopology.


class ElasticEnginePool:
    """Sleeps and wakes whole TP groups in the elastic zone.

    ``layout`` is an ``awake_topology.ZoneLayout``. The permanent rollout zone is
    never registered here: it is always awake by construction.
    """

    def __init__(self, layout):
        self._layout = layout
        self._groups: dict[int, tuple] = {}
        self._replicas: dict[int, object] = {}
        self._awake: set[int] = set()

    @property
    def layout(self):
        return self._layout

    def register_group(self, index: int, worker_handles, *, awake: bool) -> None:
        """Hand one prebuilt TP group to the pool. Called once per group at startup."""
        if not 0 <= index < self._layout.num_elastic_groups:
            raise ValueError(
                f"elastic group index {index} outside [0, {self._layout.num_elastic_groups - 1}]"
            )
        handles = tuple(worker_handles)
        if len(handles) != self._layout.tp:
            raise ValueError(
                f"elastic group {index} has {len(handles)} workers, expected TP={self._layout.tp}"
            )
        if index in self._groups:
            raise ValueError(f"elastic group {index} is already registered")
        self._groups[index] = handles
        if awake:
            self._awake.add(index)

    def handles(self, index: int) -> tuple:
        return self._groups[index]

    def register_replica(self, index: int, replica) -> None:
        """Attach the live rollout replica that owns one registered TP group."""
        if index not in self._groups:
            raise ValueError(f"elastic group {index} has no registered workers")
        if index in self._replicas:
            raise ValueError(f"elastic group {index} already has a replica")
        expected = tuple(getattr(handle, "_actor_id", handle) for handle in self._groups[index])
        actual = tuple(getattr(handle, "_actor_id", handle) for handle in replica.workers)
        if actual != expected:
            raise ValueError(f"elastic group {index} replica workers do not match")
        self._replicas[index] = replica

    def replica(self, index: int):
        return self._replicas[index]

    @property
    def all_registered(self) -> tuple:
        return tuple(sorted(self._groups))

    @property
    def all_replicas(self) -> tuple:
        return tuple(self._replicas[index] for index in sorted(self._replicas))

    @property
    def awake_indices(self) -> tuple:
        return tuple(sorted(self._awake))

    @property
    def sleeping_indices(self) -> tuple:
        return tuple(sorted(set(self._groups) - self._awake))

    def awake_handles(self) -> tuple:
        """Awake elastic handles in group-index order, to be appended after the
        permanent rollout zone when rebuilding rollout_wg."""
        handles: list = []
        for index in self.awake_indices:
            handles.extend(self._groups[index])
        return tuple(handles)

    def reconcile(self, topology) -> tuple[list[int], list[int]]:
        """Groups to sleep and to wake in order to reach `topology`'s schedule.

        ``topology`` is an ``awake_topology.AwakeTopology``.
        """
        target = set(topology.awake_elastic_indices)
        return sorted(self._awake - target), sorted(target - self._awake)

    def build_reuse_plan(self, topology, survivor_plan, old_replicas):
        """Complete target-rank reuse map plus old replica ranks to preserve.

        Physical survivors supply the permanent rollout replicas. Pool replicas
        supply every awake elastic rank. Pool replicas present in the old manager
        are kept even when they leave the target topology, because a later resize
        must wake the same live server rather than cold-build a replacement.
        """
        reuse = {}
        keep_old_ranks = set()

        for survivor in survivor_plan.get("survivors", []):
            old_rank = int(survivor["old_rank"])
            new_rank = int(survivor["new_rank"])
            reuse[new_rank] = old_replicas[old_rank]
            keep_old_ranks.add(old_rank)

        old_rank_by_identity = {id(replica): rank for rank, replica in enumerate(old_replicas)}
        for replica in self._replicas.values():
            old_rank = old_rank_by_identity.get(id(replica))
            if old_rank is not None:
                keep_old_ranks.add(old_rank)

        for group_index in topology.awake_elastic_indices:
            new_rank = topology.elastic_replica_rank(group_index)
            replica = self._replicas[group_index]
            existing = reuse.get(new_rank)
            if existing is not None and existing is not replica:
                raise ValueError(f"replica rank {new_rank} has conflicting reuse objects")
            reuse[new_rank] = replica

        expected = set(range(topology.num_replicas))
        if set(reuse) != expected:
            raise ValueError(
                f"sleep-pool reuse is incomplete: got={sorted(reuse)} expected={sorted(expected)}"
            )
        return reuse, keep_old_ranks

    async def sleep_groups(self, indices) -> float:
        """Release the GPU memory of the named groups. Returns elapsed seconds.

        Already-sleeping groups are skipped rather than re-slept. State is only
        updated after every replica has acknowledged: a group that half-slept must
        not be recorded as asleep, or a live engine ends up outside rollout_wg
        with no route back (G4).
        """
        pending = [i for i in indices if i in self._awake]
        started_at = time.monotonic()
        if pending:
            missing = [index for index in pending if index not in self._replicas]
            if missing:
                raise ValueError(f"elastic groups {missing} have no registered replica")
            await asyncio.gather(*(self._replicas[index].sleep() for index in pending))
            self._awake.difference_update(pending)
        return time.monotonic() - started_at

    async def wake_groups(self, indices) -> float:
        """Remap the named groups' memory. Returns elapsed seconds.

        Waking does not restore weights: under VLLM_SLEEP_LEVEL=2 there is no CPU
        copy, and post-resize weights arrive from the driver's single sync entry
        point (G1). Callers must run that sync before generating.
        """
        pending = [i for i in indices if i in self._groups and i not in self._awake]
        started_at = time.monotonic()
        if pending:
            missing = [index for index in pending if index not in self._replicas]
            if missing:
                raise ValueError(f"elastic groups {missing} have no registered replica")
            await asyncio.gather(*(self._replicas[index].wake_up() for index in pending))
            self._awake.update(pending)
        return time.monotonic() - started_at
