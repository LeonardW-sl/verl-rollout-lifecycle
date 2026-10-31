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
"""Single source of truth for "which rollout engines are awake right now".

The effective rollout topology used to be recomputed independently at four call
sites. When two of them disagreed, a survivor adopted at a new rank >= a locally
derived num_replicas was silently dropped while the log still reported it as
reused. An earlier refactor collapsed two of those four inside
_staged_resize_shared_pool, but only function-locally. The two it did not reach
-- collective membership and src_rank -- are exactly the ones sleep mode exposes
(guardrails G2/G4), so this module takes the algorithm out of the trainer
entirely.

Deliberately stdlib-only: the *_on_cpu.py test convention in this package runs
without torch/ray installed, and ray_trainer imports this at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass


def partition_engines(bundles, tp):
    """Group a flat, ordered GPU-bundle list into TP-sized engines.

    Each engine is identified by its *sorted* bundle tuple so that engine
    identity does not depend on the order the caller happened to list cards in.

    Returns None when the caller must fall back to a full rebuild: empty bundle
    list, non-positive TP, or a card count that is not a whole number of TP
    groups. None rather than an exception preserves the has_plan=False contract
    of the survivor plan this backs.
    """
    if tp <= 0 or not bundles or len(bundles) % tp != 0:
        return None
    return [tuple(sorted(bundles[i : i + tp])) for i in range(0, len(bundles), tp)]


def diff_engines(old_bundles, new_bundles, tp):
    """The one algorithm for "which engines survive a topology change".

    An engine whose bundle set is unchanged across the resize can stay alive.
    A survivor is not a distinct state -- it is just a group that happened to
    stay active -- so axis 1 (not killed) and axis 2 (not slept) both read this
    same function instead of keeping private copies that can drift.

    Returns the survivor/kill/build decomposition; see the module docstring for
    why this shape is fixed.
    """
    old_engines = partition_engines(old_bundles, tp)
    new_engines = partition_engines(new_bundles, tp)
    if old_engines is None or new_engines is None:
        return {"has_plan": False, "survivors": [], "to_kill_old_ranks": [], "to_build_new_ranks": []}

    old_by_set = {eng: rank for rank, eng in enumerate(old_engines)}
    new_by_set = {eng: rank for rank, eng in enumerate(new_engines)}
    return {
        "has_plan": True,
        "survivors": [
            {"old_rank": old_by_set[eng], "new_rank": new_rank, "bundles": list(eng)}
            for new_rank, eng in enumerate(new_engines)
            if eng in old_by_set
        ],
        "to_kill_old_ranks": sorted(rank for eng, rank in old_by_set.items() if eng not in new_by_set),
        "to_build_new_ranks": sorted(rank for eng, rank in new_by_set.items() if eng not in old_by_set),
    }


@dataclass(frozen=True)
class ZoneLayout:
    """Static three-zone deployment geometry.

    Cards split into A (permanent actor) | R (permanent rollout) | E (elastic).
    Only the elastic zone changes hands across a resize, so only it is pooled;
    the permanent rollout zone is always awake and never enters the pool.
    """

    total_gpus: int
    permanent_rollout_gpus: int
    elastic_gpus: int
    tp: int

    def __post_init__(self):
        if self.tp <= 0:
            raise ValueError(f"tp must be positive, got tp={self.tp}")
        if self.total_gpus <= 0:
            raise ValueError(f"total_gpus must be positive, got {self.total_gpus}")
        if self.permanent_rollout_gpus < 0 or self.elastic_gpus < 0:
            raise ValueError(
                f"zone sizes must be non-negative, got R={self.permanent_rollout_gpus} E={self.elastic_gpus}"
            )
        if self.permanent_rollout_gpus % self.tp != 0:
            raise ValueError(
                f"permanent rollout zone R={self.permanent_rollout_gpus} is not a whole number of "
                f"TP={self.tp} groups (section 2 constraint 1)"
            )
        if self.elastic_gpus % self.tp != 0:
            raise ValueError(
                f"elastic zone E={self.elastic_gpus} is not a whole number of TP={self.tp} groups "
                f"(section 2 constraint 2)"
            )
        if self.actor_gpus < 0:
            raise ValueError(
                f"zones oversubscribe the cluster: N={self.total_gpus} < "
                f"R={self.permanent_rollout_gpus} + E={self.elastic_gpus}"
            )

    @property
    def actor_gpus(self) -> int:
        """A = N - R - E."""
        return self.total_gpus - self.permanent_rollout_gpus - self.elastic_gpus

    @property
    def num_elastic_groups(self) -> int:
        """m = E / TP: how many TP groups the schedule can hand back and forth."""
        return self.elastic_gpus // self.tp

    def elastic_group_bundles(self, group_index: int) -> list[int]:
        """Global GPU bundle indices owned by one elastic TP group.

        Physical card order is A | E | R, NOT A | R | E. That is forced by the
        existing split plans rather than chosen: a rollout split always owns the
        highest bundles (_rollout_spec_bundles takes the tail range), so the
        cards rollout keeps in every plan are the last R, and the ones it only
        holds after expanding are the middle E. Worked through on the experiment
        topology (N=8, TP=2, A=2, R=2, E=4):

            [6,2] -> actor [0..5], rollout [6,7]
            [2,6] -> actor [0,1],  rollout [2..7]

        so [0,1] is always actor (A), [6,7] is always rollout (R), and [2,3,4,5]
        changes hands (E). Getting this backwards would pin the pool to cards the
        actor permanently owns, which does not fail loudly -- it silently
        double-books them.

        Groups are indexed low-to-high within E. A partial rollout split owns
        the tail of E because rollout itself owns the tail of the shared pool.
        """
        if not 0 <= group_index < self.num_elastic_groups:
            raise ValueError(
                f"elastic group index {group_index} outside [0, {self.num_elastic_groups - 1}]"
            )
        start = self.actor_gpus + group_index * self.tp
        return list(range(start, start + self.tp))

    def awake_elastic_group_indices(self, schedule_k: int) -> tuple[int, ...]:
        """Elastic groups owned by rollout at schedule ``k``.

        Rollout owns the highest shared-pool bundles, so a partial schedule
        takes the tail of E. For N=8, R=2, E=4, TP=2: k=1 means group 1
        ([4,5]), not group 0 ([2,3]).
        """
        if not 0 <= schedule_k <= self.num_elastic_groups:
            raise ValueError(
                f"schedule k={schedule_k} outside [0, {self.num_elastic_groups}]"
            )
        first = self.num_elastic_groups - schedule_k
        return tuple(range(first, self.num_elastic_groups))

    def schedule_from_rollout_cards(self, rollout_cards: int) -> int:
        """Derive schedule k from an existing split plan's rollout card count.

        k = (rollout_cards - R) / TP, k in [0, m]. Deriving k this way means the
        experiment scripts keep their current FIRST/SECOND_SPLIT_PLAN semantics
        and only gain R and E as new deployment parameters.

        Raises rather than clamping: a topology that does not land on a whole
        schedule point must fail loudly, never degrade into a silent full
        rebuild that would read as a passing run in the trace.
        """
        delta = rollout_cards - self.permanent_rollout_gpus
        if delta < 0:
            raise ValueError(
                f"rollout_cards={rollout_cards} is below the permanent rollout zone "
                f"R={self.permanent_rollout_gpus}; the permanent zone may never sleep "
                f"(section 2 constraint 5)"
            )
        if delta % self.tp != 0:
            raise ValueError(
                f"rollout_cards={rollout_cards} minus R={self.permanent_rollout_gpus} is not a whole "
                f"number of TP={self.tp} groups; refusing to fall back to a full rebuild"
            )
        k = delta // self.tp
        if k > self.num_elastic_groups:
            raise ValueError(
                f"schedule k={k} exceeds the elastic zone's m={self.num_elastic_groups} groups "
                f"(rollout_cards={rollout_cards}, R={self.permanent_rollout_gpus}, E={self.elastic_gpus})"
            )
        return k


@dataclass(frozen=True)
class AwakeTopology:
    """Immutable snapshot of the effective rollout topology at one schedule point.

    Every consumer reads these properties; nobody recomputes them. That is the
    entire purpose of this class -- see the module docstring for the failure it
    exists to prevent.

    Worker handles are held opaquely (this module never calls them), so the CPU
    tests can drive it with plain strings.
    """

    layout: ZoneLayout
    schedule_k: int
    permanent_rollout_workers: tuple
    elastic_group_workers: tuple
    actor_workers: tuple

    @classmethod
    def from_schedule(
        cls,
        layout: ZoneLayout,
        schedule_k: int,
        permanent_rollout_workers,
        elastic_group_workers,
        actor_workers,
    ) -> AwakeTopology:
        """Validate the pieces once, at construction, so consumers cannot be handed
        an inconsistent topology and then each discover a different part of it."""
        if not 0 <= schedule_k <= layout.num_elastic_groups:
            raise ValueError(
                f"schedule k={schedule_k} outside [0, {layout.num_elastic_groups}] for E={layout.elastic_gpus} "
                f"TP={layout.tp}"
            )
        groups = tuple(tuple(group) for group in elastic_group_workers)
        if len(groups) != layout.num_elastic_groups:
            raise ValueError(
                f"expected {layout.num_elastic_groups} elastic groups (m = E / TP), got {len(groups)}"
            )
        for index, group in enumerate(groups):
            if len(group) != layout.tp:
                raise ValueError(
                    f"elastic group {index} has {len(group)} workers, expected TP={layout.tp}"
                )
        permanent = tuple(permanent_rollout_workers)
        if len(permanent) != layout.permanent_rollout_gpus:
            raise ValueError(
                f"permanent rollout zone has {len(permanent)} workers, expected R={layout.permanent_rollout_gpus}"
            )
        return cls(
            layout=layout,
            schedule_k=schedule_k,
            permanent_rollout_workers=permanent,
            elastic_group_workers=groups,
            actor_workers=tuple(actor_workers),
        )

    @property
    def awake_elastic_indices(self) -> tuple:
        """Groups handed to rollout at this schedule point."""
        return self.layout.awake_elastic_group_indices(self.schedule_k)

    @property
    def sleeping_elastic_indices(self) -> tuple:
        awake = set(self.awake_elastic_indices)
        return tuple(index for index in range(self.layout.num_elastic_groups) if index not in awake)

    @property
    def awake_rollout_workers(self) -> tuple:
        """Ordered handles. List position pins the replica rank: agent_loop slices
        workers[rank*TP:(rank+1)*TP] and ExternalZeroMQDistributedExecutor slices
        the ZMQ addresses identically, so this order is a contract, not a
        convenience. Physical bundle order is awake elastic groups first, then
        the permanent rollout tail, matching ``diff_engines``.
        """
        workers = []
        for index in self.awake_elastic_indices:
            workers.extend(self.elastic_group_workers[index])
        workers.extend(self.permanent_rollout_workers)
        return tuple(workers)

    @property
    def awake_rollout_bundles(self) -> tuple:
        """Physical bundle order corresponding to ``awake_rollout_workers``."""
        bundles = []
        for index in self.awake_elastic_indices:
            bundles.extend(self.layout.elastic_group_bundles(index))
        permanent_start = self.layout.total_gpus - self.layout.permanent_rollout_gpus
        bundles.extend(range(permanent_start, self.layout.total_gpus))
        return tuple(bundles)

    def elastic_replica_rank(self, group_index: int) -> int:
        """Replica rank of one awake elastic group in physical bundle order."""
        if group_index not in self.awake_elastic_indices:
            raise ValueError(f"elastic group {group_index} is not awake")
        return self.awake_elastic_indices.index(group_index)

    @property
    def sleeping_workers(self) -> tuple:
        """Handles that must NOT appear in rollout_wg or the collective (G4)."""
        workers = []
        for index in self.sleeping_elastic_indices:
            workers.extend(self.elastic_group_workers[index])
        return tuple(workers)

    @property
    def num_replicas(self) -> int:
        return len(self.awake_rollout_workers) // self.layout.tp

    @property
    def collective_members(self) -> tuple:
        """G2: the weight-sync collective is actor workers plus AWAKE rollout
        workers only. sync_rollout_weights broadcasts to the whole group, so a
        sleeping member would hang it rather than merely miss an update."""
        return tuple(self.actor_workers) + self.awake_rollout_workers

    def validate_src_rank(self, src_rank: int) -> int:
        """G2: the broadcast source must be a live actor member of the collective.

        src_rank is a staging-policy decision, not a topology one -- ray_trainer's
        _sync_rollout_weights_after_resize_async picks 1 when use_alt_src holds
        (pinned_cpu backend, optimizer not staged, more than one actor worker).
        This validates that choice against the actual actor set instead of
        re-deriving it, which would fork staging policy into a topology object.
        """
        if not 0 <= src_rank < len(self.actor_workers):
            raise ValueError(
                f"src_rank={src_rank} is not a live actor member: the collective has "
                f"{len(self.actor_workers)} actor workers"
            )
        return src_rank
