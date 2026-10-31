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
"""awake_topology: zone geometry and the single engine-diff algorithm.

Imported by file path so the suite stays runnable without torch/ray. If this
import ever needs a heavy dependency, that is a bug in awake_topology.py.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "verl"
    / "experimental"
    / "one_step_off_policy"
    / "awake_topology.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("awake_topology_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: under `from __future__ import annotations`,
    # @dataclass resolves its string annotations via sys.modules[cls.__module__],
    # which is None for an unregistered module -> AttributeError at class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def at():
    return _load_module()


# --- partition_engines ---------------------------------------------------------


def test_partition_groups_by_tp(at):
    assert at.partition_engines([2, 3, 4, 5, 6, 7], tp=2) == [(2, 3), (4, 5), (6, 7)]


def test_partition_tp1_is_one_card_per_engine(at):
    assert at.partition_engines([5, 6, 7], tp=1) == [(5,), (6,), (7,)]


@pytest.mark.parametrize("bundles,tp", [([], 2), ([1, 2, 3], 2), ([1, 2], 0), ([1, 2], -1)])
def test_partition_returns_none_when_full_rebuild_required(at, bundles, tp):
    assert at.partition_engines(bundles, tp) is None


# --- diff_engines: identical semantics to the pre-refactor survivor plan --------


def test_diff_scenario_a_expansion_tp2(at):
    """[6,2]->[2,6] TP=2: old engine (6,7) survives, landing at new_rank 2."""
    plan = at.diff_engines([6, 7], [2, 3, 4, 5, 6, 7], tp=2)
    assert plan["has_plan"] is True
    assert [(s["old_rank"], s["new_rank"], s["bundles"]) for s in plan["survivors"]] == [(0, 2, [6, 7])]
    assert plan["to_build_new_ranks"] == [0, 1]
    assert plan["to_kill_old_ranks"] == []


def test_diff_scenario_b_expansion_tp1(at):
    """[6,2]->[5,3] TP=1: both old engines survive but shift rank; one to build."""
    plan = at.diff_engines([6, 7], [5, 6, 7], tp=1)
    assert plan["has_plan"] is True
    pairs = sorted((s["old_rank"], s["new_rank"], tuple(s["bundles"])) for s in plan["survivors"])
    assert pairs == [(0, 1, (6,)), (1, 2, (7,))]
    assert plan["to_build_new_ranks"] == [0]
    assert plan["to_kill_old_ranks"] == []


def test_diff_pure_shrink_kills_the_engines_that_leave(at):
    plan = at.diff_engines([2, 3, 4, 5, 6, 7], [6, 7], tp=2)
    assert [(s["old_rank"], s["new_rank"]) for s in plan["survivors"]] == [(2, 0)]
    assert plan["to_kill_old_ranks"] == [0, 1]
    assert plan["to_build_new_ranks"] == []


def test_diff_non_divisible_reports_no_plan_rather_than_raising(at):
    """Preserves the pre-refactor contract: the CALLER decides to full-rebuild.
    The trainer-level guard that raises on non-divisible lives in ZoneLayout."""
    plan = at.diff_engines([4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], tp=2)
    assert plan["has_plan"] is False
    assert plan["survivors"] == []
    assert plan["to_build_new_ranks"] == []
    assert plan["to_kill_old_ranks"] == []


def test_diff_disjoint_topologies_have_no_survivors(at):
    plan = at.diff_engines([0, 1], [6, 7], tp=2)
    assert plan["has_plan"] is True
    assert plan["survivors"] == []
    assert plan["to_kill_old_ranks"] == [0]
    assert plan["to_build_new_ranks"] == [0]


def test_survivor_bundles_are_sorted_so_identity_is_order_independent(at):
    plan = at.diff_engines([7, 6], [6, 7], tp=2)
    assert plan["survivors"] == [{"old_rank": 0, "new_rank": 0, "bundles": [6, 7]}]


# --- ZoneLayout ----------------------------------------------------------------


def test_zone_layout_experiment_geometry(at):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    assert layout.actor_gpus == 2
    assert layout.num_elastic_groups == 2


@pytest.mark.parametrize(
    "rollout_cards,expected_k",
    [(2, 0), (4, 1), (6, 2)],
)
def test_schedule_derived_from_existing_split_plan(at, rollout_cards, expected_k):
    """k = (rollout_cards - R) / TP. [6,2] -> k=0, [2,6] -> k=2, so the existing
    split-plan semantics carry over untouched."""
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    assert layout.schedule_from_rollout_cards(rollout_cards) == expected_k


@pytest.mark.parametrize("rollout_cards", [1, 3, 5, 8, 0])
def test_schedule_raises_rather_than_silently_full_rebuilding(at, rollout_cards):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    with pytest.raises(ValueError):
        layout.schedule_from_rollout_cards(rollout_cards)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_gpus": 8, "permanent_rollout_gpus": 3, "elastic_gpus": 4, "tp": 2},  # R % TP
        {"total_gpus": 8, "permanent_rollout_gpus": 2, "elastic_gpus": 3, "tp": 2},  # E % TP
        {"total_gpus": 8, "permanent_rollout_gpus": 2, "elastic_gpus": 8, "tp": 2},  # A < 0
        {"total_gpus": 8, "permanent_rollout_gpus": 2, "elastic_gpus": 4, "tp": 0},  # TP <= 0
        {"total_gpus": 8, "permanent_rollout_gpus": -2, "elastic_gpus": 4, "tp": 2},  # negative
    ],
)
def test_zone_layout_rejects_invalid_geometry(at, kwargs):
    with pytest.raises(ValueError):
        at.ZoneLayout(**kwargs)


def test_module_is_stdlib_only(at):
    """The CPU suite and the trainer both import this at module scope."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for heavy in ("import torch", "import ray", "import vllm", "from verl", "import numpy"):
        assert heavy not in source, f"awake_topology.py must not depend on {heavy}"


# --- elastic zone card mapping -------------------------------------------------


def _spec_bundles(size, index):
    """Mirror of ray_trainer._rollout_spec_bundles: split index i owns the
    contiguous tail range [sum(size[:i]), sum(size[:i]) + size[i])."""
    start = sum(size[:index])
    return list(range(start, start + size[index]))


def test_elastic_bundles_match_the_real_split_plans(at):
    """Zone order is A | E | R, derived from the split plans.

    A rollout split always owns the HIGHEST bundles, so the cards rollout keeps
    across both plans are the permanent R zone, and the ones it only gains on
    expansion are the elastic zone. Reversing this to A | R | E would pin the
    pool onto cards the actor permanently owns -- which does not fail loudly, it
    silently double-books them.
    """
    first_rollout = set(_spec_bundles([6, 2], 1))    # [6,2] -> rollout [6,7]
    second_rollout = set(_spec_bundles([2, 6], 1))   # [2,6] -> rollout [2..7]
    permanent = sorted(first_rollout & second_rollout)
    elastic = sorted(second_rollout - first_rollout)
    actor = sorted(set(_spec_bundles([6, 2], 0)) & set(_spec_bundles([2, 6], 0)))
    assert (actor, elastic, permanent) == ([0, 1], [2, 3, 4, 5], [6, 7])

    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    assert list(range(layout.actor_gpus)) == actor
    from_layout = [b for g in range(layout.num_elastic_groups) for b in layout.elastic_group_bundles(g)]
    assert from_layout == elastic


def test_elastic_groups_are_contiguous_tp_sized_and_ordered(at):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    assert layout.elastic_group_bundles(0) == [2, 3]
    assert layout.elastic_group_bundles(1) == [4, 5]


def test_elastic_bundles_tp1(at):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=1, elastic_gpus=3, tp=1)
    assert [layout.elastic_group_bundles(g) for g in range(3)] == [[4], [5], [6]]


@pytest.mark.parametrize("bad_index", [-1, 2, 99])
def test_elastic_bundles_reject_out_of_range_group(at, bad_index):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    with pytest.raises(ValueError):
        layout.elastic_group_bundles(bad_index)


def test_elastic_bundles_never_overlap_the_permanent_zones(at):
    """Exhaustive over every valid geometry at N=8: the pool must never be handed
    a card belonging to A or R."""
    for tp in (1, 2, 4):
        for r in range(0, 9, tp):
            for e in range(tp, 9, tp):
                if r + e > 8:
                    continue
                layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=r, elastic_gpus=e, tp=tp)
                actor = set(range(layout.actor_gpus))
                permanent = set(range(8 - r, 8))
                got = [b for g in range(layout.num_elastic_groups) for b in layout.elastic_group_bundles(g)]
                assert len(got) == e
                assert set(got).isdisjoint(actor), f"tp={tp} R={r} E={e}: elastic overlaps actor"
                assert set(got).isdisjoint(permanent), f"tp={tp} R={r} E={e}: elastic overlaps permanent rollout"
                assert got == sorted(got)


# --- AwakeTopology -------------------------------------------------------------


def _make_topology(at, schedule_k, *, tp=2, r=2, e=4, actors=2):
    """Experiment geometry with string stand-ins for Ray handles."""
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=r, elastic_gpus=e, tp=tp)
    return at.AwakeTopology.from_schedule(
        layout=layout,
        schedule_k=schedule_k,
        permanent_rollout_workers=[f"perm{i}" for i in range(r)],
        elastic_group_workers=[
            [f"elastic{g}_w{w}" for w in range(tp)] for g in range(e // tp)
        ],
        actor_workers=[f"actor{i}" for i in range(actors)],
    )


def test_k0_all_elastic_groups_asleep(at):
    topo = _make_topology(at, schedule_k=0)
    assert topo.awake_elastic_indices == ()
    assert topo.sleeping_elastic_indices == (0, 1)
    assert topo.awake_rollout_workers == ("perm0", "perm1")
    assert topo.awake_rollout_bundles == (6, 7)
    assert topo.num_replicas == 1


def test_k2_all_elastic_groups_awake(at):
    topo = _make_topology(at, schedule_k=2)
    assert topo.awake_elastic_indices == (0, 1)
    assert topo.sleeping_elastic_indices == ()
    assert topo.num_replicas == 3
    assert topo.sleeping_workers == ()


def test_awake_worker_order_pins_rank(at):
    """agent_loop slices workers[rank*TP:(rank+1)*TP] and the ZMQ executor slices
    addresses the same way, so position == rank is a hard contract."""
    topo = _make_topology(at, schedule_k=2)
    assert topo.awake_rollout_workers == (
        "elastic0_w0",
        "elastic0_w1",
        "elastic1_w0",
        "elastic1_w1",
        "perm0",
        "perm1",
    )
    assert topo.awake_rollout_bundles == (2, 3, 4, 5, 6, 7)
    assert topo.elastic_replica_rank(0) == 0
    assert topo.elastic_replica_rank(1) == 1
    for rank in range(topo.num_replicas):
        group = topo.awake_rollout_workers[rank * 2 : (rank + 1) * 2]
        assert len(group) == 2


def test_partial_schedule_wakes_the_tail_elastic_group(at):
    topo = _make_topology(at, schedule_k=1)
    assert topo.awake_elastic_indices == (1,)
    assert topo.sleeping_elastic_indices == (0,)
    assert topo.sleeping_workers == ("elastic0_w0", "elastic0_w1")
    assert topo.awake_rollout_workers == (
        "elastic1_w0",
        "elastic1_w1",
        "perm0",
        "perm1",
    )
    assert topo.awake_rollout_bundles == (4, 5, 6, 7)
    assert topo.elastic_replica_rank(1) == 0
    assert topo.num_replicas == 2


def test_collective_excludes_sleeping_workers(at):
    """G2/G4: sync_rollout_weights broadcasts to the whole group, so a sleeping
    member would hang it."""
    topo = _make_topology(at, schedule_k=1)
    members = set(topo.collective_members)
    assert members == {"actor0", "actor1", "perm0", "perm1", "elastic1_w0", "elastic1_w1"}
    assert members.isdisjoint(set(topo.sleeping_workers))


def test_collective_members_start_with_actors(at):
    topo = _make_topology(at, schedule_k=0)
    assert topo.collective_members[:2] == ("actor0", "actor1")


def test_validate_src_rank_accepts_live_actor_rank(at):
    topo = _make_topology(at, schedule_k=1, actors=2)
    assert topo.validate_src_rank(0) == 0
    assert topo.validate_src_rank(1) == 1


def test_validate_src_rank_rejects_rank_outside_actor_set(at):
    """use_alt_src picks src_rank=1, which is only valid with >1 actor worker."""
    topo = _make_topology(at, schedule_k=1, actors=1)
    with pytest.raises(ValueError):
        topo.validate_src_rank(1)


@pytest.mark.parametrize("bad_k", [-1, 3])
def test_from_schedule_rejects_out_of_range_k(at, bad_k):
    with pytest.raises(ValueError):
        _make_topology(at, schedule_k=bad_k)


def test_from_schedule_rejects_wrong_group_count(at):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    with pytest.raises(ValueError):
        at.AwakeTopology.from_schedule(
            layout=layout,
            schedule_k=1,
            permanent_rollout_workers=["perm0", "perm1"],
            elastic_group_workers=[["e0", "e1"]],  # m is 2, only 1 supplied
            actor_workers=["actor0"],
        )


def test_from_schedule_rejects_group_that_is_not_tp_sized(at):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    with pytest.raises(ValueError):
        at.AwakeTopology.from_schedule(
            layout=layout,
            schedule_k=1,
            permanent_rollout_workers=["perm0", "perm1"],
            elastic_group_workers=[["e0"], ["e1", "e2"]],
            actor_workers=["actor0"],
        )


def test_from_schedule_rejects_wrong_permanent_zone_size(at):
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    with pytest.raises(ValueError):
        at.AwakeTopology.from_schedule(
            layout=layout,
            schedule_k=1,
            permanent_rollout_workers=["perm0"],  # R is 2
            elastic_group_workers=[["e0", "e1"], ["e2", "e3"]],
            actor_workers=["actor0"],
        )


def test_topology_is_immutable(at):
    """Consumers read; nobody recomputes or mutates."""
    import dataclasses

    topo = _make_topology(at, schedule_k=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        topo.schedule_k = 2


def test_scenario_a_expansion_is_a_schedule_change(at):
    """[6,2] -> [2,6] is k 0 -> 2 under N=8 TP=2 A=2 R=2 E=4."""
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    assert layout.schedule_from_rollout_cards(2) == 0
    assert layout.schedule_from_rollout_cards(6) == 2
    before = _make_topology(at, schedule_k=0)
    after = _make_topology(at, schedule_k=2)
    assert before.num_replicas == 1
    assert after.num_replicas == 3
    assert set(before.sleeping_workers) == set(after.awake_rollout_workers) - set(
        before.awake_rollout_workers
    )
