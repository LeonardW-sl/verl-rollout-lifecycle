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
"""ElasticEnginePool driven with fake handles (no ray, no GPU)."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import pathlib
import sys

import pytest

_PKG = pathlib.Path(__file__).resolve().parents[3] / "verl" / "experimental" / "one_step_off_policy"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: under `from __future__ import annotations`,
    # @dataclass resolves its string annotations via sys.modules[cls.__module__],
    # which is None for an unregistered module -> AttributeError at class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def at():
    return _load("awake_topology")


@pytest.fixture(scope="module")
def pool_mod():
    return _load("elastic_engine_pool")


class _FakeWorker:
    def __init__(self, name):
        self.name = name
        self._actor_id = name

    def __repr__(self):
        return f"<FakeWorker {self.name}>"


class _FakeReplica:
    def __init__(self, workers, *, delay=0.0, fail_on=None):
        self.workers = tuple(workers)
        self.replica_rank = -1
        self.calls = []
        self.delay = delay
        self.fail_on = fail_on

    async def sleep(self):
        self.calls.append("sleep")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_on == "sleep":
            raise RuntimeError("engine refused to sleep")

    async def wake_up(self):
        self.calls.append("wake_up")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_on == "wake_up":
            raise RuntimeError("engine refused to wake")


def _layout(at, tp=2, r=2, e=4):
    return at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=r, elastic_gpus=e, tp=tp)


def _pool_with_groups(at, pool_mod, awake_indices=(0, 1), *, register_replicas=True):
    layout = _layout(at)
    pool = pool_mod.ElasticEnginePool(layout)
    groups = {}
    for index in range(layout.num_elastic_groups):
        handles = [_FakeWorker(f"g{index}_w{w}") for w in range(layout.tp)]
        groups[index] = handles
        pool.register_group(index, handles, awake=index in awake_indices)
        if register_replicas:
            pool.register_replica(index, _FakeReplica(handles))
    return pool, groups


def test_register_tracks_awake_and_sleeping(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=(0,))
    assert pool.awake_indices == (0,)
    assert pool.sleeping_indices == (1,)


def test_register_rejects_wrong_group_size(at, pool_mod):
    pool = pool_mod.ElasticEnginePool(_layout(at))
    with pytest.raises(ValueError):
        pool.register_group(0, [_FakeWorker("only-one")], awake=True)


def test_register_rejects_out_of_range_index(at, pool_mod):
    pool = pool_mod.ElasticEnginePool(_layout(at))
    with pytest.raises(ValueError):
        pool.register_group(5, [_FakeWorker("a"), _FakeWorker("b")], awake=True)


def test_register_rejects_duplicate_index(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod)
    with pytest.raises(ValueError):
        pool.register_group(0, [_FakeWorker("a"), _FakeWorker("b")], awake=True)


def test_register_replica_rejects_mismatched_workers(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, register_replicas=False)
    with pytest.raises(ValueError, match="replica workers do not match"):
        pool.register_replica(0, _FakeReplica([_FakeWorker("x"), _FakeWorker("y")]))


def test_register_replica_rejects_duplicate(at, pool_mod):
    pool, groups = _pool_with_groups(at, pool_mod)
    with pytest.raises(ValueError, match="already has a replica"):
        pool.register_replica(0, _FakeReplica(groups[0]))


def test_group_requires_replica_before_transition(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, register_replicas=False)
    with pytest.raises(ValueError, match="have no registered replica"):
        asyncio.run(pool.sleep_groups([0]))


def test_wake_ignores_indices_the_pool_never_registered(at, pool_mod):
    """wake_groups filters on membership, not just awake-ness.

    Dropping the `i in self._groups` half of that filter raises KeyError inside
    the gather -- during a resize that aborts the wake with some groups already
    remapped, which is the worst possible failure point. reconcile() derives its
    indices from an AwakeTopology, so a pool that was built from a different
    schedule can legitimately be asked about a group it does not hold.
    """
    pool, groups = _pool_with_groups(at, pool_mod, awake_indices=())
    elapsed = asyncio.run(pool.wake_groups([0, 99]))
    assert elapsed >= 0.0
    assert pool.awake_indices == (0,)
    assert pool.replica(0).calls == ["wake_up"]


def test_transition_durations_are_measured_not_hardcoded(at, pool_mod):
    """pool/sleep_s and pool/wake_s record real durations.

    A constant 0.0 would read as an implausibly perfect result rather than as a
    broken probe, so assert the value tracks real elapsed time.
    """
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=(0, 1))

    pool._replicas[0] = _FakeReplica(pool.handles(0), delay=0.05)
    elapsed = asyncio.run(pool.sleep_groups([0]))
    assert elapsed >= 0.05, f"elapsed {elapsed} did not measure the 50ms sleep"


def test_index_views_are_sorted(at, pool_mod):
    """awake_handles() relies on awake_indices being ordered, and rank is pinned
    by position, so a set-iteration order leak would misroute a replica."""
    # N=10 keeps A=2 while giving 4 elastic groups, so registration order can be
    # scrambled independently of index order.
    layout = at.ZoneLayout(total_gpus=10, permanent_rollout_gpus=2, elastic_gpus=8, tp=2)
    pool = pool_mod.ElasticEnginePool(layout)
    for index in (3, 0, 2, 1):
        handles = [_FakeWorker(f"g{index}_a"), _FakeWorker(f"g{index}_b")]
        pool.register_group(index, handles, awake=index in (3, 0, 2, 1))
        pool.register_replica(index, _FakeReplica(handles))
    assert pool.awake_indices == (0, 1, 2, 3)
    asyncio.run(pool.sleep_groups([2, 0]))
    assert pool.awake_indices == (1, 3)
    assert pool.sleeping_indices == (0, 2)


def test_sleep_calls_replica_once(at, pool_mod):
    pool, groups = _pool_with_groups(at, pool_mod)
    elapsed = asyncio.run(pool.sleep_groups([1]))
    assert elapsed >= 0.0
    assert pool.replica(1).calls == ["sleep"]
    assert pool.replica(0).calls == []
    assert pool.awake_indices == (0,)


def test_wake_calls_wake_up_not_sleep(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=())
    asyncio.run(pool.wake_groups([0, 1]))
    for index in (0, 1):
        assert pool.replica(index).calls == ["wake_up"]
    assert pool.awake_indices == (0, 1)


def test_sleeping_an_already_asleep_group_is_a_noop(at, pool_mod):
    """Idempotent: vLLM sleep on a slept engine is not safe to repeat blindly."""
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=())
    asyncio.run(pool.sleep_groups([0]))
    assert pool.replica(0).calls == []
    assert pool.sleeping_indices == (0, 1)


def test_waking_an_already_awake_group_is_a_noop(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=(0, 1))
    asyncio.run(pool.wake_groups([0]))
    assert pool.replica(0).calls == []


def test_state_is_unchanged_when_a_replica_raises(at, pool_mod):
    """A half-slept group must not be recorded as fully asleep -- that would put a
    live engine outside rollout_wg with no way back (G4)."""
    pool, groups = _pool_with_groups(at, pool_mod)

    pool._replicas[1] = _FakeReplica(groups[1], fail_on="sleep")
    with pytest.raises(RuntimeError):
        asyncio.run(pool.sleep_groups([1]))
    assert pool.awake_indices == (0, 1)


def test_reconcile_targets_the_topology_schedule(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=(0, 1))
    layout = _layout(at)
    shrunk = at.AwakeTopology.from_schedule(
        layout=layout,
        schedule_k=0,
        permanent_rollout_workers=["p0", "p1"],
        elastic_group_workers=[["a", "b"], ["c", "d"]],
        actor_workers=["actor0", "actor1"],
    )
    to_sleep, to_wake = pool.reconcile(shrunk)
    assert to_sleep == [0, 1]
    assert to_wake == []


def test_reconcile_on_expansion(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=())
    layout = _layout(at)
    grown = at.AwakeTopology.from_schedule(
        layout=layout,
        schedule_k=2,
        permanent_rollout_workers=["p0", "p1"],
        elastic_group_workers=[["a", "b"], ["c", "d"]],
        actor_workers=["actor0", "actor1"],
    )
    to_sleep, to_wake = pool.reconcile(grown)
    assert to_sleep == []
    assert to_wake == [0, 1]


def test_reconcile_is_empty_when_already_at_target(at, pool_mod):
    pool, _ = _pool_with_groups(at, pool_mod, awake_indices=(1,))
    layout = _layout(at)
    same = at.AwakeTopology.from_schedule(
        layout=layout,
        schedule_k=1,
        permanent_rollout_workers=["p0", "p1"],
        elastic_group_workers=[["a", "b"], ["c", "d"]],
        actor_workers=["actor0", "actor1"],
    )
    assert pool.reconcile(same) == ([], [])


def test_expansion_merges_pool_and_permanent_replicas(at, pool_mod):
    pool, groups = _pool_with_groups(at, pool_mod, awake_indices=())
    permanent = _FakeReplica([_FakeWorker("perm0"), _FakeWorker("perm1")])
    topology = at.AwakeTopology.from_schedule(
        layout=_layout(at),
        schedule_k=2,
        permanent_rollout_workers=permanent.workers,
        elastic_group_workers=[groups[0], groups[1]],
        actor_workers=["actor0", "actor1"],
    )
    survivor_plan = {
        "survivors": [{"old_rank": 0, "new_rank": 2, "bundles": [6, 7]}]
    }
    reuse, keep = pool.build_reuse_plan(topology, survivor_plan, [permanent])
    assert reuse == {0: pool.replica(0), 1: pool.replica(1), 2: permanent}
    assert keep == {0}


def test_shrink_keeps_pool_replica_outside_target(at, pool_mod):
    pool, groups = _pool_with_groups(at, pool_mod, awake_indices=(1,))
    permanent = _FakeReplica([_FakeWorker("perm0"), _FakeWorker("perm1")])
    topology = at.AwakeTopology.from_schedule(
        layout=_layout(at),
        schedule_k=0,
        permanent_rollout_workers=permanent.workers,
        elastic_group_workers=[groups[0], groups[1]],
        actor_workers=["actor0", "actor1"],
    )
    survivor_plan = {
        "survivors": [{"old_rank": 1, "new_rank": 0, "bundles": [6, 7]}]
    }
    reuse, keep = pool.build_reuse_plan(
        topology,
        survivor_plan,
        [pool.replica(1), permanent],
    )
    assert reuse == {0: permanent}
    assert keep == {0, 1}


def test_awake_handles_are_ordered_by_group_index(at, pool_mod):
    pool, groups = _pool_with_groups(at, pool_mod, awake_indices=(0, 1))
    assert list(pool.awake_handles()) == groups[0] + groups[1]


def test_awake_handles_skip_sleeping_groups_and_sort_by_index(at, pool_mod):
    """Registration order must not leak into the handle order, and a sleeping group
    must not appear at all.

    Both halves are load-bearing for G4: this tuple is appended after the permanent
    zone to rebuild rollout_wg, and list position pins the replica rank (agent_loop
    slices workers[rank*TP:(rank+1)*TP]). A sleeping handle in the list hangs the
    weight-sync broadcast; a mis-ordered one silently misroutes a replica.
    """
    layout = _layout(at)
    sleeping = pool_mod.ElasticEnginePool(layout)
    g1 = [_FakeWorker("g1_w0"), _FakeWorker("g1_w1")]
    g0 = [_FakeWorker("g0_w0"), _FakeWorker("g0_w1")]
    sleeping.register_group(1, g1, awake=True)
    sleeping.register_group(0, g0, awake=False)
    assert list(sleeping.awake_handles()) == g1

    both_awake = pool_mod.ElasticEnginePool(layout)
    h1 = [_FakeWorker("h1_w0"), _FakeWorker("h1_w1")]
    h0 = [_FakeWorker("h0_w0"), _FakeWorker("h0_w1")]
    both_awake.register_group(1, h1, awake=True)
    both_awake.register_group(0, h0, awake=True)
    assert list(both_awake.awake_handles()) == h0 + h1


def test_pool_does_not_import_ray(pool_mod):
    source = (_PKG / "elastic_engine_pool.py").read_text(encoding="utf-8")
    for heavy in ("import ray", "import torch", "import vllm"):
        assert heavy not in source


def test_pool_imports_only_stdlib(pool_mod):
    """Stronger than the substring scan above, which a docstring could satisfy by
    accident and which cannot see a relative `from .awake_topology import ...`.

    The CPU suite loads this module by file path with no verl package on sys.path,
    so any package-coupled import breaks every test here, not just one.
    """
    tree = ast.parse((_PKG / "elastic_engine_pool.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 marks a relative import, i.e. one that needs the package
            imported.add("." * node.level + (node.module or "").split(".")[0])
    assert imported <= {"__future__", "asyncio", "time"}, f"unexpected imports: {sorted(imported)}"
