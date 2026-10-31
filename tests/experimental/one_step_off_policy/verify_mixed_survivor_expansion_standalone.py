"""Design-oracle / no-deps check for mixed-survivor geometry (NOT a production test).

This file **ports** plan/merge algorithms so it can run without ray/torch/pytest.
It does **not** import ``ray_trainer.py``. For tests bound to the live source, use
``test_mixed_survivor_expansion_on_cpu.py`` (AST-loads real
``_compute_rollout_survivor_plan``). Keep this as a fast offline geometry oracle only.
"""

import sys

import numpy as np


class FakePG:
    def __init__(self, bundle_count):
        self.bundle_count = bundle_count


class RayResourcePool:
    def __init__(self, process_on_nodes=None, use_gpu=True, name_prefix="", max_colocate_count=10, **kw):
        self.store = process_on_nodes or [8]
        self.use_gpu = use_gpu
        self.name_prefix = name_prefix
        self.max_colocate_count = max_colocate_count
        self.pgs = None

    @property
    def world_size(self):
        return sum(self.store)

    def get_placement_groups(self, **kw):
        if self.pgs is None:
            self.pgs = [FakePG(n) for n in self.store]
        return self.pgs


class SubRayResourcePool(RayResourcePool):
    def __init__(self, placement_groups, start_bundle_index, subgroup_world_size, **kwargs):
        super().__init__(**kwargs)
        self.pgs = placement_groups
        self.start_bundle_index = start_bundle_index
        self.subgroup_world_size = subgroup_world_size

    @property
    def world_size(self):
        return self.subgroup_world_size


def split_resource_pool(resource_pool, split_size):
    """Port of single_controller/ray/base.py:193-238."""
    if isinstance(split_size, int):
        assert resource_pool.world_size % split_size == 0
        split_size_list = [split_size] * (resource_pool.world_size // split_size)
    else:
        split_size_list = list(split_size)
    assert sum(split_size_list) == resource_pool.world_size

    if isinstance(resource_pool, SubRayResourcePool):
        start_idx_list = np.cumsum([resource_pool.start_bundle_index] + split_size_list[:-1])
    else:
        start_idx_list = np.cumsum([0] + split_size_list[:-1])

    pgs = resource_pool.get_placement_groups()
    return [
        SubRayResourcePool(
            process_on_nodes=resource_pool.store,
            use_gpu=resource_pool.use_gpu,
            name_prefix=f"{resource_pool.name_prefix}_split_{i}",
            max_colocate_count=resource_pool.max_colocate_count,
            placement_groups=pgs,
            start_bundle_index=int(start_idx_list[i]),
            subgroup_world_size=split_size_list[i],
        )
        for i in range(len(split_size_list))
    ]


def spec_bundles(spec):
    """Port of _rollout_spec_bundles."""
    size, index = list(spec["size"]), spec.get("index", 0)
    start = sum(size[:index])
    return list(range(start, start + size[index]))


def compute_survivor_plan(old_bundles, new_bundles, tp):
    """Port of _compute_rollout_survivor_plan (ray_trainer.py:2305-2353)."""
    if not old_bundles or not new_bundles or tp <= 0:
        return {"has_plan": False, "survivors": [], "to_kill_old_ranks": [], "to_build_new_ranks": []}
    if len(old_bundles) % tp or len(new_bundles) % tp:
        return {"has_plan": False, "survivors": [], "to_kill_old_ranks": [], "to_build_new_ranks": []}

    def engines(b):
        return [tuple(sorted(b[i : i + tp])) for i in range(0, len(b), tp)]

    old_by = {e: r for r, e in enumerate(engines(old_bundles))}
    new_by = {e: r for r, e in enumerate(engines(new_bundles))}
    return {
        "has_plan": True,
        "survivors": [
            {"old_rank": old_by[e], "new_rank": nr, "bundles": list(e)}
            for nr, e in enumerate(engines(new_bundles))
            if e in old_by
        ],
        "to_kill_old_ranks": sorted(r for e, r in old_by.items() if e not in new_by),
        "to_build_new_ranks": sorted(r for e, r in new_by.items() if e not in old_by),
    }


def resolve_pool(spec, base_world=8):
    """Port of _resolve_role_pool for mode=='split'."""
    base = RayResourcePool(process_on_nodes=[base_world])
    return split_resource_pool(base, list(spec["size"]))[spec.get("index", 0)]


def run_patch(old_spec, new_spec, tp, allow_mixed, old_workers, *, drop_rank=None):
    """The patched decision + merge logic, transcribed from the edited source.

    drop_rank: test hook to punch a hole in handles_by_new_rank so the
    missing_ranks guard can be exercised.
    """
    plan = compute_survivor_plan(spec_bundles(old_spec), spec_bundles(new_spec), tp)

    survivor_base = plan is not None and plan["has_plan"] and len(plan["survivors"]) > 0
    to_build = plan["to_build_new_ranks"] if plan else []
    use_survivor = survivor_base and (len(to_build) == 0 or allow_mixed)
    use_mixed = use_survivor and len(to_build) > 0
    if not use_survivor:
        return {"path": "full_rebuild", "use_mixed": False, "plan": plan}

    handles_by_new_rank, survivor_handles = {}, []
    for s in sorted(plan["survivors"], key=lambda x: x["new_rank"]):
        st, en = tp * s["old_rank"], tp * (s["old_rank"] + 1)
        if en > len(old_workers):
            raise IndexError(f"survivor old_rank {s['old_rank']} out of bounds")
        h = old_workers[st:en]
        survivor_handles.extend(h)
        handles_by_new_rank[s["new_rank"]] = list(h)

    pool = resolve_pool(new_spec)
    if pool is None:
        raise ValueError("unresolvable rollout pool spec")
    expected_cards = int(pool.world_size)
    if expected_cards % tp:
        raise ValueError(f"new rollout cards {expected_cards} not divisible by TP {tp}")
    num_new_replicas = expected_cards // tp

    delta_ranks = sorted(r for r in to_build if r < num_new_replicas)
    built = []
    if delta_ranks:
        replica_pools = split_resource_pool(pool, tp)
        for nr in delta_ranks:
            p = replica_pools[nr]
            bundles = list(range(p.start_bundle_index, p.start_bundle_index + p.world_size))
            if len(bundles) != tp:
                raise ValueError(f"delta group for new_rank {nr} has {len(bundles)} workers, expected TP={tp}")
            handles_by_new_rank[nr] = [f"delta_r{nr}_w{i}(gpu{b})" for i, b in enumerate(bundles)]
            built.append((nr, bundles))

    if drop_rank is not None:
        handles_by_new_rank.pop(drop_rank, None)

    missing = [r for r in range(num_new_replicas) if r not in handles_by_new_rank]
    if missing:
        raise ValueError(f"merged rollout group is missing workers for new_ranks {missing}")

    merged = []
    for nr in range(num_new_replicas):
        merged.extend(handles_by_new_rank[nr])

    return {
        "path": "survivor",
        "use_mixed": use_mixed,
        "plan": plan,
        "num_new_replicas": num_new_replicas,
        "merged": merged,
        "built": built,
        "survivor_handles": survivor_handles,
        "handles_by_new_rank": handles_by_new_rank,
        "tp": tp,
    }


def old_worker_list(old_spec):
    """Old rollout group has one worker per card it owned -- NOT tp workers."""
    return [f"old_w{i}" for i in range(len(spec_bundles(old_spec)))]


def main():
    results = []

    def check(label, cond):
        results.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    TP = 2
    OLD = {"size": [6, 2], "index": 1}
    NEW = {"size": [2, 6], "index": 1}

    print("=== mixed expansion scenario: [6,2] -> [2,6], TP=2 ===")
    plan = compute_survivor_plan(spec_bundles(OLD), spec_bundles(NEW), TP)
    print(f"  survivors={[(s['old_rank'], s['new_rank']) for s in plan['survivors']]} "
          f"to_build={plan['to_build_new_ranks']}")
    check("survivor is the (6,7) engine, adopted at new_rank=2",
          [(s["old_rank"], s["new_rank"]) for s in plan["survivors"]] == [(0, 2)])
    check("to_build == [0, 1]", plan["to_build_new_ranks"] == [0, 1])

    print("\n=== gate ===")
    check("expansion with flag OFF -> legacy full rebuild",
          run_patch(OLD, NEW, TP, False, old_worker_list(OLD))["path"] == "full_rebuild")
    SH_OLD, SH_NEW = {"size": [2, 6], "index": 1}, {"size": [6, 2], "index": 1}
    for flag in (False, True):
        rs = run_patch(SH_OLD, SH_NEW, TP, flag, old_worker_list(SH_OLD))
        check(f"pure shrink -> survivor path, mixed=False (allow_mixed={flag})",
              rs["path"] == "survivor" and rs["use_mixed"] is False)

    print("\n=== mixed ON: (a)/(b)/(c) ===")
    r = run_patch(OLD, NEW, TP, True, old_worker_list(OLD))
    check("survivor path taken with mixed=True", r["path"] == "survivor" and r["use_mixed"])
    check("(c) num_replicas from NEW topology == 3 (survivor count was 1)", r["num_new_replicas"] == 3)
    check("(b) delta built for ranks [0,1] only", [nr for nr, _ in r["built"]] == [0, 1])
    check("(b) delta lands on GPUs [2,3] and [4,5]", [b for _, b in r["built"]] == [[2, 3], [4, 5]])
    check("(b) survivor's GPUs [6,7] never rebuilt",
          all(6 not in b and 7 not in b for _, b in r["built"]))
    print(f"  merged group ({len(r['merged'])} workers):")
    for i, h in enumerate(r["merged"]):
        print(f"    [{i}] {h}")
    check("(a) merged length == new-topology cards (6)", len(r["merged"]) == 6)
    check("(a) survivor sits at offset new_rank*TP == 4", r["merged"][4:6] == r["survivor_handles"])
    check("(a) slice workers[rank*TP:(rank+1)*TP] recovers every replica",
          all(r["merged"][nr * TP:(nr + 1) * TP] == r["handles_by_new_rank"][nr]
              for nr in range(r["num_new_replicas"])))
    flat = [b for _, bs in r["built"] for b in bs] + [6, 7]
    check("no two replicas share a GPU", len(flat) == len(set(flat)))

    print("\n=== guards ===")
    # Non-divisible new topology: plan returns has_plan=False -> full rebuild
    # (this is the real _compute_rollout_survivor_plan behaviour, not a raise).
    nd = run_patch({"size": [4, 4], "index": 1}, {"size": [1, 7], "index": 1}, TP, True,
                   old_worker_list({"size": [4, 4], "index": 1}))
    check("non-divisible new topology -> has_plan=False -> full rebuild",
          nd["path"] == "full_rebuild" and nd["plan"]["has_plan"] is False)
    # A hole in the merged group must fail loudly, not silently drop a replica.
    try:
        run_patch(OLD, NEW, TP, True, old_worker_list(OLD), drop_rank=1)
        check("hole in merged group raises", False)
    except ValueError as e:
        check(f"hole in merged group raises ({str(e)[:46]}...)", "missing workers" in str(e))

    print("\n=== TP generality (nothing hardcoded to TP=2) ===")
    for tp, spec_old in ((1, {"size": [6, 2], "index": 1}),
                         (2, {"size": [6, 2], "index": 1}),
                         (3, {"size": [5, 3], "index": 1})):
        try:
            rt = run_patch(spec_old, NEW, tp, True, old_worker_list(spec_old))
            surv = {s["new_rank"] for s in rt["plan"]["survivors"]}
            built_gpus = {b for _, bs in rt["built"] for b in bs}
            surv_gpus = {b for s in rt["plan"]["survivors"] for b in s["bundles"]}
            check(f"TP={tp}: num_replicas={6 // tp}, merged={6}, survivor GPUs untouched",
                  rt["num_new_replicas"] == 6 // tp
                  and len(rt["merged"]) == 6
                  and not (built_gpus & surv_gpus)
                  and surv)
        except (ValueError, IndexError) as e:
            check(f"TP={tp} unexpectedly raised: {e}", False)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed -> {'ALL CHECKS PASSED' if passed == total else 'FAILURES REMAIN'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
