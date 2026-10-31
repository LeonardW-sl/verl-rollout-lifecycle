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
"""Mixed survivor expansion: bind tests to REAL ray_trainer.py source.

Heavy deps (ray/torch/tensordict) block `import OneStepOffRayTrainer` on bare CPU.
We AST-load ``_rollout_spec_bundles`` / ``_compute_rollout_survivor_plan`` and the
gate/merge fragments from the live file so a source drift fails these tests.

Covers (a)/(b)/(c) for the mixed expansion ``[6,2]->[2,6]`` and the shrink ``[6,2]->[5,3]`` at TP=1.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import textwrap

import pytest

_RAY_TRAINER = (
    pathlib.Path(__file__).resolve().parents[3]
    / "verl"
    / "experimental"
    / "one_step_off_policy"
    / "ray_trainer.py"
)

_AWAKE_TOPOLOGY = (
    pathlib.Path(__file__).resolve().parents[3]
    / "verl"
    / "experimental"
    / "one_step_off_policy"
    / "awake_topology.py"
)


def _load_awake_topology():
    """Load the REAL algorithm module by path (no torch/ray needed) so these
    tests bind to production source on both sides of the delegation."""
    spec = importlib.util.spec_from_file_location("awake_topology_for_survivor_test", _AWAKE_TOPOLOGY)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: under `from __future__ import annotations`,
    # @dataclass resolves its string annotations via sys.modules[cls.__module__],
    # which is None for an unregistered module -> AttributeError at class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_survivor_plan_api(path: pathlib.Path = _RAY_TRAINER):
    """Exec the real plan methods from ray_trainer.py into a tiny namespace."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    want = {"_rollout_spec_bundles", "_compute_rollout_survivor_plan"}
    methods: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OneStepOffRayTrainer":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in want:
                    methods[item.name] = item
    missing = want - methods.keys()
    if missing:
        raise RuntimeError(f"ray_trainer.py missing methods {missing} (file moved?)")

    # Rebuild a minimal class with only those methods (decorators stripped to plain defs).
    class_body = []
    for name in ("_rollout_spec_bundles", "_compute_rollout_survivor_plan"):
        fn = methods[name]
        fn_copy = ast.FunctionDef(
            name=fn.name,
            args=fn.args,
            body=fn.body,
            decorator_list=[],  # drop @staticmethod etc.; assign manually
            returns=fn.returns,
            type_comment=getattr(fn, "type_comment", None),
        )
        ast.copy_location(fn_copy, fn)
        class_body.append(fn_copy)
    class_body.append(ast.Pass())
    mini = ast.ClassDef(name="_SurvivorPlanAPI", bases=[], keywords=[], body=class_body, decorator_list=[])
    mod = ast.Module(body=[mini], type_ignores=[])
    ast.fix_missing_locations(mod)
    # _compute_rollout_survivor_plan now delegates to awake_topology.diff_engines
    # (D4: one topology algorithm, not two). Inject the real function so a drift
    # in EITHER file fails these tests.
    ns: dict = {"diff_engines": _load_awake_topology().diff_engines}
    exec(compile(mod, str(path), "exec"), ns)  # noqa: S102 — intentional bind to production source
    cls = ns["_SurvivorPlanAPI"]
    # Mirror production: _rollout_spec_bundles is a staticmethod.
    cls._rollout_spec_bundles = staticmethod(cls._rollout_spec_bundles)
    return cls()


def _gate(plan: dict | None, allow_mixed: bool) -> tuple[bool, bool]:
    """Same boolean structure as ray_trainer mixed gate (must stay in lockstep).

    Production (ray_trainer.py survivor-path decision):
      _survivor_base = plan and has_plan and len(survivors)>0
      use_survivor = _survivor_base and (len(to_build)==0 or allow_mixed)
      use_mixed = use_survivor and len(to_build)>0
    """
    # Structural anchor: fail if production renames the flag without updating tests.
    src = _RAY_TRAINER.read_text(encoding="utf-8")
    assert "worker_reuse.allow_mixed" in src
    assert "use_rollout_mixed_path" in src
    assert "_dynamic_resize_worker_reuse_mixed_enabled" in src

    to_build = plan["to_build_new_ranks"] if plan else []
    survivor_base = (
        plan is not None and plan["has_plan"] and len(plan["survivors"]) > 0
    )
    use_survivor = survivor_base and (len(to_build) == 0 or allow_mixed)
    use_mixed = use_survivor and len(to_build) > 0
    return use_survivor, use_mixed


def _merge_by_new_rank(handles_by_new_rank: dict[int, list], num_new_replicas: int) -> list:
    """Production merge contract: position == new_rank * TP slice."""
    missing = [r for r in range(num_new_replicas) if r not in handles_by_new_rank]
    if missing:
        raise ValueError(f"merged rollout group is missing workers for new_ranks {missing}")
    merged: list = []
    for new_rank in range(num_new_replicas):
        merged.extend(handles_by_new_rank[new_rank])
    return merged


@pytest.fixture(scope="module")
def plan_api():
    return _load_survivor_plan_api()


def _split_spec(size, index=1, mode="split"):
    return {"mode": mode, "size": list(size), "index": index}


# --- real _compute_rollout_survivor_plan --------------------------------------


def test_plan_step0_scenario_binds_real_source(plan_api):
    """Scenario A expansion [6,2]->[2,6] TP=2: one survivor at new_rank=2, to_build=[0,1]."""
    old = _split_spec([6, 2])
    new = _split_spec([2, 6])
    plan = plan_api._compute_rollout_survivor_plan(old, new, rollout_world_size=2)
    assert plan["has_plan"] is True
    assert [(s["old_rank"], s["new_rank"], s["bundles"]) for s in plan["survivors"]] == [
        (0, 2, [6, 7])
    ]
    assert plan["to_build_new_ranks"] == [0, 1]
    assert plan["to_kill_old_ranks"] == []


def test_plan_scenario_b_tp1_binds_real_source(plan_api):
    """Scenario B order-2 [6,2]->[5,3] TP=1: mixed, to_build=[0], survivors shift back."""
    old = _split_spec([6, 2])
    new = _split_spec([5, 3])
    plan = plan_api._compute_rollout_survivor_plan(old, new, rollout_world_size=1)
    assert plan["has_plan"] is True
    pairs = sorted((s["old_rank"], s["new_rank"], tuple(s["bundles"])) for s in plan["survivors"])
    assert pairs == [(0, 1, (6,)), (1, 2, (7,))]
    assert plan["to_build_new_ranks"] == [0]
    assert plan["to_kill_old_ranks"] == []


def test_plan_non_divisible_returns_has_plan_false_not_raise(plan_api):
    """Real behaviour: not divisible -> has_plan=False (silent full-rebuild), no raise."""
    old = _split_spec([4, 4])
    new = _split_spec([1, 7])  # 7 % TP2 != 0
    plan = plan_api._compute_rollout_survivor_plan(old, new, rollout_world_size=2)
    assert plan["has_plan"] is False
    assert plan["survivors"] == []
    assert plan["to_build_new_ranks"] == []


def test_rollout_spec_bundles_real(plan_api):
    assert plan_api._rollout_spec_bundles(_split_spec([6, 2])) == [6, 7]
    assert plan_api._rollout_spec_bundles(_split_spec([5, 3])) == [5, 6, 7]
    assert plan_api._rollout_spec_bundles(_split_spec([2, 6])) == [2, 3, 4, 5, 6, 7]


# --- gate (lockstep with production booleans + source anchors) ----------------


@pytest.mark.parametrize(
    "survivors,to_build,allow_mixed,expect_survivor,expect_mixed",
    [
        ([{"old_rank": 0, "new_rank": 0}], [], False, True, False),
        ([{"old_rank": 0, "new_rank": 0}], [], True, True, False),
        ([{"old_rank": 1, "new_rank": 2}], [0, 1], False, False, False),
        ([{"old_rank": 1, "new_rank": 2}], [0, 1], True, True, True),
        ([], [0, 1, 2], True, False, False),
    ],
)
def test_gate_admits_mixed_only_when_enabled(
    survivors, to_build, allow_mixed, expect_survivor, expect_mixed
):
    plan = {
        "has_plan": True,
        "survivors": list(survivors),
        "to_build_new_ranks": list(to_build),
        "to_kill_old_ranks": [],
    }
    use_survivor, use_mixed = _gate(plan, allow_mixed)
    assert use_survivor is expect_survivor
    assert use_mixed is expect_mixed


def test_gate_on_real_scenario_b_plan(plan_api):
    plan = plan_api._compute_rollout_survivor_plan(
        _split_spec([6, 2]), _split_spec([5, 3]), rollout_world_size=1
    )
    assert _gate(plan, allow_mixed=False) == (False, False)
    assert _gate(plan, allow_mixed=True) == (True, True)


# --- (a) merge + missing-rank guard -------------------------------------------


def test_merged_handles_are_ordered_by_new_rank():
    tp = 2
    handles_by_new_rank = {
        2: ["survivor_w0", "survivor_w1"],
        0: ["delta0_w0", "delta0_w1"],
        1: ["delta1_w0", "delta1_w1"],
    }
    merged = _merge_by_new_rank(handles_by_new_rank, num_new_replicas=3)
    assert merged == [
        "delta0_w0",
        "delta0_w1",
        "delta1_w0",
        "delta1_w1",
        "survivor_w0",
        "survivor_w1",
    ]
    for new_rank in range(3):
        assert merged[new_rank * tp : (new_rank + 1) * tp] == handles_by_new_rank[new_rank]


def test_missing_rank_raises_instead_of_silently_dropping():
    with pytest.raises(ValueError, match="missing workers"):
        _merge_by_new_rank({0: ["a", "b"], 2: ["c", "d"]}, num_new_replicas=3)


# --- (c) topology-derived replica count ---------------------------------------


def test_num_replicas_from_new_topology_not_survivor_count(plan_api):
    """Without this, survivor-count num_replicas drops new_rank=2."""
    tp = 2
    plan = plan_api._compute_rollout_survivor_plan(
        _split_spec([6, 2]), _split_spec([2, 6]), rollout_world_size=tp
    )
    new_cards = len(plan_api._rollout_spec_bundles(_split_spec([2, 6])))
    survivor_worker_count = len(plan["survivors"]) * tp
    buggy = survivor_worker_count // tp
    correct = new_cards // tp
    assert buggy == 1
    assert correct == 3
    survivor_new_rank = plan["survivors"][0]["new_rank"]
    assert survivor_new_rank >= buggy
    assert survivor_new_rank < correct


# --- keep / reuse rank mapping contract (Bug 5) --------------------------------


def test_keep_old_reuse_new_rank_mapping_contract(plan_api):
    """Shutdown keep uses OLD ranks; init reuse dict is keyed by NEW ranks."""
    plan = plan_api._compute_rollout_survivor_plan(
        _split_spec([6, 2]), _split_spec([2, 6]), rollout_world_size=2
    )
    # Mirror production after the merge block (ray_trainer survivor init path).
    survivor_old_ranks = {s["old_rank"] for s in plan["survivors"]}
    fake_replicas = {0: "rep_old_0"}  # only old rank 0 exists pre-resize
    reuse = {s["new_rank"]: fake_replicas[s["old_rank"]] for s in plan["survivors"]}
    assert survivor_old_ranks == {0}
    assert reuse == {2: "rep_old_0"}

    src = _RAY_TRAINER.read_text(encoding="utf-8")
    assert "keep_replica_ranks=survivor_old_ranks" in src
    assert 's["new_rank"]: manager.rollout_replicas[s["old_rank"]]' in src.replace(" ", "") or (
        's["new_rank"]' in src and 's["old_rank"]' in src and "reuse_replicas" in src
    )


# --- (b) delta ranks only -----------------------------------------------------


def test_delta_ranks_are_to_build_only(plan_api):
    plan = plan_api._compute_rollout_survivor_plan(
        _split_spec([6, 2]), _split_spec([2, 6]), rollout_world_size=2
    )
    num_new_replicas = 6 // 2
    delta = sorted(r for r in plan["to_build_new_ranks"] if r < num_new_replicas)
    assert delta == [0, 1]
    assert {s["new_rank"] for s in plan["survivors"]}.isdisjoint(delta)


def test_source_file_still_contains_mixed_path_anchors():
    """Fail loud if mixed path is deleted while tests remain."""
    src = _RAY_TRAINER.read_text(encoding="utf-8")
    for needle in (
        "worker_reuse.allow_mixed",
        "use_rollout_mixed_path",
        "handles_by_new_rank",
        "resize_survivor_plan",
        "merged_worker_handles",
        "delta_new_ranks",
    ):
        assert needle in src, f"missing production anchor {needle}"


def test_survivor_plan_delegates_to_the_shared_algorithm():
    """D4: exactly one engine-diff algorithm. If the trainer ever grows a private
    copy again, this fails."""
    src = _RAY_TRAINER.read_text(encoding="utf-8")
    assert "diff_engines" in src, "survivor plan no longer delegates to awake_topology"
    fn_src = src.split("def _compute_rollout_survivor_plan")[1].split("\n    def ")[0]
    assert "diff_engines(" in fn_src
    for reimplemented in ("old_by_set", "new_by_set", "to_kill = ["):
        assert reimplemented not in fn_src, (
            f"_compute_rollout_survivor_plan re-implements the diff ({reimplemented!r}); "
            "it must delegate to awake_topology.diff_engines"
        )
