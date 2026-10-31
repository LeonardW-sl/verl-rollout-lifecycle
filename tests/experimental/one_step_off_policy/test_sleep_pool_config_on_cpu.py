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
"""Sleep-pool config plumbing, AST-bound to real ray_trainer source."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

import pytest

_PKG = pathlib.Path(__file__).resolve().parents[3] / "verl" / "experimental" / "one_step_off_policy"
_RAY_TRAINER = _PKG / "ray_trainer.py"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cfg_test", _PKG / f"{name}.py")
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


def _trainer_method_src(name: str) -> str:
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OneStepOffRayTrainer":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return ast.unparse(item)
    raise RuntimeError(f"OneStepOffRayTrainer has no method {name}")


def test_master_switch_defaults_off():
    """Off must leave the legacy path byte-for-byte, like worker_reuse before it."""
    src = _RAY_TRAINER.read_text(encoding="utf-8")
    assert '_build_dynamic_resize_flag("sleep_pool.enable", default=False)' in src.replace("'", '"')


def test_zone_sizes_are_read_as_ints():
    src = _trainer_method_src("_build_sleep_pool_layout")
    assert "sleep_pool.permanent_rollout_gpus" in src
    assert "sleep_pool.elastic_gpus" in src
    assert "shared_pool.n_gpus_per_node" in src


def test_layout_is_none_when_disabled():
    src = _trainer_method_src("_build_sleep_pool_layout")
    assert "return None" in src


def test_int_helper_exists_and_is_not_a_bool_cast():
    """_build_dynamic_resize_flag casts to bool; R=2 would become True."""
    src = _trainer_method_src("_build_dynamic_resize_int")
    assert "int(" in src
    assert "bool(" not in src


def test_experiment_geometry_round_trips(at):
    """N=8 R=2 E=4 TP=2 -> A=2, m=2, and the two split plans map to k=0 and k=2."""
    layout = at.ZoneLayout(total_gpus=8, permanent_rollout_gpus=2, elastic_gpus=4, tp=2)
    assert (layout.actor_gpus, layout.num_elastic_groups) == (2, 2)
    assert layout.schedule_from_rollout_cards(2) == 0
    assert layout.schedule_from_rollout_cards(6) == 2


def test_enabled_pool_with_zero_elastic_zone_is_rejected(at):
    """E=0 means nothing can ever sleep -- almost certainly a misconfigured run,
    and silently doing nothing would read as a passing sleep-pool experiment."""
    src = _trainer_method_src("_build_sleep_pool_layout")
    assert "elastic_gpus" in src and "raise" in src


def test_each_config_value_reaches_its_own_zone_parameter():
    """Bind config key -> ZoneLayout keyword, so swapping R and E fails here.

    The greps above only prove both key names appear somewhere in the method.
    They pass unchanged if the two values are handed to ZoneLayout in each
    other's slots, which is not cosmetic: under the experiment topology
    (N=8 R=2 E=4 TP=2) a swap gives m=1 instead of 2, makes the [6,2] plan
    raise "rollout_cards=2 is below the permanent rollout zone R=4", and turns
    the [2,6] plan's k from 2 into 1 -- i.e. one elastic group silently never
    sleeps. Match the assignment structurally instead.
    """
    tree = ast.parse(_trainer_method_src("_build_sleep_pool_layout"))

    # local name -> the dynamic_resize key it was read from
    read_from: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        call = node.value
        if not isinstance(target, ast.Name) or not isinstance(call, ast.Call):
            continue
        if not ast.unparse(call.func).endswith("_build_dynamic_resize_int"):
            continue
        if call.args and isinstance(call.args[0], ast.Constant):
            read_from[target.id] = call.args[0].value

    zone_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("ZoneLayout")
    )
    passed_as = {
        kw.arg: ast.unparse(kw.value) for kw in zone_call.keywords if kw.arg is not None
    }

    expected = {
        "total_gpus": "shared_pool.n_gpus_per_node",
        "permanent_rollout_gpus": "sleep_pool.permanent_rollout_gpus",
        "elastic_gpus": "sleep_pool.elastic_gpus",
    }
    for param, config_key in expected.items():
        local = passed_as.get(param)
        assert local is not None, f"ZoneLayout(...) is missing the {param}= argument"
        assert read_from.get(local) == config_key, (
            f"ZoneLayout({param}=) receives {local!r}, which was read from "
            f"{read_from.get(local)!r}, expected {config_key!r}"
        )

    # tp is not read via _build_dynamic_resize_int, so resolve its local
    # assignment (tp = int(self.config...tensor_model_parallel_size)) instead.
    tp_arg = passed_as.get("tp")
    assert tp_arg is not None, "ZoneLayout(...) is missing the tp= argument"
    tp_sources = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == tp_arg
    ]
    assert tp_sources, f"ZoneLayout(tp={tp_arg}) has no visible assignment in this method"
    assert any("tensor_model_parallel_size" in src for src in tp_sources), (
        f"ZoneLayout(tp=) must derive from rollout.tensor_model_parallel_size, got {tp_sources!r}"
    )
