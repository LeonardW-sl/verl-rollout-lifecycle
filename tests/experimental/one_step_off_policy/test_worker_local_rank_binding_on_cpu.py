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
"""Elastic-zone workers must bind one card each under a group-wide CVD.

Third instance of one collision class, at the layer the first two fixes miss:

  vLLM _init_worker already binds its engine device (vllm_rollout.py).
  first attempt set LOCAL_RANK in DetachAsyncRolloutWorker.__init__ -- INERT,
                because WorkerDict.__init__ runs Worker.__init__ before
                constructing this class (ray/base.py:919-929), and the only
                set_device sits behind `if is_ray_noset_visible_devices`
                (base/worker.py:282-290), false under the zero-quota spawn

So the env var alone changes nothing; set_device is what binds. Without it both
TP workers of an elastic group take cuda:0 of a shared CUDA_VISIBLE_DEVICES and
FSDP model init dies with "Duplicate GPU detected".

fsdp_workers.py cannot be imported on CPU (pulls torch/ray/vllm), so this reads
the real source with AST -- same technique as the sibling tests in this package.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_FSDP_WORKERS = (
    pathlib.Path(__file__).resolve().parents[3] / "verl" / "experimental" / "one_step_off_policy" / "fsdp_workers.py"
)

_HELPER = "_scope_cuda_device_for_group_wide_cvd"


def _tree() -> ast.Module:
    return ast.parse(_FSDP_WORKERS.read_text(encoding="utf-8"))


def _func_src(name: str) -> str:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise RuntimeError(f"function {name} not found in {_FSDP_WORKERS} (renamed?)")


def _init_src() -> str:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ClassDef) and node.name == "DetachAsyncRolloutWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return ast.unparse(item)
            raise RuntimeError("DetachAsyncRolloutWorker has no __init__ (moved?)")
    raise RuntimeError("class DetachAsyncRolloutWorker not found (file moved?)")


@pytest.fixture(scope="module")
def helper_src() -> str:
    return _func_src(_HELPER)


@pytest.fixture(scope="module")
def init_src() -> str:
    return _init_src()


def test_calls_set_device_not_just_env(helper_src: str):
    """The env var is inert here -- an actual device binding is required.

    This is the assertion the first attempt would have failed.
    """
    assert "get_torch_device().set_device(local_rank)" in helper_src


def test_derives_index_from_rank(helper_src: str):
    """RANK is the only per-worker value available under a shared worker_env."""
    assert "int(os.environ['RANK']) % len(visible)" in helper_src


def test_counts_visible_from_cuda_visible_devices(helper_src: str):
    assert "os.environ.get('CUDA_VISIBLE_DEVICES', '')" in helper_src


def test_single_card_paths_untouched(helper_src: str):
    """Every pre-existing worker sees one card and must keep the default binding."""
    assert "len(visible) <= 1" in helper_src
    assert "return" in helper_src


def test_invoked_before_delegated_constructor(init_src: str):
    """Must run before the model is built, i.e. before a CUDA context exists."""
    assert _HELPER in init_src
    assert init_src.index(_HELPER) < init_src.index("LocalInitActorRolloutRefWorker.__init__")
