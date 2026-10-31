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
"""vLLM worker local_rank must be derived from RANK when several cards are visible.

With a group-wide CUDA_VISIBLE_DEVICES="0,1" and the old hardcoded local_rank=0,
BOTH workers of a TP=2 group bind the same physical card. Deriving local_rank
from RANK on the same spawn separates them.

That collision is silent -- two engines quietly share one card and the other sits
idle -- so it needs a regression guard. vllm_rollout.py cannot be imported on CPU
(it pulls in vllm/torch), hence the AST approach used elsewhere in this package.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_VLLM_ROLLOUT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "verl"
    / "workers"
    / "rollout"
    / "vllm_rollout"
    / "vllm_rollout.py"
)


def _method_src(class_name: str, method_name: str) -> str:
    tree = ast.parse(_VLLM_ROLLOUT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return ast.unparse(item)
    raise RuntimeError(f"{class_name}.{method_name} not found in {_VLLM_ROLLOUT}")


def _code_only(src: str) -> str:
    """Body without the docstring or comments.

    ast.unparse already drops comments, but the docstring survives and this
    method's docstring names the very tokens the assertions forbid.
    """
    tree = ast.parse(src)
    fn = tree.body[0]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        fn.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


@pytest.fixture(scope="module")
def init_worker_src() -> str:
    for class_name in ("vLLMAsyncRollout", "vLLMRollout"):
        try:
            return _code_only(_method_src(class_name, "_init_worker"))
        except RuntimeError:
            continue
    raise RuntimeError("no _init_worker found in vllm_rollout.py (class renamed?)")


def test_local_rank_is_not_hardcoded_zero(init_worker_src):
    """The old rule was `0 if not ray_noset_visible_devices() else ...`.

    Under the sleep pool's zero-quota spawn Ray sets no per-worker CVD, so that
    branch made every worker in a TP group bind cuda:0.
    """
    assert "local_rank" in init_worker_src
    assert "device_count" in init_worker_src, (
        "local_rank must consider how many cards are actually visible"
    )


def test_local_rank_derives_from_rank(init_worker_src):
    """RANK is the only per-worker value available: _create_worker sets it
    (ray/base.py:541-549) but never sets LOCAL_RANK, and worker_env is one dict
    per group."""
    assert "%" in init_worker_src, "expected a modulo of RANK against the visible count"
    assert "worker_rank" in init_worker_src or 'os.environ["RANK"]' in init_worker_src


def test_single_card_workers_still_bind_zero(init_worker_src):
    """The standard path must stay byte-for-byte: with one visible card the only
    valid index is 0, so the new rule has to be gated on visible > 1."""
    assert "> 1" in init_worker_src, "the RANK-derived branch must be gated on visible > 1"


def test_ray_noset_branch_is_preserved(init_worker_src):
    """When Ray does assign accelerator ids, they remain authoritative."""
    assert "ray_noset_visible_devices" in init_worker_src
    assert "get_accelerator_ids" in init_worker_src


def test_rank_is_still_passed_through(init_worker_src):
    """vLLM needs the global rank regardless of the local binding."""
    assert "'rank'" in init_worker_src or '"rank"' in init_worker_src
