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
"""Split-arch sleep/wake override: bind assertions to REAL source via AST.

Heavy deps (ray/torch/vllm) block importing the worker classes on bare CPU, so
we parse the production files instead. Same technique as
test_mixed_survivor_expansion_on_cpu.py.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_ONE_STEP_OFF = (
    pathlib.Path(__file__).resolve().parents[3]
    / "verl"
    / "experimental"
    / "one_step_off_policy"
    / "fsdp_workers.py"
)
_BASE_WORKERS = (
    pathlib.Path(__file__).resolve().parents[3] / "verl" / "workers" / "fsdp_workers.py"
)


def _class_node(path: pathlib.Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise RuntimeError(f"class {class_name} not found in {path} (file moved?)")


def _method(cls_node: ast.ClassDef, name: str):
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    raise RuntimeError(f"{cls_node.name} has no method {name}")


def _has_direct_rollout_register(fn) -> bool:
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        if getattr(dec.func, "id", None) != "register":
            continue
        for kw in dec.keywords:
            if kw.arg == "dispatch_mode" and ast.unparse(kw.value) == "Dispatch.DIRECT_ROLLOUT_METHOD":
                return True
    return False


def _code_only(fn) -> str:
    """Method body WITHOUT its docstring.

    ast.unparse() keeps docstrings, so a plain substring match cannot tell code
    that *touches* actor_module_fsdp from prose that *names* it while explaining
    why we must not. That is not hypothetical: the override's own docstring
    mentions trainer_mode() and actor_module_fsdp.train(), so the naive version
    of these assertions failed the correct implementation AND passed a
    kv_cache-only wake_up -- the exact bug they exist to catch.
    """
    clone = ast.parse(ast.unparse(fn)).body[0]
    body = clone.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        clone.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(clone)
    return ast.unparse(clone)


def _called_funcs(fn) -> set:
    """Names of everything actually invoked, e.g. {"self.rollout.release"}."""
    return {ast.unparse(node.func) for node in ast.walk(fn) if isinstance(node, ast.Call)}


def _resume_tags(fn):
    """The literal tags= list handed to .resume(), or None if absent/non-literal."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(".resume"):
            for kw in node.keywords:
                if kw.arg == "tags" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    return [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
    return None


@pytest.fixture(scope="module")
def detach_async_rollout_worker() -> ast.ClassDef:
    return _class_node(_ONE_STEP_OFF, "DetachAsyncRolloutWorker")


@pytest.mark.parametrize("method_name", ["sleep", "wake_up"])
def test_override_exists_and_is_async(detach_async_rollout_worker, method_name):
    fn = _method(detach_async_rollout_worker, method_name)
    assert isinstance(fn, ast.AsyncFunctionDef), f"{method_name} must be async (callers await it)"


@pytest.mark.parametrize("method_name", ["sleep", "wake_up"])
def test_override_keeps_direct_rollout_dispatch(detach_async_rollout_worker, method_name):
    """base.py:864 binds the bare name ONLY for DIRECT_ROLLOUT_METHOD; else it becomes rollout_sleep."""
    fn = _method(detach_async_rollout_worker, method_name)
    assert _has_direct_rollout_register(fn), (
        f"{method_name} lost @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)"
    )


@pytest.mark.parametrize("method_name", ["sleep", "wake_up"])
def test_override_never_touches_actor_module(detach_async_rollout_worker, method_name):
    """The whole point: a pure rollout process has no actor module to touch."""
    code = _code_only(_method(detach_async_rollout_worker, method_name))
    for forbidden in ("actor_module_fsdp", "trainer_mode", "rollout_mode", "load_fsdp_model_to_gpu"):
        assert forbidden not in code, f"{method_name} must not reference {forbidden}"


@pytest.mark.parametrize("method_name", ["sleep", "wake_up"])
def test_override_is_gated_on_free_cache_engine(detach_async_rollout_worker, method_name):
    code = _code_only(_method(detach_async_rollout_worker, method_name))
    assert "free_cache_engine" in code


@pytest.mark.parametrize("method_name", ["sleep", "wake_up"])
def test_override_guards_missing_rollout_and_engine(detach_async_rollout_worker, method_name):
    code = _code_only(_method(detach_async_rollout_worker, method_name))
    assert "getattr(self, 'rollout', None)" in code
    assert "getattr(rollout, 'inference_engine', None)" in code


def test_sleep_releases_the_engine(detach_async_rollout_worker):
    calls = _called_funcs(_method(detach_async_rollout_worker, "sleep"))
    assert any(name.endswith(".release") for name in calls), f"sleep calls nothing named *.release: {calls}"


def test_wake_up_maps_weights_not_just_kv_cache(detach_async_rollout_worker):
    """Skipping the "weights" tag leaves weight memory unmapped, so G1's broadcast
    has nowhere to write. Under level=2 the map is
    allocation-only -- there is no CPU copy to restore.

    Reads the tags= list structurally. A substring match on the unparsed body
    passes a kv_cache-only resume, because the docstring above contains the word
    "weights" -- i.e. the naive assertion greenlights the precise bug it names.
    """
    fn = _method(detach_async_rollout_worker, "wake_up")
    assert any(name.endswith(".resume") for name in _called_funcs(fn)), "wake_up calls nothing named *.resume"
    tags = _resume_tags(fn)
    assert tags is not None, "wake_up's resume() needs a literal tags=[...] list this test can inspect"
    for required in ("weights", "kv_cache"):
        assert required in tags, f"resume tags missing {required!r} (got {tags})"


@pytest.mark.parametrize("method_name", ["sleep", "wake_up"])
def test_override_acknowledges_with_true(detach_async_rollout_worker, method_name):
    """The return value is the acknowledgement ElasticEnginePool gathers on.

    The pool only commits a group's awake/asleep state after every worker's call
    resolves, so this ack is the difference between "the engine confirmed" and
    "the engine was asked". The base class returns True; match it.
    """
    fn = _method(detach_async_rollout_worker, method_name)
    returns = [
        ast.unparse(node.value)
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert returns == ["True"], f"{method_name} should return True, found {returns}"


def test_base_class_blocking_point_still_exists():
    """Justification anchor. If upstream ever guards actor_module_fsdp in
    trainer_mode(), this override's rationale changes and a human should re-read
    it rather than let it silently rot."""
    trainer_mode = _method(_class_node(_BASE_WORKERS, "ActorRolloutRefWorker"), "trainer_mode")
    body = ast.unparse(trainer_mode)
    assert "self.actor_module_fsdp.train()" in body, (
        "upstream trainer_mode() changed -- re-verify why DetachAsyncRolloutWorker overrides sleep()"
    )
