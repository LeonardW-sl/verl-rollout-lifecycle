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
"""Skipping the FP32 FSDP shell on pure-rollout processes.

Two layers, because only one of them can run on bare CPU:

- ``should_skip_...`` is a pure predicate in shell_skip.py, imported by file
  path so the decision is tested as behaviour (no torch/ray/omegaconf needed).
- the call site inside ``_commit_local_init_model`` is checked via AST, the same
  technique as test_split_sleep_wake_override_on_cpu.py, because importing
  fsdp_workers.py needs the whole heavy stack.

Why the shell is skippable at all: the shell has no active consumer on a pure
rollout process, and vLLM takes its tokenizer from HFModelConfig, not from the
worker.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import pathlib
import sys
import textwrap

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SHELL_SKIP_PATH = _REPO / "verl" / "experimental" / "one_step_off_policy" / "shell_skip.py"
_ONE_STEP_OFF = _REPO / "verl" / "experimental" / "one_step_off_policy" / "fsdp_workers.py"
_DYNAMIC_RESIZE_YAML = (
    _REPO / "verl" / "experimental" / "one_step_off_policy" / "config" / "one_step_off_ppo_trainer_dynamic_resize.yaml"
)


def _load_shell_skip():
    spec = importlib.util.spec_from_file_location("shell_skip_under_test", _SHELL_SKIP_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: under `from __future__ import annotations`,
    # @dataclass resolves string annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ss():
    return _load_shell_skip()


# --- the predicate: who skips the shell ----------------------------------------


def test_pure_rollout_skips_when_flag_on(ss):
    assert ss.should_skip_rollout_fsdp_shell(is_actor=False, is_rollout=True, flag_enabled=True) is True


def test_pure_rollout_keeps_shell_when_flag_off(ss):
    """Default must reproduce the old behaviour -- this is the comparison baseline."""
    assert ss.should_skip_rollout_fsdp_shell(is_actor=False, is_rollout=True, flag_enabled=False) is False


def test_actor_never_skips_even_with_flag_on(ss):
    """The actor trains from this shell. Skipping it would be silent corruption."""
    assert ss.should_skip_rollout_fsdp_shell(is_actor=True, is_rollout=True, flag_enabled=True) is False
    assert ss.should_skip_rollout_fsdp_shell(is_actor=True, is_rollout=False, flag_enabled=True) is False


def test_ref_only_role_unaffected(ss):
    """ref builds its own shell further down; this predicate must not claim it."""
    assert ss.should_skip_rollout_fsdp_shell(is_actor=False, is_rollout=False, flag_enabled=True) is False


# --- config plumbing -----------------------------------------------------------


def test_flag_defaults_to_false(ss):
    """OFF by default: turning it on by default would be a config change."""
    assert ss.resolve_skip_rollout_fsdp_shell_flag(None) is False
    assert ss.resolve_skip_rollout_fsdp_shell_flag({}) is False
    assert ss.resolve_skip_rollout_fsdp_shell_flag({"dynamic_resize": {}}) is False


def test_flag_reads_worker_scoped_key(ss):
    """The worker sees actor_rollout_ref.dynamic_resize.*, not trainer.*."""
    cfg = {"dynamic_resize": {"skip_rollout_fsdp_shell": True}}
    assert ss.resolve_skip_rollout_fsdp_shell_flag(cfg) is True


def test_flag_accepts_string_true_from_hydra(ss):
    """Hydra/OmegaConf interpolation can hand through the string form."""
    assert ss.resolve_skip_rollout_fsdp_shell_flag({"dynamic_resize": {"skip_rollout_fsdp_shell": "true"}}) is True
    assert ss.resolve_skip_rollout_fsdp_shell_flag({"dynamic_resize": {"skip_rollout_fsdp_shell": "false"}}) is False


def test_flag_reads_attribute_style_config_without_the_key(ss):
    """A config object exposing dynamic_resize but not the key must fall back to False.

    OmegaConf nodes are attribute-style and a missing key is the normal shape before
    the yaml default lands, so this must not raise.
    """

    class _Node:
        pass

    cfg = _Node()
    cfg.dynamic_resize = _Node()
    assert ss.resolve_skip_rollout_fsdp_shell_flag(cfg) is False

    cfg.dynamic_resize.skip_rollout_fsdp_shell = True
    assert ss.resolve_skip_rollout_fsdp_shell_flag(cfg) is True


def test_yaml_bridges_flag_from_trainer_to_worker():
    """Without this bridge the worker never sees the flag and ON silently no-ops."""
    text = _DYNAMIC_RESIZE_YAML.read_text(encoding="utf-8")
    assert "skip_rollout_fsdp_shell: ${trainer.dynamic_resize.skip_rollout_fsdp_shell}" in text, (
        "actor_rollout_ref.dynamic_resize must interpolate the trainer-side flag"
    )
    assert "skip_rollout_fsdp_shell: false" in text, "trainer-side default must be false"


# --- the call site -------------------------------------------------------------


def _commit_method() -> ast.FunctionDef:
    tree = ast.parse(_ONE_STEP_OFF.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LocalInitActorRolloutRefWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_commit_local_init_model":
                    return item
    raise RuntimeError("LocalInitActorRolloutRefWorker._commit_local_init_model not found (file moved?)")


def _init_self_assignments() -> set[str]:
    """Attribute names assigned on self in LocalInitActorRolloutRefWorker.__init__."""
    tree = ast.parse(_ONE_STEP_OFF.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LocalInitActorRolloutRefWorker":
            init = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
            return {
                t.attr
                for n in ast.walk(init)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Attribute) and getattr(t.value, "id", None) == "self"
            }
    raise RuntimeError("LocalInitActorRolloutRefWorker not found")


def test_build_model_optimizer_call_is_guarded_by_the_predicate():
    """The shell build must sit behind the skip decision, on its own ancestor chain.

    Walking the enclosing ``If`` nodes rather than every ``If`` in the method:
    a whole-function substring search passes even when the guard is attached to
    an unrelated branch.
    """
    fn = _commit_method()
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    calls = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_build_model_optimizer"
    ]
    assert calls, "_commit_local_init_model no longer calls _build_model_optimizer"

    def _guarded_by_skip(call: ast.Call) -> bool:
        node = call
        while node in parents:
            parent = parents[node]
            if (
                isinstance(parent, ast.If)
                and node in parent.body
                and "_skip_rollout_fsdp_shell" in ast.dump(parent.test)
            ):
                return True
            node = parent
        return False

    guarded = [c for c in calls if _guarded_by_skip(c)]
    assert guarded, "no _build_model_optimizer call sits inside an if whose test reads _skip_rollout_fsdp_shell"


def test_actor_module_fsdp_stays_none_placeholder():
    """Skipping leaves the attribute at its __init__ placeholder, so it must exist."""
    assert "actor_module_fsdp" in _init_self_assignments()


def test_side_effect_attributes_are_rebound_when_shell_is_skipped():
    """tokenizer/processor/generation_config are side effects of the skipped build.

    generation_config has no __init__ placeholder (unlike the other two), so a read
    after a skip would be AttributeError rather than the None its readers handle
    (verl/workers/fsdp_workers.py:967). The skip branch must rebind all three.
    """
    fn = _commit_method()
    src = ast.get_source_segment(_ONE_STEP_OFF.read_text(encoding="utf-8"), fn) or ""
    skip_branch = [
        n
        for n in ast.walk(ast.parse(textwrap.dedent(src)))
        if isinstance(n, ast.If)
        and ast.dump(n.test) == ast.dump(ast.parse("self._skip_rollout_fsdp_shell").body[0].value)
    ]
    assert skip_branch, "no `if self._skip_rollout_fsdp_shell:` rebind branch in _commit_local_init_model"

    rebound = {
        t.attr
        for node in skip_branch
        for n in ast.walk(node)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and getattr(t.value, "id", None) == "self"
    }
    assert {"tokenizer", "processor", "generation_config"} <= rebound, (
        f"skip branch rebinds only {sorted(rebound)}; generation_config in particular has no __init__ placeholder"
    )


def test_standalone_rollout_ckpt_manager_is_skipped_with_the_shell():
    """With the shell gone the manager holds model=None, i.e. a load-only empty shell.

    The rollout group never restore-loads anyway (ray_trainer.py ignores
    resume_from_path for Role.Rollout), so constructing it would only hide a future
    real call behind `assert self.model is not None`.
    """
    fn = _commit_method()
    src = ast.get_source_segment(_ONE_STEP_OFF.read_text(encoding="utf-8"), fn) or ""
    guards = [
        ast.unparse(n.test)
        for n in ast.walk(ast.parse(textwrap.dedent(src)))
        if isinstance(n, ast.If)
        and any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "FSDPCheckpointManager" for c in ast.walk(n))
    ]
    rollout_guards = [g for g in guards if "not self._is_actor" in g]
    assert rollout_guards, "standalone-rollout ckpt manager block not found"
    assert all("_skip_rollout_fsdp_shell" in g for g in rollout_guards), (
        f"rollout ckpt manager is still built when the shell is skipped: {rollout_guards}"
    )


def test_commit_method_has_no_undefined_local_names():
    """Every name read in _commit_local_init_model must be bound somewhere in it.

    Adding a guard around a block while leaving a reader of one of its locals
    downstream is exactly how this change first broke: the ckpt-manager block moved
    behind the flag but still read a variable defined in the removed branch, which
    would have been a NameError on the *default* (flag OFF) path.
    """
    fn = _commit_method()
    src = textwrap.dedent(ast.get_source_segment(_ONE_STEP_OFF.read_text(encoding="utf-8"), fn) or "")
    tree = ast.parse(src)

    bound = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)

    read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    module_tree = ast.parse(_ONE_STEP_OFF.read_text(encoding="utf-8"))
    module_globals = {n.id for n in ast.walk(module_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    module_globals |= {
        (alias.asname or alias.name).split(".")[0]
        for n in ast.walk(module_tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for alias in n.names
    }
    module_globals |= {
        n.name for n in ast.walk(module_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    unresolved = read - bound - module_globals - set(dir(builtins))
    assert not unresolved, f"names read but never bound in _commit_local_init_model: {sorted(unresolved)}"
