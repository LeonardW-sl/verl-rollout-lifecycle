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
"""Decide whether a worker process should skip building the FP32 FSDP shell.

A pure-rollout process builds a full FSDP model shell it never trains
(fsdp_workers.py:997 `if self._is_actor or self._is_rollout`). The shell is FP32
because the rollout branch passes a bare `FSDPEngineConfig()`, whose
`model_dtype` defaults to "fp32" (workers/config/engine.py:171), so the
`torch.float32 if self._is_actor else torch.bfloat16` fallback never runs. Size
is 4*P/W, and the prebuilt elastic group shards over W=TP=2
(ray_trainer.py:1489), so the shell can occupy a large share of every card an
actor is training on. That is what makes the large-model case not fit.

The shell has no active consumer on that process: vLLM sources its tokenizer
from HFModelConfig rather than the worker (vllm_rollout.py:256), and the two
shell dereferences in `_build_rollout` are behind `fsdp_version(...) == 1`,
which is 0 for None. Kept behind a flag anyway so enabling stays reversible and
the default reproduces the old behaviour exactly.

Lives in its own module (no torch/ray/omegaconf import) so the decision is
testable on bare CPU, same reason as resize_budget.py.
"""

from __future__ import annotations

from typing import Any

#: Config key, relative to a `dynamic_resize` node.
SKIP_ROLLOUT_FSDP_SHELL_KEY = "skip_rollout_fsdp_shell"


def _as_bool(value: Any) -> bool:
    """Interpret a config value that may arrive as a bool or a string.

    Hydra overrides on the command line hand through strings, so `=false` must
    not read as truthy.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_skip_rollout_fsdp_shell_flag(config: Any) -> bool:
    """Read `dynamic_resize.skip_rollout_fsdp_shell` off a worker-side config.

    The worker is handed `actor_rollout_ref`, so the flag is read at
    `dynamic_resize.*` here and bridged from `trainer.dynamic_resize.*` by
    one_step_off_ppo_trainer_dynamic_resize.yaml. Defaults to False: turning it
    on by default would change shipped behaviour.

    Accepts plain dicts as well as OmegaConf nodes so it stays importable
    without omegaconf.
    """
    if config is None:
        return False
    dynamic_resize = config.get("dynamic_resize", None) if hasattr(config, "get") else None
    if dynamic_resize is None:
        dynamic_resize = getattr(config, "dynamic_resize", None)
    if dynamic_resize is None:
        return False
    if hasattr(dynamic_resize, "get"):
        value = dynamic_resize.get(SKIP_ROLLOUT_FSDP_SHELL_KEY, False)
    else:
        value = getattr(dynamic_resize, SKIP_ROLLOUT_FSDP_SHELL_KEY, False)
    return _as_bool(value)


def should_skip_rollout_fsdp_shell(*, is_actor: bool, is_rollout: bool, flag_enabled: bool) -> bool:
    """True only for a pure-rollout process with the flag on.

    `is_actor` wins unconditionally: an actor -- including the colocated
    actor_rollout role -- trains from this shell, so skipping it there would be
    silent corruption rather than a saving.
    """
    if not flag_enabled:
        return False
    if is_actor:
        return False
    return bool(is_rollout)
