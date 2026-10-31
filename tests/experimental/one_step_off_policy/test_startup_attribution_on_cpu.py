"""CPU guards for A-line startup attribution. Source-level; imports no verl."""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_RAY_TRAINER = _ROOT / "verl" / "experimental" / "one_step_off_policy" / "ray_trainer.py"
_AGENT_LOOP = _ROOT / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
_TRAINER_CLASS = "OneStepOffRayTrainer"
_NESTED_CODE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_PER_GROUP_SPANS = (
    ("elastic_group_build_wg", ("self", "_build_role_group")),
    ("elastic_group_prepare", ("self", "_prepare_role_group_init")),
    ("elastic_group_commit", ("self", "_commit_role_group_init")),
)


def _method_src(method_name: str) -> str:
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OneStepOffRayTrainer":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return ast.unparse(item)
    raise AssertionError(f"OneStepOffRayTrainer has no method {method_name}")


# ---------------------------------------------------------------------------
# AST helpers. Substring checks pass even when a span wraps nothing (the real
# call moved below the `with` block), so every structural claim below is made
# against the parse tree instead of the unparsed text.
# ---------------------------------------------------------------------------


def _method_node(method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == _TRAINER_CLASS]
    assert len(classes) == 1, f"ray_trainer.py must define exactly one top-level {_TRAINER_CLASS} class"
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    assert len(methods) == 1, f"{_TRAINER_CLASS} must define exactly one direct {method_name} method"
    return methods[0]


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    attributes: list[str] = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return (node.id, *reversed(attributes))


def _walk_without_nested_code(node: ast.AST) -> Iterator[ast.AST]:
    if isinstance(node, _NESTED_CODE):
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_without_nested_code(child)


def _direct_executable_nodes(node: ast.AST) -> Iterator[ast.AST]:
    for statement in node.body:  # type: ignore[attr-defined]
        yield from _walk_without_nested_code(statement)


def _resize_trace_span_name(node: ast.AST) -> object:
    """The literal span-name constant of a `with resize_trace_span(...)` statement."""
    if not isinstance(node, (ast.With, ast.AsyncWith)):
        return None
    for item in node.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and _attribute_chain(call.func) == ("resize_trace_span",):
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                return call.args[1].value
            return "<dynamic>"
    return None


def _span_statement(scope: ast.AST, span_name: str) -> ast.With | ast.AsyncWith:
    """The unique `with resize_trace_span(..., "<span_name>", ...)` statement in scope."""
    spans = [part for part in ast.walk(scope) if _resize_trace_span_name(part) == span_name]
    assert len(spans) == 1, f"expected exactly one {span_name} span, found {len(spans)}"
    return spans[0]


def _span_body_nodes(span: ast.With | ast.AsyncWith) -> list[ast.AST]:
    """Everything the span actually TIMES: its body, never its own arguments.

    Deliberately excludes `span.items`, so a span whose only mention of the real
    call is inside its own metadata cannot masquerade as wrapping it.
    """
    nodes: list[ast.AST] = []
    for statement in span.body:
        nodes.extend(ast.walk(statement))
    return nodes


def _calls_to(scope: ast.AST, chain: tuple[str, ...]) -> list[ast.Call]:
    return [part for part in ast.walk(scope) if isinstance(part, ast.Call) and _attribute_chain(part.func) == chain]


def _assert_span_wraps_the_only_call(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    span_name: str,
    chain: tuple[str, ...],
) -> ast.With | ast.AsyncWith:
    """The span must contain the method's one and only call to `chain`.

    Identity (`is`) rather than a count is what rejects the mutation the audit
    defeated: keeping `with resize_trace_span(..., "<name>"): pass` and moving the
    real call below the block leaves every substring and count check green.
    """
    span = _span_statement(method, span_name)
    dotted = ".".join(chain)
    in_body = [
        part
        for part in _span_body_nodes(span)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == chain
    ]
    in_method = _calls_to(method, chain)
    assert len(in_body) == 1, (
        f"{span_name} must time exactly one {dotted}() call in its with-body, found {len(in_body)}; "
        "a span that names the call but does not enclose it measures nothing"
    )
    assert len(in_method) == 1, f"{method.name} must call {dotted}() exactly once, found {len(in_method)}"
    assert in_method[0] is in_body[0], (
        f"the {dotted}() call {method.name} really executes is not the one inside {span_name}"
    )
    return span


def _span_metadata_expr(span: ast.With | ast.AsyncWith) -> ast.expr:
    for item in span.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and _attribute_chain(call.func) == ("resize_trace_span",):
            for keyword in call.keywords:
                if keyword.arg == "metadata":
                    return keyword.value
    raise AssertionError(f"span {ast.unparse(span.items[0].context_expr)} passes no metadata= keyword")


def _resolve_dict(expr: ast.expr, scope: ast.AST) -> ast.Dict:
    """The dict literal an expression evaluates to: a literal, or `dict(<local>)`."""
    if isinstance(expr, ast.Dict):
        return expr
    if isinstance(expr, ast.Call) and _attribute_chain(expr.func) == ("dict",) and len(expr.args) == 1:
        inner = expr.args[0]
        if isinstance(inner, ast.Name):
            assignments = [
                statement.value
                for statement in ast.walk(scope)
                if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == inner.id for target in statement.targets)
            ]
            assert len(assignments) == 1, (
                f"local {inner.id} must be assigned exactly once so its metadata keys are unambiguous, "
                f"found {len(assignments)} assignments"
            )
            return _resolve_dict(assignments[0], scope)
        return _resolve_dict(inner, scope)
    raise AssertionError(f"cannot resolve metadata expression {ast.unparse(expr)} to a dict literal")


def _dict_items(node: ast.Dict) -> dict[str, ast.expr]:
    items: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str), (
            f"metadata keys must be string literals, got {ast.unparse(key) if key else '**'}"
        )
        items[key.value] = value
    return items


def _span_metadata(span: ast.With | ast.AsyncWith, scope: ast.AST) -> dict[str, ast.expr]:
    return _dict_items(_resolve_dict(_span_metadata_expr(span), scope))


# ---------------------------------------------------------------------------
# The baseline tests. These are the floor: they only prove the names exist.
# ---------------------------------------------------------------------------


def test_per_group_spans_are_split_into_build_prepare_commit():
    src = _method_src("_prepare_elastic_pool_workers")
    for event in ("elastic_group_build_wg", "elastic_group_prepare", "elastic_group_commit"):
        assert event in src, f"{event} missing; a single loop-wide span cannot show which group is slow"


def test_per_group_spans_carry_group_identity():
    src = _method_src("_prepare_elastic_pool_workers")
    assert "'group_index': group_index" in src.replace('"', "'")
    assert "'bundles'" in src.replace('"', "'"), "generic prepare/commit spans only carry role/phase"


def test_warmup_and_weight_sync_group_are_attributed():
    src = _method_src("_init_models")
    assert "warmup_communicator_cache_keys" in src, "this call site had zero spans"
    assert "create_weight_sync_group_full" in src, "inner comm_* spans cover only one branch each"


def test_finalize_keeps_exactly_one_event_loop():
    src = _method_src("_finalize_elastic_pool_replicas")
    assert src.count("asyncio.run(") == 1, "splitting into two loops changes runtime structure, not just tracing"
    assert "_launch_and_sleep" in src


def test_finalize_metadata_reports_pool_shape():
    src = _method_src("_finalize_elastic_pool_replicas")
    for key in ("awake_groups", "sleeping_groups", "num_replicas", "initial_k"):
        assert key in src, f"finalize span metadata must include {key}"


# ---------------------------------------------------------------------------
# Structural guards: each span must WRAP its real call, not merely name it.
# ---------------------------------------------------------------------------


def test_each_per_group_span_encloses_its_real_call():
    method = _method_node("_prepare_elastic_pool_workers")
    for span_name, chain in _PER_GROUP_SPANS:
        _assert_span_wraps_the_only_call(method, span_name, chain)


def test_per_group_spans_run_once_per_group_inside_the_spawn_loop():
    """Inside the loop = one timed span per group. Outside = one loop-wide total."""
    method = _method_node("_prepare_elastic_pool_workers")
    loops = [
        statement
        for statement in method.body
        if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name)
        and statement.target.id == "group_index"
    ]
    assert len(loops) == 1, f"expected one `for group_index in ...` loop, found {len(loops)}"
    spans = []
    for span_name, _chain in _PER_GROUP_SPANS:
        span = _span_statement(method, span_name)
        assert any(part is span for part in ast.walk(loops[0])), (
            f"{span_name} sits outside the per-group loop; it would then time every group at once and "
            "could not attribute the startup gap to a group"
        )
        spans.append(span)
    assert [span.lineno for span in spans] == sorted(span.lineno for span in spans), (
        "build -> prepare -> commit order must be preserved; Phase 1 reorders nothing"
    )


def test_per_group_span_metadata_binds_to_the_live_loop_values():
    method = _method_node("_prepare_elastic_pool_workers")
    for span_name, _chain in _PER_GROUP_SPANS:
        metadata = _span_metadata(_span_statement(method, span_name), method)
        assert set(metadata) == {"group_index", "bundles", "tp", "initially_awake"}, (
            f"{span_name} metadata keys are {sorted(metadata)}; a trace row without them cannot be joined "
            "back to the group whose build it timed"
        )
        group_index = metadata["group_index"]
        assert isinstance(group_index, ast.Name) and group_index.id == "group_index", (
            f"{span_name} must report the loop variable, got {ast.unparse(group_index)}"
        )
        assert any(isinstance(part, ast.Name) and part.id == "bundles" for part in ast.walk(metadata["bundles"])), (
            f"{span_name} must report the group's own bundles, got {ast.unparse(metadata['bundles'])}"
        )
        assert any(_attribute_chain(part) == ("layout", "tp") for part in ast.walk(metadata["tp"])), (
            f"{span_name} must report the real layout TP, got {ast.unparse(metadata['tp'])}"
        )
        awake = metadata["initially_awake"]
        assert isinstance(awake, ast.Constant) and awake.value is False, (
            "these spans only ever wrap genuinely-spawned groups (the initially-awake branch continues "
            f"before them), so initially_awake must be False, got {ast.unparse(awake)}"
        )


def test_warmup_and_weight_sync_spans_enclose_their_real_calls():
    method = _method_node("_init_models")
    warmup = _assert_span_wraps_the_only_call(
        method, "warmup_communicator_cache_keys", ("self", "_warmup_communicator_cache_keys")
    )
    sync = _assert_span_wraps_the_only_call(
        method, "create_weight_sync_group_full", ("self", "_create_weight_sync_group")
    )
    assert warmup.lineno < sync.lineno, "warmup runs before the weight sync group is created"


def test_weights_info_handoff_stays_between_the_two_new_spans():
    """Instrumentation only: the untimed statements between the calls must not move."""
    method = _method_node("_init_models")
    warmup = _span_statement(method, "warmup_communicator_cache_keys")
    sync = _span_statement(method, "create_weight_sync_group_full")
    timed = _span_body_nodes(warmup) + _span_body_nodes(sync)
    for chain in (("self", "_get_actor_weights_info"), ("self", "_set_actor_weights_info")):
        calls = _calls_to(method, chain)
        assert len(calls) == 1, f"{'.'.join(chain)} must still be called exactly once, found {len(calls)}"
        assert not any(part is calls[0] for part in timed), (
            f"{'.'.join(chain)} was pulled inside a new span; that inflates the measured cost with work "
            "the span does not claim to measure"
        )
        assert warmup.lineno < calls[0].lineno < sync.lineno, (
            f"{'.'.join(chain)} must stay in its original position between the two spans"
        )


def test_launch_and_sleep_spans_live_in_one_coroutine_and_one_event_loop():
    method = _method_node("_finalize_elastic_pool_replicas")
    coroutines = [
        part
        for part in ast.walk(method)
        if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef)) and part.name == "_launch_and_sleep"
    ]
    assert len(coroutines) == 1, f"expected one _launch_and_sleep coroutine, found {len(coroutines)}"
    runs = _calls_to(method, ("asyncio", "run"))
    assert len(runs) == 1, (
        f"finalize must keep exactly one asyncio.run, found {len(runs)}; a second event loop changes runtime "
        "structure rather than only adding tracing"
    )
    driven = runs[0].args[0]
    assert isinstance(driven, ast.Call) and _attribute_chain(driven.func) == ("_launch_and_sleep",), (
        f"the single event loop must drive _launch_and_sleep, got {ast.unparse(driven)}"
    )
    for span_name in ("elastic_pool_launch_engines", "elastic_pool_initial_sleep"):
        span = _span_statement(method, span_name)
        assert any(part is span for part in ast.walk(coroutines[0])), (
            f"{span_name} must sit inside _launch_and_sleep; hoisting it out of the coroutine would time the "
            "coroutine's construction instead of its awaited work"
        )


def test_launch_span_times_init_hybrid_and_sleep_span_times_sleep():
    method = _method_node("_finalize_elastic_pool_replicas")
    launch = _span_statement(method, "elastic_pool_launch_engines")
    sleep = _span_statement(method, "elastic_pool_initial_sleep")
    for span, span_name, chain in (
        (launch, "elastic_pool_launch_engines", ("replica", "init_hybrid")),
        (sleep, "elastic_pool_initial_sleep", ("replica", "sleep")),
    ):
        body = _span_body_nodes(span)
        gathers = [
            part
            for part in body
            if isinstance(part, ast.Await)
            and isinstance(part.value, ast.Call)
            and _attribute_chain(part.value.func) == ("asyncio", "gather")
        ]
        assert len(gathers) == 1, f"{span_name} must await exactly one asyncio.gather, found {len(gathers)}"
        in_body = [part for part in body if isinstance(part, ast.Call) and _attribute_chain(part.func) == chain]
        in_method = _calls_to(method, chain)
        dotted = ".".join(chain)
        assert len(in_body) == 1 and len(in_method) == 1 and in_body[0] is in_method[0], (
            f"the real {dotted}() call must be the one {span_name} encloses; found {len(in_body)} in the span "
            f"and {len(in_method)} in the method"
        )
        assert any(part is in_body[0] for part in ast.walk(gathers[0])), (
            f"{dotted}() must be gathered inside {span_name}, not awaited beside it"
        )
    assert launch.lineno < sleep.lineno, "engines are launched before they are put to sleep"


def test_finalize_span_encloses_the_prebuild_accounting():
    method = _method_node("_finalize_elastic_pool_replicas")
    span = _span_statement(method, "finalize_elastic_pool_replicas")
    body = _span_body_nodes(span)
    assert any(isinstance(part, ast.Call) and _attribute_chain(part.func) == ("asyncio", "run") for part in body), (
        "the finalize span must cover the replica launch it exists to attribute"
    )
    prebuild_writes = [
        part
        for part in body
        if isinstance(part, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "prebuild_s" for target in part.targets)
    ]
    assert len(prebuild_writes) == 1, "the finalize span must close after the prebuild accounting it reports"
    assert any(isinstance(part, ast.Call) and _attribute_chain(part.func) == ("logger", "info") for part in body), (
        "the finalize span must reach through the accounting log, per the intended finalize boundary"
    )


def test_finalize_span_closes_before_the_memory_probe():
    """Regression guard: a B-line probe RPC inside an A-line span inflates its duration."""
    method = _method_node("_finalize_elastic_pool_replicas")
    span = _span_statement(method, "finalize_elastic_pool_replicas")
    body = _span_body_nodes(span)
    nested_samples = [
        part
        for part in body
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == ("self", "_sample_pool_memory")
    ]
    assert not nested_samples, (
        f"{len(nested_samples)} _sample_pool_memory call(s) sit inside the finalize span; the probe's RPC cost "
        "would then be reported as startup finalize time"
    )
    assert not [part for part in body if _resize_trace_span_name(part) == "memory_probe_startup"], (
        "the memory_probe_startup span must stay outside the A-line finalize span"
    )


def test_finalize_span_metadata_reports_the_live_pool_shape():
    """A substring-only version would pass on the probe metadata alone."""
    method = _method_node("_finalize_elastic_pool_replicas")
    span = _span_statement(method, "finalize_elastic_pool_replicas")
    metadata = _span_metadata(span, method)
    assert set(metadata) == {"initial_k", "awake_groups", "sleeping_groups", "num_replicas"}, (
        f"finalize span metadata keys are {sorted(metadata)}; without the pool shape the measured duration "
        "cannot be normalised per replica"
    )
    expected_sources = {
        "initial_k": ("self", "_sleep_pool_initial_k"),
        "awake_groups": ("self", "_elastic_pool", "awake_indices"),
        "sleeping_groups": ("self", "_elastic_pool", "sleeping_indices"),
        "num_replicas": ("self", "_sleep_pool_layout", "num_elastic_groups"),
    }
    for key, chain in expected_sources.items():
        assert any(_attribute_chain(part) == chain for part in ast.walk(metadata[key])), (
            f"{key} must be read from the live {'.'.join(chain)}, got {ast.unparse(metadata[key])}; the span "
            "opens before the local `pool`/`layout` aliases are bound"
        )


# ---------------------------------------------------------------------------
# The async rollout manager build is the other half of the startup gap.
# ---------------------------------------------------------------------------


def _class_method_node(
    path: pathlib.Path, class_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    assert len(classes) == 1, f"{path.name} must define exactly one top-level {class_name} class"
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    assert len(methods) == 1, f"{class_name} must define exactly one direct {method_name} method"
    return methods[0]


def _span_keyword(span: ast.With | ast.AsyncWith, name: str) -> ast.expr:
    for item in span.items:
        call = item.context_expr
        if isinstance(call, ast.Call) and _attribute_chain(call.func) == ("resize_trace_span",):
            for keyword in call.keywords:
                if keyword.arg == name:
                    return keyword.value
    raise AssertionError(f"span passes no {name}= keyword")


def test_init_workers_attributes_the_async_rollout_manager_build():
    """The startup gap is comm_full_build.end -> sync_rollout_weights.start.

    That window is exactly _init_async_rollout_manager() + _finalize_elastic_pool_
    replicas(). Instrumentation covered the second; without a span on the first, the
    cold build of the initially-awake vLLM engines lands in an unlabelled residual
    that only cross-lane inference can recover.
    """
    method = _method_node("init_workers")
    span = _assert_span_wraps_the_only_call(
        method, "init_async_rollout_manager", ("self", "_init_async_rollout_manager")
    )
    lane = _span_keyword(span, "lane")
    assert isinstance(lane, ast.Constant) and lane.value == "trainer_main", (
        f"the span must be on the trainer lane to bound the gap, got {ast.unparse(lane)}"
    )


def test_init_async_rollout_manager_span_measures_only_that_call():
    """A span that also swallows its neighbours cannot attribute the gap between them."""
    method = _method_node("init_workers")
    span = _span_statement(method, "init_async_rollout_manager")
    body = _span_body_nodes(span)
    for chain in (("self", "_init_models"), ("self", "_finalize_elastic_pool_replicas")):
        calls = _calls_to(method, chain)
        assert len(calls) == 1, f"init_workers must call {'.'.join(chain)}() exactly once"
        assert not any(part is calls[0] for part in body), (
            f"{'.'.join(chain)}() was pulled inside init_async_rollout_manager; its cost would then be "
            "reported as async rollout manager build time"
        )
    models = _calls_to(method, ("self", "_init_models"))[0]
    finalize = _calls_to(method, ("self", "_finalize_elastic_pool_replicas"))[0]
    assert models.lineno < span.lineno < finalize.lineno, (
        "instrumentation only: the startup call order _init_models -> _init_async_rollout_manager -> "
        "_finalize_elastic_pool_replicas must be preserved"
    )


_AGENT_LOOP_INIT_SPANS = (
    ("init_llm_servers", ("self", "_initialize_llm_servers")),
    ("init_agent_workers", ("self", "_init_agent_loop_workers")),
    ("init_sleep", ("self", "sleep")),
)


def test_agent_loop_manager_sync_init_mirrors_the_async_init_spans():
    """__init__ is the startup path; _async_init is the resize path.

    Only the resize twin was instrumented, so the startup cold-build of the
    initially-awake engines -- plausibly the dominant term in the gap -- had no
    breakdown at all.
    """
    method = _class_method_node(_AGENT_LOOP, "AgentLoopManager", "__init__")
    for span_name, chain in _AGENT_LOOP_INIT_SPANS:
        span = _assert_span_wraps_the_only_call(method, span_name, chain)
        lane = _span_keyword(span, "lane")
        assert isinstance(lane, ast.Constant) and lane.value == "agent_loop_manager", (
            f"{span_name} must report the same lane as its _async_init twin, got {ast.unparse(lane)}"
        )


def test_agent_loop_manager_sync_init_builds_its_trace_config_before_the_first_span():
    """resize_trace_span(self._resize_trace_config, ...) raises AttributeError unless
    __init__ binds the attribute first -- __init__ never needed it before."""
    method = _class_method_node(_AGENT_LOOP, "AgentLoopManager", "__init__")
    writes = [
        part
        for part in ast.walk(method)
        if isinstance(part, ast.Assign)
        and any(_attribute_chain(target) == ("self", "_resize_trace_config") for target in part.targets)
    ]
    assert len(writes) == 1, f"__init__ must bind self._resize_trace_config exactly once, found {len(writes)}"
    first_span = min(_span_statement(method, name).lineno for name, _chain in _AGENT_LOOP_INIT_SPANS)
    assert writes[0].lineno < first_span, "the trace config must be bound before the first span reads it"


def test_agent_loop_manager_sync_init_keeps_the_sleep_guarded_by_free_cache_engine():
    """Instrumentation only: the span must not turn a conditional sleep into an
    unconditional one, which would sleep engines the run expects to be awake."""
    method = _class_method_node(_AGENT_LOOP, "AgentLoopManager", "__init__")
    span = _span_statement(method, "init_sleep")
    guards = [
        part
        for part in ast.walk(method)
        if isinstance(part, ast.If)
        and any(
            _attribute_chain(sub) == ("self", "config", "actor_rollout_ref", "rollout", "free_cache_engine")
            for sub in ast.walk(part.test)
        )
    ]
    assert len(guards) == 1, f"__init__ must keep exactly one free_cache_engine guard, found {len(guards)}"
    assert any(part is span for part in ast.walk(guards[0])), (
        "init_sleep must stay inside the free_cache_engine branch; hoisting it out would make the sleep "
        "unconditional"
    )

