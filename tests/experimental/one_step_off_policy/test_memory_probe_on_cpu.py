"""CPU guards for the Phase 1 memory probe. Parses source via ast; imports no verl."""

from __future__ import annotations

import ast
import pathlib
import tempfile
from collections.abc import Iterator

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_FSDP_WORKERS = _ROOT / "verl" / "experimental" / "one_step_off_policy" / "fsdp_workers.py"
_TARGET_CLASS = "DetachAsyncRolloutWorker"
_CUDA_MEMORY_CALLS = {
    ("torch", "cuda", "mem_get_info"),
    ("torch", "cuda", "memory_allocated"),
    ("torch", "cuda", "memory_reserved"),
}
_NESTED_CODE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _find_method_in_tree(
    tree: ast.Module, source_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == _TARGET_CLASS]
    assert len(classes) == 1, f"{source_name} must define exactly one top-level {_TARGET_CLASS} class"
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    assert len(methods) == 1, f"{_TARGET_CLASS} must define exactly one direct {method_name} method"
    return methods[0]


def _find_method(path: pathlib.Path, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return _find_method_in_tree(tree, path.name, method_name)


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


def _direct_executable_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    for statement in node.body:
        yield from _walk_without_nested_code(statement)


def _single_direct_assignment(node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> ast.expr:
    assignments = [
        statement
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets)
    ]
    writes = [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Name) and part.id == name and isinstance(part.ctx, (ast.Store, ast.Del))
    ]
    assert len(assignments) == 1 and len(writes) == 1, (
        f"{node.name} must assign local {name} exactly once in its direct executable body"
    )
    return assignments[0].value


def _returned_dict(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Dict:
    returns = [statement for statement in node.body if isinstance(statement, ast.Return)]
    assert len(returns) == 1, f"expected one direct return, found {len(returns)}"
    value = returns[0].value
    if isinstance(value, ast.Name):
        assignment = _single_direct_assignment(node, value.id)
        assert assignment.lineno < returns[0].lineno, (
            f"returned local {value.id} must be assigned before it is returned"
        )
        value = assignment
    assert isinstance(value, ast.Dict), "get_memory_snapshot must directly return its uniquely assigned snapshot dict"
    return value


def _dict_value(node: ast.Dict, key: str) -> ast.expr:
    values = [
        value
        for key_node, value in zip(node.keys, node.values, strict=True)
        if isinstance(key_node, ast.Constant) and key_node.value == key
    ]
    assert len(values) == 1, f"returned snapshot dict must have exactly one {key!r} key"
    return values[0]


def _is_getattr_rollout_sleep_level(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _attribute_chain(node.func) == ("getattr",)
        and len(node.args) == 3
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "rollout"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "sleep_level"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value is None
    )


def _same_expression(node: ast.AST, source: str) -> bool:
    expected = ast.parse(source, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)


def _reads_self_rollout(node: ast.AST) -> bool:
    return any(
        _same_expression(node, source)
        for source in (
            "self.rollout",
            "getattr(self, 'rollout', None)",
            "self.rollout if hasattr(self, 'rollout') else None",
        )
    )


def _assert_cuda_memory_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    calls: dict[tuple[str, ...], list[ast.Call]] = {name: [] for name in _CUDA_MEMORY_CALLS}
    for part in _direct_executable_nodes(node):
        if isinstance(part, ast.Call) and (name := _attribute_chain(part.func)) in calls:
            calls[name].append(part)

    for name, matches in calls.items():
        qualified_name = ".".join(name)
        assert len(matches) == 1, f"expected exactly one direct {qualified_name} call, found {len(matches)}"
        call = matches[0]
        assert len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "device", (
            f"{qualified_name} must receive exactly the local device Name"
        )
        assert not call.keywords, f"{qualified_name} must be called exactly with (device)"


def _assert_runtime_sleep_level(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    actual_sleep_level = _dict_value(_returned_dict(node), "actual_sleep_level")
    if _attribute_chain(actual_sleep_level) == ("self", "rollout", "sleep_level"):
        return

    assert _is_getattr_rollout_sleep_level(actual_sleep_level), (
        "actual_sleep_level must be self.rollout.sleep_level or getattr(rollout, 'sleep_level', None)"
    )
    rollout_source = _single_direct_assignment(node, "rollout")
    assert _reads_self_rollout(rollout_source), "the unique local rollout assignment must originate from self.rollout"


def test_snapshot_uses_direct_rollout_dispatch_so_name_has_no_prefix():
    node = _find_method(_FSDP_WORKERS, "get_memory_snapshot")
    dispatch_values = [
        keyword.value
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and _attribute_chain(decorator.func) == ("register",)
        for keyword in decorator.keywords
        if keyword.arg == "dispatch_mode"
    ]
    assert any(_attribute_chain(value) == ("Dispatch", "DIRECT_ROLLOUT_METHOD") for value in dispatch_values), (
        "must have @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD) so the trainer can call the bare name"
    )


def test_snapshot_reports_all_three_memory_views():
    _assert_cuda_memory_calls(_find_method(_FSDP_WORKERS, "get_memory_snapshot"))


def test_cuda_memory_calls_in_uncalled_nested_function_are_rejected():
    tree = ast.parse(_FSDP_WORKERS.read_text(encoding="utf-8"))
    node = _find_method_in_tree(tree, _FSDP_WORKERS.name, "get_memory_snapshot")
    cuda_statement_indexes = [
        index
        for index, statement in enumerate(node.body)
        if any(
            isinstance(part, ast.Call) and _attribute_chain(part.func) in _CUDA_MEMORY_CALLS
            for part in _walk_without_nested_code(statement)
        )
    ]
    assert cuda_statement_indexes, "mutation setup found no direct CUDA sampling statements"

    nested_function = ast.parse("def _uncalled_cuda_sampling():\n    pass\n").body[0]
    assert isinstance(nested_function, ast.FunctionDef)
    nested_function.body = [node.body[index] for index in cuda_statement_indexes]
    first_index = cuda_statement_indexes[0]
    node.body = [
        statement for index, statement in enumerate(node.body) if index not in cuda_statement_indexes
    ]
    node.body.insert(first_index, nested_function)
    ast.fix_missing_locations(tree)

    with tempfile.TemporaryDirectory() as directory:
        mutated_path = pathlib.Path(directory) / _FSDP_WORKERS.name
        mutated_path.write_text(ast.unparse(tree), encoding="utf-8")
        mutated_method = _find_method(mutated_path, "get_memory_snapshot")

    nested_calls = [
        part
        for part in ast.walk(mutated_method)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) in _CUDA_MEMORY_CALLS
    ]
    assert len(nested_calls) == len(_CUDA_MEMORY_CALLS), "mutation must retain all CUDA calls in nested code"
    try:
        _assert_cuda_memory_calls(mutated_method)
    except AssertionError as exc:
        assert "expected exactly one direct" in str(exc)
    else:
        raise AssertionError("CUDA AST guard accepted calls that exist only in an uncalled nested function")


def test_snapshot_reads_runtime_sleep_level_not_a_constant():
    _assert_runtime_sleep_level(_find_method(_FSDP_WORKERS, "get_memory_snapshot"))


_RAY_TRAINER = _ROOT / "verl" / "experimental" / "one_step_off_policy" / "ray_trainer.py"
_TRAINER_CLASS = "OneStepOffRayTrainer"
_SAMPLE_RESULT_KEYS = {
    "samples",
    "expected_samples",
    "observed_samples",
    "error_count",
    "sampling_rpc_ms",
}
_FINGERPRINT_EXPRESSIONS = {
    "model_path": "self.config.actor_rollout_ref.model.path",
    "gpu_memory_utilization": "rollout_cfg.gpu_memory_utilization",
    "tensor_model_parallel_size": "rollout_cfg.tensor_model_parallel_size",
    "max_num_seqs": "rollout_cfg.max_num_seqs",
    "enforce_eager": "rollout_cfg.enforce_eager",
    "free_cache_engine": "rollout_cfg.free_cache_engine",
}


def _find_trainer_method_in_tree(
    tree: ast.Module, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == _TRAINER_CLASS]
    assert len(classes) == 1, f"ray_trainer.py must define exactly one top-level {_TRAINER_CLASS} class"
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    assert len(methods) == 1, f"{_TRAINER_CLASS} must define exactly one direct {method_name} method"
    return methods[0]


def _find_trainer_method(method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return _find_trainer_method_in_tree(ast.parse(_RAY_TRAINER.read_text(encoding="utf-8")), method_name)


def _target_names(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Name) for item in node.elts):
        return tuple(item.id for item in node.elts)
    return None


def _dict_items(node: ast.Dict) -> dict[str, ast.expr]:
    items: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str), "dict keys must be string literals"
        assert key.value not in items, f"dict must not repeat key {key.value!r}"
        items[key.value] = value
    return items


def _sampler_dispatch_loops(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[ast.For, ast.For]:
    outer_loops = [
        statement
        for statement in node.body
        if isinstance(statement, ast.For)
        and _target_names(statement.target) == ("group_index",)
        and _same_expression(statement.iter, "group_indices")
    ]
    assert len(outer_loops) == 1, "sampler must directly iterate every requested group_index"
    outer = outer_loops[0]

    # handles() dispatch is guarded (a raise here must not abort a live
    # resize), so the real handles list is bound to a local by a try/except
    # rather than enumerated directly off pool.handles(group_index).
    handles_assignments = [
        part
        for part in _walk_without_nested_code(outer)
        if isinstance(part, ast.Assign)
        and any(_target_names(target) == ("handles",) for target in part.targets)
        and isinstance(part.value, ast.Call)
        and _attribute_chain(part.value.func) == ("list",)
        and len(part.value.args) == 1
        and _same_expression(part.value.args[0], "pool.handles(group_index)")
    ]
    assert len(handles_assignments) == 1, "sampler must bind pool.handles(group_index) to a local handles list"

    inner_loops = [
        statement
        for statement in _walk_without_nested_code(outer)
        if isinstance(statement, ast.For)
        and _target_names(statement.target) == ("tp_rank", "handle")
        and _same_expression(statement.iter, "enumerate(handles)")
    ]
    assert len(inner_loops) == 1, "sampler must enumerate(handles) once handles() has been dispatched"
    assert handles_assignments[0].lineno < inner_loops[0].lineno, (
        "handles must be resolved (and guarded) before the inner loop enumerates it"
    )
    return outer, inner_loops[0]


def _is_bare_snapshot_remote(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _attribute_chain(node.func) == ("handle", "get_memory_snapshot", "remote")
        and not node.args
        and not node.keywords
    )


def _assert_sampler_ast_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    private_accesses = [
        chain
        for part in _direct_executable_nodes(node)
        if (chain := _attribute_chain(part)) is not None
        and any(component in {"_replicas", "_worker_group"} for component in chain)
    ]
    assert not private_accesses, f"sampler must use public handles(), found private access {private_accesses}"

    outer, inner = _sampler_dispatch_loops(node)
    assert not any(isinstance(part, ast.Break) for part in _walk_without_nested_code(outer)), (
        "sampler dispatch loops must not stop after the first group or TP rank"
    )

    ref_appends = [
        part
        for part in _walk_without_nested_code(inner)
        if isinstance(part, ast.Call)
        and _attribute_chain(part.func) == ("refs", "append")
        and len(part.args) == 1
        and not part.keywords
        and _is_bare_snapshot_remote(part.args[0])
    ]
    assert len(ref_appends) == 1, "each handle must append one bare get_memory_snapshot.remote() ref"

    ray_gets = [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == ("ray", "get")
    ]
    assert len(ray_gets) == 1, f"sampler must perform exactly one ray.get, found {len(ray_gets)}"
    ray_get = ray_gets[0]
    assert not any(part is ray_get for part in _walk_without_nested_code(outer)), (
        "the single ray.get must run after all dispatch loops, not serially per handle"
    )
    assert len(ray_get.args) == 1 and isinstance(ray_get.args[0], ast.Name) and ray_get.args[0].id == "refs", (
        "ray.get must await the complete refs list"
    )
    assert len(ray_get.keywords) == 1 and ray_get.keywords[0].arg == "timeout", (
        "ray.get must have exactly one bounded timeout keyword"
    )
    assert isinstance(ray_get.keywords[0].value, ast.Name) and ray_get.keywords[0].value.id == "timeout_s"
    assert outer.end_lineno is not None and outer.end_lineno < ray_get.lineno, (
        "all handle refs must be submitted before the bounded wait"
    )

    returns = [part for part in _direct_executable_nodes(node) if isinstance(part, ast.Return)]
    assert returns, "sampler must return completeness metadata"
    for returned in returns:
        assert isinstance(returned.value, ast.Dict), "every sampler exit must directly return its result dict"
        assert set(_dict_items(returned.value)) == _SAMPLE_RESULT_KEYS, (
            "every sampler result must return samples and all four completeness fields"
        )


def _assert_coordinate_payload_pairing_is_strict(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """The coordinate/payload pairing must be `zip(..., strict=True)`.

    `coordinates` and `payloads` are produced by two different loops, so a length
    divergence is invisible without strict=True: every record after the divergence
    point silently inherits another rank's (group_index, tp_rank) and group 3's
    memory gets reported as group 1's. strict=True is what makes that loud.
    """
    zips = [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == ("zip",)
    ]
    assert len(zips) == 1, f"sampler must pair coordinates with payloads in exactly one zip, found {len(zips)}"
    pairing = zips[0]
    assert [_attribute_chain(argument) for argument in pairing.args] == [("coordinates",), ("payloads",)], (
        f"the pairing must zip the coordinate list with the payload list, got {ast.unparse(pairing)}"
    )
    assert any(
        isinstance(statement, ast.For) and statement.iter is pairing
        for statement in _direct_executable_nodes(node)
    ), "the pairing must drive the loop that attributes each payload to its own rank"
    strict = {keyword.arg: keyword.value for keyword in pairing.keywords}.get("strict")
    assert isinstance(strict, ast.Constant) and strict.value is True, (
        f"{ast.unparse(pairing)} must pass strict=True: the two lists are built by separate loops, so an "
        "unequal length would silently mis-attribute every sample after the divergence point"
    )


def _assert_fingerprint_ast_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    rollout_cfg = _single_direct_assignment(node, "rollout_cfg")
    assert _same_expression(rollout_cfg, "self.config.actor_rollout_ref.rollout"), (
        "rollout_cfg must come from the real flat actor_rollout_ref.rollout config"
    )
    returned = _returned_dict(node)
    items = _dict_items(returned)
    assert set(items) == set(_FINGERPRINT_EXPRESSIONS), "fingerprint must return exactly the six required keys"
    for key, expression in _FINGERPRINT_EXPRESSIONS.items():
        assert _same_expression(items[key], expression), f"{key} must read exact config expression {expression}"


def _assert_completeness_ast_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    result_get_keys = {
        part.args[0].value
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Call)
        and _attribute_chain(part.func) == ("result", "get")
        and part.args
        and isinstance(part.args[0], ast.Constant)
        and isinstance(part.args[0].value, str)
    }
    assert {"samples", "expected_samples", "observed_samples", "error_count"} <= result_get_keys, (
        "completeness must read samples plus expected/observed/error metadata"
    )
    assert any(
        isinstance(part, ast.Call)
        and _attribute_chain(part.func) == ("len",)
        and len(part.args) == 1
        and isinstance(part.args[0], ast.Name)
        and part.args[0].id == "samples"
        for part in _direct_executable_nodes(node)
    ), "completeness must compare len(samples) with observed metadata"
    assert any(
        isinstance(part, ast.Call) and _attribute_chain(part.func) == ("cls", "_memory_sample_invalid")
        for part in _direct_executable_nodes(node)
    ), "completeness must recompute invalid records instead of trusting error_count"
    assert any(
        isinstance(part, ast.Compare)
        and any(isinstance(operator, ast.Eq) for operator in part.ops)
        and any(
            isinstance(candidate, ast.Call)
            and _attribute_chain(candidate.func) == ("sample", "get")
            and candidate.args
            and isinstance(candidate.args[0], ast.Constant)
            and candidate.args[0].value == "actual_sleep_level"
            for candidate in (part.left, *part.comparators)
        )
        for part in _direct_executable_nodes(node)
    ), "completeness must compare every rank's runtime actual_sleep_level"


def _compile_trainer_class(*method_names: str, ray_object=None, extra_namespace: dict | None = None):
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    source_class = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == _TRAINER_CLASS]
    assert len(source_class) == 1
    selected = [
        statement
        for statement in source_class[0].body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in method_names
    ]
    assert {method.name for method in selected} == set(method_names)
    synthetic_class = ast.ClassDef(
        name=_TRAINER_CLASS,
        bases=[],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[synthetic_class], type_ignores=[]))
    namespace = {"ray": ray_object, "time": type("Clock", (), {"monotonic": staticmethod(lambda: 1.0)})}
    namespace.update(extra_namespace or {})
    exec(compile(module, str(_RAY_TRAINER), "exec"), namespace)
    return namespace[_TRAINER_CLASS]


def _valid_memory_payload(*, sleep_level: int = 1, **overrides) -> dict:
    payload = {
        "worker_rank": 0,
        "actual_sleep_level": sleep_level,
        "memory_allocated_bytes": 10,
        "memory_reserved_bytes": 20,
        "memory_free_bytes": 80,
        "memory_total_bytes": 100,
        "sampling_error": None,
    }
    payload.update(overrides)
    return payload


class _FakeRemoteMethod:
    def __init__(self, payload, issued: list) -> None:
        self.payload = payload
        self.issued = issued

    def remote(self):
        self.issued.append(self.payload)
        return self.payload


class _FakeHandle:
    def __init__(self, payload, issued: list) -> None:
        self.get_memory_snapshot = _FakeRemoteMethod(payload, issued)


class _FakePool:
    def __init__(self, groups: dict[int, list], issued: list) -> None:
        self.groups = groups
        self.issued = issued

    def handles(self, group_index: int) -> tuple:
        return tuple(_FakeHandle(payload, self.issued) for payload in self.groups[group_index])


class _SamplerLayout:
    """The sleep-pool layout the sampler must read its per-group rank count from."""

    def __init__(self, tp: int) -> None:
        self.tp = tp


class _FakeRay:
    def __init__(self, issued: list, *, error: Exception | None = None) -> None:
        self.issued = issued
        self.error = error
        self.calls: list[tuple[list, float, int]] = []

    def get(self, refs: list, *, timeout: float):
        self.calls.append((list(refs), timeout, len(self.issued)))
        if self.error is not None:
            raise self.error
        return list(refs)


def _trainer_for_payloads(groups: dict[int, list], *, error: Exception | None = None, tp: int = 2):
    issued: list = []
    fake_ray = _FakeRay(issued, error=error)
    trainer_class = _compile_trainer_class(
        "_memory_sample_invalid",
        "_sample_pool_memory",
        "_memory_samples_complete",
        ray_object=fake_ray,
    )
    trainer = trainer_class()
    trainer._elastic_pool = _FakePool(groups, issued)
    trainer._sleep_pool_layout = _SamplerLayout(tp)
    return trainer, fake_ray, issued


def test_sampler_ast_enforces_all_rank_batched_public_dispatch():
    _assert_sampler_ast_contract(_find_trainer_method("_sample_pool_memory"))


def test_sampler_timeout_default_is_pinned_and_the_gate_ceiling_tracks_it():
    """Widening the default to 1e9 leaves every other CPU test green.

    The trace gate only catches the consequence -- an implausible sampling_rpc_ms
    -- so a run that never actually reaches the timeout passes cleanly. The bound
    itself has to be pinned at the source, and the gate's ceiling is documented as
    double it, so the two are asserted together rather than left to drift.
    """
    node = _find_trainer_method("_sample_pool_memory")
    defaults = dict(zip([arg.arg for arg in node.args.kwonlyargs], node.args.kw_defaults, strict=True))
    assert "timeout_s" in defaults, f"timeout_s must stay keyword-only, found {sorted(defaults)}"
    default = defaults["timeout_s"]
    assert isinstance(default, ast.Constant) and default.value == 30.0, (
        f"the sampler's bounded wait must default to 30.0s, got {ast.unparse(default) if default else None}"
    )

    gate_tree = ast.parse((_ROOT / "scripts" / "phase1_memory_gate.py").read_text(encoding="utf-8"))
    ceilings = [
        statement.value
        for statement in gate_tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MAX_SAMPLING_RPC_MS" for target in statement.targets)
    ]
    assert len(ceilings) == 1, "the gate must define MAX_SAMPLING_RPC_MS exactly once"
    assert isinstance(ceilings[0], ast.Constant), "the ceiling must be a literal the gate can be read against"
    assert ceilings[0].value == default.value * 2 * 1000.0, (
        f"the gate ceiling {ceilings[0].value} ms no longer equals double the sampler's {default.value}s timeout; "
        "one of the two moved without the other"
    )


def test_sampler_ast_rejects_first_rank_serial_wait_mutation():
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    node = _find_trainer_method_in_tree(tree, "_sample_pool_memory")
    _, inner = _sampler_dispatch_loops(node)
    wait_statements = [
        statement
        for statement in node.body
        if any(
            isinstance(part, ast.Call) and _attribute_chain(part.func) == ("ray", "get")
            for part in _walk_without_nested_code(statement)
        )
    ]
    assert len(wait_statements) == 1, "mutation setup must find the single batched wait statement"
    node.body.remove(wait_statements[0])
    inner.body.extend([wait_statements[0], ast.Break()])
    ast.fix_missing_locations(tree)

    try:
        _assert_sampler_ast_contract(node)
    except AssertionError:
        pass
    else:
        raise AssertionError("sampler AST guard accepted first-rank serial waiting with break")


def test_sampler_pairs_coordinates_with_payloads_strictly():
    _assert_coordinate_payload_pairing_is_strict(_find_trainer_method("_sample_pool_memory"))


def test_sampler_zip_guard_rejects_dropping_strict():
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    node = _find_trainer_method_in_tree(tree, "_sample_pool_memory")
    zips = [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == ("zip",)
    ]
    assert len(zips) == 1, "mutation setup expects the single coordinate/payload zip"
    zips[0].keywords = [keyword for keyword in zips[0].keywords if keyword.arg != "strict"]
    ast.fix_missing_locations(tree)
    assert "zip(coordinates, payloads)" in ast.unparse(node), "mutation must keep the zip the weak test looked for"

    try:
        _assert_coordinate_payload_pairing_is_strict(node)
    except AssertionError as exc:
        assert "strict=True" in str(exc)
    else:
        raise AssertionError("sampler guard accepted a non-strict zip that can silently mis-attribute samples")


def test_sampler_reports_every_invalid_record_once_and_owns_coordinates():
    missing_allocated = _valid_memory_payload()
    missing_allocated.pop("memory_allocated_bytes")
    missing_reserved = _valid_memory_payload()
    missing_reserved.pop("memory_reserved_bytes")
    missing_free = _valid_memory_payload()
    missing_free.pop("memory_free_bytes")
    missing_total = _valid_memory_payload()
    missing_total.pop("memory_total_bytes")
    missing_sleep = _valid_memory_payload()
    missing_sleep.pop("actual_sleep_level")
    multi_invalid = _valid_memory_payload(sampling_error="failed")
    multi_invalid.pop("memory_total_bytes")
    multi_invalid.pop("actual_sleep_level")
    payloads = [
        _valid_memory_payload(group_index=999, tp_rank=999),
        _valid_memory_payload(sampling_error="failed"),
        missing_allocated,
        missing_reserved,
        missing_free,
        missing_total,
        missing_sleep,
        _valid_memory_payload(memory_free_bytes=101),
        "malformed",
        multi_invalid,
    ]
    trainer, fake_ray, issued = _trainer_for_payloads({7: payloads}, tp=len(payloads))

    result = trainer._sample_pool_memory([7], timeout_s=4.5)

    assert len(issued) == len(payloads)
    assert fake_ray.calls == [(payloads, 4.5, len(payloads))]
    assert result["expected_samples"] == len(payloads)
    assert result["observed_samples"] == len(payloads)
    assert result["error_count"] == len(payloads) - 1
    assert [(sample["group_index"], sample["tp_rank"]) for sample in result["samples"]] == [
        (7, tp_rank) for tp_rank in range(len(payloads))
    ]
    assert not trainer._memory_samples_complete(result)


def test_sampler_timeout_counts_each_rank_once():
    groups = {2: [_valid_memory_payload(), _valid_memory_payload()]}
    trainer, _, _ = _trainer_for_payloads(groups, error=TimeoutError("late"))

    result = trainer._sample_pool_memory([2])

    assert result["expected_samples"] == 2
    assert result["observed_samples"] == 2
    assert result["error_count"] == 2
    assert all(sample["sampling_error"] == "TimeoutError: late" for sample in result["samples"])
    assert not trainer._memory_samples_complete(result)


def test_sampler_survives_pool_handles_raising():
    """A wedged/renamed handles() must not propagate and abort a live resize."""

    class _FailingHandles:
        def handles(self, group_index):
            raise KeyError(f"group {group_index} missing")

    trainer_class = _compile_trainer_class(
        "_memory_sample_invalid", "_sample_pool_memory", "_memory_samples_complete", ray_object=_FakeRay([])
    )
    trainer = trainer_class()
    trainer._elastic_pool = _FailingHandles()
    trainer._sleep_pool_layout = _SamplerLayout(2)

    result = trainer._sample_pool_memory([0, 1])

    assert result["expected_samples"] == 4, "two groups at TP=2 were requested, whatever handles() did"
    assert result["observed_samples"] == 2
    assert result["error_count"] == 2
    assert [sample["group_index"] for sample in result["samples"]] == [0, 1]
    assert all(sample["tp_rank"] == -1 for sample in result["samples"])
    assert all(sample["sampling_error"].startswith("handles failed: KeyError:") for sample in result["samples"])
    assert not trainer._memory_samples_complete(result)


def test_sampler_survives_remote_dispatch_raising():
    """A handle whose .remote() raises before ray.get must not abort a live resize."""

    class _FailingRemote:
        def remote(self):
            raise RuntimeError("dispatch broke")

    class _FailingHandle:
        get_memory_snapshot = _FailingRemote()

    class _FailingHandlePool:
        def handles(self, group_index):
            return [_FailingHandle(), _FailingHandle()]

    trainer_class = _compile_trainer_class(
        "_memory_sample_invalid", "_sample_pool_memory", "_memory_samples_complete", ray_object=_FakeRay([])
    )
    trainer = trainer_class()
    trainer._elastic_pool = _FailingHandlePool()
    trainer._sleep_pool_layout = _SamplerLayout(2)

    result = trainer._sample_pool_memory([0])

    assert result["expected_samples"] == 2
    assert result["observed_samples"] == 2
    assert result["error_count"] == 2
    assert [sample["group_index"] for sample in result["samples"]] == [0, 0]
    assert [sample["tp_rank"] for sample in result["samples"]] == [0, 1]
    assert all(sample["sampling_error"].startswith("dispatch failed: RuntimeError:") for sample in result["samples"])
    assert not trainer._memory_samples_complete(result)


def test_sampler_mixes_dispatch_failures_with_real_samples():
    """One failing group alongside a healthy one must not lose the healthy samples."""

    class _MixedHandles:
        def handles(self, group_index):
            if group_index == 0:
                raise KeyError("group 0 missing")
            return tuple(_FakeHandle(_valid_memory_payload(), []) for _ in range(2))

    issued: list = []

    class _RecordingHandles(_MixedHandles):
        def handles(self, group_index):
            if group_index == 0:
                raise KeyError("group 0 missing")
            return tuple(_FakeHandle(_valid_memory_payload(), issued) for _ in range(2))

    fake_ray = _FakeRay(issued)
    trainer_class = _compile_trainer_class(
        "_memory_sample_invalid", "_sample_pool_memory", "_memory_samples_complete", ray_object=fake_ray
    )
    trainer = trainer_class()
    trainer._elastic_pool = _RecordingHandles()
    trainer._sleep_pool_layout = _SamplerLayout(2)

    result = trainer._sample_pool_memory([0, 1])

    assert result["expected_samples"] == 4, "the failed group's two ranks are still expected"
    assert result["observed_samples"] == 3
    assert result["error_count"] == 1
    failing = [sample for sample in result["samples"] if sample["group_index"] == 0]
    healthy = [sample for sample in result["samples"] if sample["group_index"] == 1]
    assert len(failing) == 1 and failing[0]["tp_rank"] == -1
    assert len(healthy) == 2 and not any(sample.get("sampling_error") for sample in healthy)


# --- the expectation must not be self-derived from what dispatch produced ---


def test_sampler_expectation_comes_from_the_layout_not_from_dispatch():
    """A truncated handle list must not read as a complete sample set.

    `pool.handles(0)` returning one handle at TP=2 is a broken RPC path, but while
    expected_samples was `len(refs) + len(dispatch_failures)` it equalled the number
    of samples by construction, so observed == expected was a tautology and the
    startup fail-fast stayed silent.
    """
    trainer, _, _ = _trainer_for_payloads({0: [_valid_memory_payload()]}, tp=2)

    result = trainer._sample_pool_memory([0])

    assert result["expected_samples"] == 2, (
        "the expectation must be len(group_indices) * layout.tp, independent of what handles() returned"
    )
    assert result["observed_samples"] == 1
    assert not trainer._memory_samples_complete(result), "a half-sampled group must never read as complete"


def test_sampler_expectation_survives_a_pool_that_yields_no_handles_at_all():
    """The 48257a9fa signature: every group returns zero handles.

    Self-derivation made expected_samples 0 here, which both reads as "nothing was
    asked for" and short-circuits the startup guard.
    """
    trainer, _, _ = _trainer_for_payloads({0: [], 1: []}, tp=2)

    result = trainer._sample_pool_memory([0, 1])

    assert result["expected_samples"] == 4, "two groups at TP=2 must expect four rank samples"
    assert result["observed_samples"] == 0
    assert not trainer._memory_samples_complete(result)


def test_sampler_expectation_scales_with_the_real_tp_not_a_constant():
    """Pins the multiplication: a hard-coded 2 would pass the TP=2 cases above."""
    trainer, _, _ = _trainer_for_payloads({0: [_valid_memory_payload() for _ in range(4)]}, tp=4)

    result = trainer._sample_pool_memory([0])

    assert result["expected_samples"] == 4
    assert result["observed_samples"] == 4
    assert trainer._memory_samples_complete(result), "a genuinely complete TP=4 group must still pass"


def test_sampler_expectation_is_zero_when_no_groups_are_requested():
    """No groups asked for is not a failure; the startup guard keys on the request."""
    trainer, _, _ = _trainer_for_payloads({}, tp=2)

    result = trainer._sample_pool_memory([])

    assert result["expected_samples"] == 0
    assert result["observed_samples"] == 0
    assert result["samples"] == []


def test_memory_samples_complete_recomputes_metadata_and_requires_equal_sleep_levels():
    trainer_class = _compile_trainer_class("_memory_sample_invalid", "_memory_samples_complete")
    complete = {
        "samples": [_valid_memory_payload(sleep_level=2), _valid_memory_payload(sleep_level=2)],
        "expected_samples": 2,
        "observed_samples": 2,
        "error_count": 0,
        "sampling_rpc_ms": 1.0,
    }
    assert trainer_class._memory_samples_complete(complete)

    mutations = []
    missing_samples = dict(complete)
    missing_samples.pop("samples")
    mutations.append(missing_samples)
    wrong_length = dict(complete, samples=complete["samples"][:1])
    mutations.append(wrong_length)
    wrong_observed = dict(complete, observed_samples=1)
    mutations.append(wrong_observed)
    false_error_metadata = dict(complete, error_count=1)
    mutations.append(false_error_metadata)
    disagreeing_sleep = dict(
        complete,
        samples=[_valid_memory_payload(sleep_level=1), _valid_memory_payload(sleep_level=2)],
    )
    mutations.append(disagreeing_sleep)
    invalid_sample = _valid_memory_payload()
    invalid_sample.pop("memory_total_bytes")
    hidden_invalid = dict(complete, samples=[invalid_sample, _valid_memory_payload()], error_count=0)
    mutations.append(hidden_invalid)
    accurately_reported_invalid = dict(hidden_invalid, error_count=1)
    mutations.append(accurately_reported_invalid)

    for mutation in mutations:
        assert not trainer_class._memory_samples_complete(mutation)


def test_completeness_ast_reads_real_samples_metadata_and_sleep_levels():
    _assert_completeness_ast_contract(_find_trainer_method("_memory_samples_complete"))


def test_fingerprint_ast_reads_exact_flat_config_fields():
    _assert_fingerprint_ast_contract(_find_trainer_method("_get_memory_config_fingerprint"))


def test_fingerprint_ast_rejects_hardcoded_values_with_string_decoys():
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    node = _find_trainer_method_in_tree(tree, "_get_memory_config_fingerprint")
    returned = _returned_dict(node)
    returned.values = [ast.Constant(index) for index, _ in enumerate(returned.values)]
    node.body.insert(
        1,
        ast.Expr(
            value=ast.Constant(
                "self.config.actor_rollout_ref.model.path rollout_cfg.gpu_memory_utilization "
                "rollout_cfg.tensor_model_parallel_size rollout_cfg.max_num_seqs "
                "rollout_cfg.enforce_eager rollout_cfg.free_cache_engine"
            )
        ),
    )
    ast.fix_missing_locations(tree)

    try:
        _assert_fingerprint_ast_contract(node)
    except AssertionError:
        pass
    else:
        raise AssertionError("fingerprint AST guard accepted hardcoded values hidden by string decoys")


def test_fingerprint_behavior_returns_the_six_live_config_values():
    trainer_class = _compile_trainer_class("_get_memory_config_fingerprint")
    trainer = trainer_class()
    rollout = type(
        "RolloutConfig",
        (),
        {
            "gpu_memory_utilization": object(),
            "tensor_model_parallel_size": object(),
            "max_num_seqs": object(),
            "enforce_eager": object(),
            "free_cache_engine": object(),
        },
    )()
    model = type("ModelConfig", (), {"path": object()})()
    actor_rollout_ref = type("ActorRolloutRef", (), {"rollout": rollout, "model": model})()
    trainer.config = type("Config", (), {"actor_rollout_ref": actor_rollout_ref})()

    assert trainer._get_memory_config_fingerprint() == {
        "model_path": model.path,
        "gpu_memory_utilization": rollout.gpu_memory_utilization,
        "tensor_model_parallel_size": rollout.tensor_model_parallel_size,
        "max_num_seqs": rollout.max_num_seqs,
        "enforce_eager": rollout.enforce_eager,
        "free_cache_engine": rollout.free_cache_engine,
    }


def _calls_to(node: ast.AST, chain: tuple[str, ...]) -> list[ast.Call]:
    return [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == chain
    ]


def _single_call(node: ast.AST, chain: tuple[str, ...]) -> ast.Call:
    calls = _calls_to(node, chain)
    assert len(calls) == 1, f"{node.name} must call {'.'.join(chain)} exactly once, found {len(calls)}"
    return calls[0]


def _local_assignments(node: ast.AST, name: str) -> list[ast.Assign]:
    return [
        statement
        for statement in _direct_executable_nodes(node)
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == (name,) for target in statement.targets)
    ]


def _single_local_assignment(node: ast.AST, name: str) -> ast.Assign:
    assignments = _local_assignments(node, name)
    assert len(assignments) == 1, f"{node.name} must assign {name} exactly once, found {len(assignments)}"
    return assignments[0]


def _attribute_writes(node: ast.AST, chain: tuple[str, ...]) -> list[ast.Assign | ast.AnnAssign]:
    writes: list[ast.Assign | ast.AnnAssign] = []
    for statement in _direct_executable_nodes(node):
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if any(_attribute_chain(target) == chain for target in targets):
            writes.append(statement)
    return writes


def _dict_value_nodes(node: ast.AST, key: str) -> list[ast.expr]:
    """Values bound to `key` by any dict built in this method's direct body.

    Accepts both `{"key": value}` and `dict(key=value)`: the guards care that the
    right expression reaches the trace, not which spelling built the dict.
    """
    values: list[ast.expr] = []
    for part in _direct_executable_nodes(node):
        if isinstance(part, ast.Dict):
            values.extend(
                value
                for key_node, value in zip(part.keys, part.values, strict=True)
                if isinstance(key_node, ast.Constant) and key_node.value == key
            )
        elif isinstance(part, ast.Call) and _attribute_chain(part.func) == ("dict",):
            values.extend(keyword.value for keyword in part.keywords if keyword.arg == key)
    return values


def _completeness_gate(node: ast.AST) -> ast.If:
    gates = [
        statement
        for statement in _direct_executable_nodes(node)
        if isinstance(statement, ast.If)
        and any(
            isinstance(part, ast.Call) and _attribute_chain(part.func) == ("self", "_memory_samples_complete")
            for part in ast.walk(statement.test)
        )
    ]
    assert len(gates) == 1, f"expected exactly one completeness gate, found {len(gates)}"
    return gates[0]


def _assert_startup_gate_fires_on_incompleteness(gate: ast.If) -> None:
    negated = [
        part
        for part in ast.walk(gate.test)
        if isinstance(part, ast.UnaryOp)
        and isinstance(part.op, ast.Not)
        and isinstance(part.operand, ast.Call)
        and _attribute_chain(part.operand.func) == ("self", "_memory_samples_complete")
    ]
    assert len(negated) == 1, (
        "the gate must raise when samples are NOT complete; an un-negated check aborts on healthy "
        "runs and stays silent on the empty samples it exists to catch"
    )
    raises = [
        part.exc
        for part in ast.walk(gate)
        if isinstance(part, ast.Raise) and isinstance(part.exc, ast.Call)
    ]
    assert len(raises) == 1 and _attribute_chain(raises[0].func) == ("RuntimeError",), (
        "the incomplete branch must raise RuntimeError, not log and continue"
    )


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


def _non_probe_span_ancestors_of_sample_calls(tree: ast.Module) -> dict[int, list]:
    """For every _sample_pool_memory call, the non-memory_probe span names enclosing it.

    Walks the whole trainer class rather than one method: an earlier cross-span
    leak (the post-sync probe drifting into resize_postprocess), so this check
    must not be scoped to a single method the way the other AST guards are.
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _attribute_chain(node.func) == ("self", "_sample_pool_memory")
    ]

    offenders: dict[int, list] = {}
    for call in calls:
        spans = []
        current = getattr(call, "parent", None)
        while current is not None:
            name = _resize_trace_span_name(current)
            if name is not None:
                spans.append(name)
            current = getattr(current, "parent", None)
        bad = [name for name in spans if not (isinstance(name, str) and name.startswith("memory_probe"))]
        if bad:
            offenders[call.lineno] = bad
    return offenders


def test_no_memory_probe_sampling_call_is_nested_in_an_a_line_span():
    """B-line probe RPC cost must never inflate an A-line resize_trace_span's duration.

    Regression guard: memory_probe_wake_after_sync's real
    sampling call used to sit inside resize_postprocess, so that span's measured
    duration silently included the probe's RPC round trip.
    """
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    offenders = _non_probe_span_ancestors_of_sample_calls(tree)
    assert not offenders, f"_sample_pool_memory calls nested inside non-probe spans: {offenders}"


def _assert_samples_whole_group_set(assignment: ast.Assign, source_chain: tuple[str, ...]) -> None:
    """The probe must sample the entire group set, never a truncated slice.

    Tolerates any spelling that passes the whole collection (list(...), tuple(...),
    sorted(...), or the attribute itself); rejects a subscript/slice, which is how
    a `[:1]`-style regression silently reduces the probe to one group.
    """
    call = assignment.value
    assert isinstance(call, ast.Call), "each probe sample must be a direct call"
    assert len(call.args) == 1 and not any(keyword.arg == "group_indices" for keyword in call.keywords), (
        "the group set must be the single positional argument"
    )
    argument = call.args[0]
    assert not any(isinstance(part, (ast.Slice, ast.Subscript)) for part in ast.walk(argument)), (
        f"{ast.unparse(argument)} truncates the group set; every group in it must be sampled"
    )
    assert any(_attribute_chain(part) == source_chain for part in ast.walk(argument)), (
        f"the probe must sample {'.'.join(source_chain)}, got {ast.unparse(argument)}"
    )


def test_startup_baseline_samples_outside_the_finalize_span():
    node = _find_trainer_method("_finalize_elastic_pool_replicas")
    samples = _sample_assignments(node)
    assert len(samples) == 2, f"startup must sample the awake and the sleeping side, found {len(samples)}"
    _assert_samples_whole_group_set(samples[0], ("pool", "awake_indices"))
    _assert_samples_whole_group_set(samples[1], ("pool", "sleeping_indices"))

    startup_spans = [
        part for part in _direct_executable_nodes(node) if _resize_trace_span_name(part) == "memory_probe_startup"
    ]
    assert len(startup_spans) == 1, "startup must emit exactly one memory_probe_startup span"

    prebuild_writes = [
        statement
        for statement in _direct_executable_nodes(node)
        if isinstance(statement, ast.Assign) and any(_target_names(t) == ("prebuild_s",) for t in statement.targets)
    ]
    assert len(prebuild_writes) == 1, "finalize must compute prebuild_s exactly once"
    log_calls = [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == ("logger", "info")
    ]
    assert log_calls, "finalize must keep its accounting log"

    first_sample_at = samples[0].lineno
    assert prebuild_writes[0].lineno < first_sample_at, "sample after prebuild accounting, not inside it"
    assert max(call.lineno for call in log_calls) < first_sample_at, (
        "startup probe must follow the complete finalize accounting block"
    )


def test_startup_baseline_fails_fast_on_incomplete_samples():
    node = _find_trainer_method("_finalize_elastic_pool_replicas")
    _assert_startup_gate_fires_on_incompleteness(_completeness_gate(node))

    gate_loops = [
        statement
        for statement in node.body
        if isinstance(statement, ast.For)
        and any(part is _completeness_gate(node) for part in ast.walk(statement))
    ]
    assert len(gate_loops) == 1, "the gate must iterate the labelled results rather than check only one side"
    labelled = gate_loops[0].iter
    assert isinstance(labelled, ast.Tuple) and len(labelled.elts) == 2, (
        "the gate must loop over exactly the awake and sleeping results"
    )
    pairs = []
    for element in labelled.elts:
        assert isinstance(element, ast.Tuple) and len(element.elts) == 3, (
            "each item must be a (label, groups, result) triple: gating on the requested groups rather than on "
            "expected_samples is what makes an all-empty probe loud"
        )
        label, groups, result = element.elts
        assert isinstance(label, ast.Constant), "the label must be a literal for the error message"
        assert isinstance(result, ast.Name), "the result must be the local a real sampling call produced"
        chain = _attribute_chain(groups)
        assert chain == ("pool", f"{label.value}_indices"), (
            f"the {label.value} side must gate on the live pool.{label.value}_indices, got {ast.unparse(groups)}"
        )
        pairs.append((label.value, result.id))
    sampled_locals = {assignment.targets[0].id for assignment in _sample_assignments(node)}
    assert {label for label, _ in pairs} == {"awake", "sleeping"}, f"both sides must be gated, got {pairs}"
    assert {name for _, name in pairs} == sampled_locals, (
        f"the gate must read the real sample locals {sorted(sampled_locals)}, got {pairs}"
    )


def test_startup_gate_rejects_inverted_completeness_condition():
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    node = _find_trainer_method_in_tree(tree, "_finalize_elastic_pool_replicas")
    gate = _completeness_gate(node)
    negated = [
        part
        for part in ast.walk(gate.test)
        if isinstance(part, ast.UnaryOp)
        and isinstance(part.op, ast.Not)
        and isinstance(part.operand, ast.Call)
        and _attribute_chain(part.operand.func) == ("self", "_memory_samples_complete")
    ]
    assert len(negated) == 1, "mutation setup expects one negated completeness call"

    class _DropNot(ast.NodeTransformer):
        def visit_UnaryOp(self, part: ast.UnaryOp):
            self.generic_visit(part)
            return part.operand if part is negated[0] else part

    gate.test = _DropNot().visit(gate.test)
    ast.fix_missing_locations(tree)
    assert "not self._memory_samples_complete" not in ast.unparse(gate), "mutation must remove the negation"
    assert "raise RuntimeError" in ast.unparse(gate), "mutation must keep the raise the weak test looked for"

    try:
        _assert_startup_gate_fires_on_incompleteness(_completeness_gate(node))
    except AssertionError as exc:
        assert "NOT complete" in str(exc)
    else:
        raise AssertionError("startup gate guard accepted an inverted condition that raises on healthy samples")


def test_startup_gate_behavior_raises_only_when_samples_are_incomplete():
    trainer_class = _compile_trainer_class("_memory_sample_invalid", "_memory_samples_complete")
    complete = {
        "samples": [_valid_memory_payload(sleep_level=2)],
        "expected_samples": 1,
        "observed_samples": 1,
        "error_count": 0,
        "sampling_rpc_ms": 1.0,
    }
    incomplete = dict(complete, samples=[_valid_memory_payload(sampling_error="boom")], error_count=1)
    empty = {
        "samples": [],
        "expected_samples": 0,
        "observed_samples": 0,
        "error_count": 0,
        "sampling_rpc_ms": 0.0,
    }

    gate = _completeness_gate(_find_trainer_method("_finalize_elastic_pool_replicas"))
    condition = ast.Expression(body=gate.test)
    ast.fix_missing_locations(condition)
    evaluate = compile(condition, str(_RAY_TRAINER), "eval")

    def _fires(groups, result) -> bool:
        return bool(eval(evaluate, {"self": trainer_class, "groups": groups, "result": result}))

    # Groups were requested: anything short of a complete sample set must fire,
    # including the all-empty payload whose expected_samples == 0 used to
    # short-circuit this condition into silence.
    outcomes = {
        "complete": _fires((0, 1), complete),
        "incomplete": _fires((0, 1), incomplete),
        "empty": _fires((0, 1), empty),
    }
    assert outcomes == {"complete": False, "incomplete": True, "empty": True}, (
        f"gate must fire whenever a requested side is not complete, got {outcomes}"
    )

    # No groups on this side is legitimate (k0 == num_groups), so nothing fires.
    assert not _fires((), empty), "a side with no groups must not be gated"
    assert not _fires((), complete), "a side with no groups must not be gated"


def _awaited_pool_call(node: ast.AST, method: str) -> ast.Await:
    awaits = [
        part
        for part in _direct_executable_nodes(node)
        if isinstance(part, ast.Await)
        and isinstance(part.value, ast.Call)
        and _attribute_chain(part.value.func) == ("pool", method)
    ]
    assert len(awaits) == 1, f"schedule must await pool.{method} exactly once, found {len(awaits)}"
    return awaits[0]


def _sample_assignments(node: ast.AST) -> list[ast.Assign]:
    """Every `local = self._sample_pool_memory(...)` in the method's direct body."""
    assignments = [
        statement
        for statement in _direct_executable_nodes(node)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and _attribute_chain(statement.value.func) == ("self", "_sample_pool_memory")
    ]
    for assignment in assignments:
        assert len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name), (
            "each probe sample must bind to exactly one plain local"
        )
    return sorted(assignments, key=lambda statement: statement.lineno)


def test_sleep_and_wake_sampling_points_exist_with_distinct_semantics():
    node = _find_trainer_method("_apply_sleep_pool_schedule")
    span_names = {
        _resize_trace_span_name(part)
        for part in _direct_executable_nodes(node)
        if _resize_trace_span_name(part) is not None
    }
    assert "memory_probe_sleep" in span_names
    assert "memory_probe_wake_mapped" in span_names, "wake_groups only maps empty memory under level 2"

    samples = _sample_assignments(node)
    assert len(samples) == 3, f"schedule needs sleep_before, sleep_after and wake_mapped, found {len(samples)}"
    sleep_before, sleep_after, wake_mapped = samples
    _assert_samples_whole_group_set(sleep_before, ("to_sleep",))
    _assert_samples_whole_group_set(sleep_after, ("to_sleep",))
    _assert_samples_whole_group_set(wake_mapped, ("to_wake",))
    sleep_await = _awaited_pool_call(node, "sleep_groups")
    wake_await = _awaited_pool_call(node, "wake_groups")

    assert sleep_before.lineno < sleep_await.lineno < sleep_after.lineno, (
        "the sleep probe must bracket the real sleep_groups await"
    )
    assert wake_await.lineno < wake_mapped.lineno, "wake occupancy must be sampled after the real wake_groups await"
    assert sleep_after.lineno < wake_await.lineno, "sleep must be fully accounted before wake begins (D6)"

    names = {"sleep_before": sleep_before, "sleep_after": sleep_after, "wake_mapped_after": wake_mapped}
    for key, assignment in names.items():
        recorded = _dict_value_nodes(node, key)
        assert len(recorded) == 1, f"trace metadata must carry exactly one {key}, found {len(recorded)}"
        assert isinstance(recorded[0], ast.Name) and recorded[0].id == assignment.targets[0].id, (
            f"{key} must publish the local its own sampling call returned, not a substitute payload"
        )

    for label in ("memory_probe_sleep", "memory_probe_wake_mapped"):
        emitted = [
            part
            for part in _direct_executable_nodes(node)
            if _resize_trace_span_name(part) == label
        ]
        assert len(emitted) == 1, f"each resize probe span must be emitted once, found {len(emitted)} {label}"


def _is_disabled_pool_early_return(statement: ast.AST) -> bool:
    """True for `if pool is None or layout is None: return None`-shaped guards."""
    if not isinstance(statement, ast.If):
        return False
    returns_none = any(
        isinstance(part, ast.Return) and isinstance(part.value, ast.Constant) and part.value.value is None
        for part in ast.walk(statement)
    )
    tests_pool_or_layout = any(
        isinstance(part, ast.Name) and part.id in ("pool", "layout") for part in ast.walk(statement.test)
    )
    return returns_none and tests_pool_or_layout


_PROBE_RESULT_LOCAL_NAMES = {"sleep_before", "sleep_after", "wake_mapped"}


def _probe_path_aborts(node: ast.AST) -> list[ast.AST]:
    """Every way the schedule could abort or redirect a live resize over probe data.

    Covers explicit raise/assert, a completeness-gate branch, any conditional
    whose test reads a probe result local directly (e.g. wake_mapped['error_count']),
    and any Return that is neither the disabled-pool guard's `return None` nor the
    method's final `return topology` -- an added early return driven by probe data
    is a business-flow change that would abort or reroute a live resize just as
    surely as a raise would.
    """
    disabled_pool_guard = next(
        (statement for statement in node.body if _is_disabled_pool_early_return(statement)), None
    )
    final_return = node.body[-1] if node.body and isinstance(node.body[-1], ast.Return) else None

    permitted_returns: set[int] = set()
    if disabled_pool_guard is not None:
        permitted_returns.update(id(part) for part in ast.walk(disabled_pool_guard) if isinstance(part, ast.Return))
    if final_return is not None:
        permitted_returns.add(id(final_return))

    aborts: list[ast.AST] = []
    for part in _direct_executable_nodes(node):
        if isinstance(part, (ast.Raise, ast.Assert)):
            aborts.append(part)
        elif isinstance(part, ast.Return) and id(part) not in permitted_returns:
            aborts.append(part)
        elif isinstance(part, (ast.If, ast.While)) and any(
            (
                isinstance(candidate, ast.Call)
                and _attribute_chain(candidate.func) == ("self", "_memory_samples_complete")
            )
            or (isinstance(candidate, ast.Name) and candidate.id in _PROBE_RESULT_LOCAL_NAMES)
            for candidate in ast.walk(part.test)
        ):
            aborts.append(part)
    return aborts


def test_schedule_has_no_completeness_gates_or_raises_in_the_probe_paths():
    aborts = _probe_path_aborts(_find_trainer_method("_apply_sleep_pool_schedule"))
    assert not aborts, (
        "resize incompleteness must stay in trace metadata rather than aborting the run; found "
        f"{[type(part).__name__ for part in aborts]}"
    )


def test_wake_after_sync_is_recorded_after_the_real_weight_sync():
    """Ensure the real _sample_pool_memory call for after_sync runs after sync await."""
    resize_node = _find_trainer_method("_staged_resize_shared_pool")

    probe_spans = [
        part
        for part in _direct_executable_nodes(resize_node)
        if _resize_trace_span_name(part) == "memory_probe_wake_after_sync"
    ]
    assert len(probe_spans) == 1, "staged resize must emit exactly one memory_probe_wake_after_sync span"

    # The real semantics check: the actual sampling call must be after the sync await.
    # Truncation is already excluded structurally below: the arg must be exactly
    # list(pending["groups"]), so list(...)[:1] and list(pending["groups"][:1]) both fail.
    after_sync_assignment = _single_local_assignment(resize_node, "after_sync")
    assert isinstance(after_sync_assignment.value, ast.Call), "after_sync must be bound to a direct call"
    assert _attribute_chain(after_sync_assignment.value.func) == ("self", "_sample_pool_memory"), (
        "after_sync must be the real sampling call"
    )
    assert len(after_sync_assignment.value.args) == 1, "_sample_pool_memory takes exactly one positional arg"
    groups_arg = after_sync_assignment.value.args[0]
    assert isinstance(groups_arg, ast.Call) and _attribute_chain(groups_arg.func) == ("list",), (
        "the groups arg must be list(...)"
    )
    assert len(groups_arg.args) == 1 and isinstance(groups_arg.args[0], ast.Subscript), (
        "list() must wrap a subscript read"
    )
    subscripted = groups_arg.args[0]
    assert isinstance(subscripted.value, ast.Name) and subscripted.value.id == "pending", (
        "the subscription must read from the local pending"
    )
    assert (
        isinstance(subscripted.slice, ast.Constant) and subscripted.slice.value == "groups"
    ), "the subscript must be ['groups']"

    sync_awaits = [
        statement
        for statement in _direct_executable_nodes(resize_node)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Await)
        and isinstance(statement.value.value, ast.Call)
        and _attribute_chain(statement.value.value.func) == ("self", "_sync_rollout_weights_after_resize_async")
    ]
    assert len(sync_awaits) == 1, "resize must have exactly one _sync_rollout_weights_after_resize_async await"
    sync_await_stmt = sync_awaits[0]

    assert after_sync_assignment.lineno > sync_await_stmt.lineno, (
        "after_sync = self._sample_pool_memory(...) must come after the weight sync await"
    )

    metadata = _dict_value_nodes(resize_node, "wake_after_sync")
    assert len(metadata) == 1, "trace metadata must carry exactly one wake_after_sync"
    assert isinstance(metadata[0], ast.Name) and metadata[0].id == "after_sync", (
        "wake_after_sync metadata must be the same local as the real sampling call returned"
    )

    schedule_probe_spans = [
        part
        for part in _direct_executable_nodes(_find_trainer_method("_apply_sleep_pool_schedule"))
        if _resize_trace_span_name(part) == "memory_probe_wake_after_sync"
    ]
    assert not schedule_probe_spans, "the post-sync probe belongs in staged resize, not the schedule"


def _steps_between_the_wake_samples(node: ast.AST) -> list[str]:
    """The real sequence the resize runs between wake_mapped and wake_after_sync.

    Derived from the source rather than written down, so inserting a fifth step
    breaks the test instead of silently invalidating the recorded label.
    """
    schedule_calls = [
        part
        for part in ast.walk(node)
        if isinstance(part, ast.Call) and _attribute_chain(part.func) == ("self", "_apply_sleep_pool_schedule")
    ]
    assert len(schedule_calls) == 1, "the wake_mapped sample is taken inside the one schedule call"
    start = schedule_calls[0].lineno
    end = _single_local_assignment(node, "after_sync").lineno

    steps: list[str] = []
    for statement in node.body:  # type: ignore[attr-defined]
        if not start < statement.lineno < end:
            continue
        name = _resize_trace_span_name(statement)
        if name == "resize_postprocess":
            for inner in statement.body:
                inner_name = _resize_trace_span_name(inner)
                if inner_name:
                    steps.append(inner_name)
                elif (
                    isinstance(inner, ast.Expr)
                    and isinstance(inner.value, ast.Await)
                    and isinstance(inner.value.value, ast.Call)
                    and isinstance(inner.value.value.func, ast.Attribute)
                ):
                    steps.append(inner.value.value.func.attr)
        elif name:
            steps.append(name)
    return steps


def test_post_sync_span_records_what_actually_runs_between_the_two_wake_samples():
    """`wake_after_sync - wake_mapped` is not "the weight broadcast".

    A manager-level wake_up_async runs in that interval and can itself allocate,
    so the delta is not attributable to the broadcast alone. The span must say so
    in its own metadata, and the list is derived from the real code so it cannot
    quietly go stale.
    """
    resize_node = _find_trainer_method("_staged_resize_shared_pool")
    recorded = _dict_value_nodes(resize_node, "between")
    assert len(recorded) == 1, "the post-sync span must carry exactly one between list"
    listed = recorded[0]
    assert isinstance(listed, ast.List) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in listed.elts
    ), f"between must be a list of string literals, got {ast.unparse(listed)}"

    names = [element.value for element in listed.elts]
    expected = _steps_between_the_wake_samples(resize_node)
    assert names == expected, (
        f"between says {names} but the code really runs {expected} between the two wake samples; "
        "an out-of-date list is worse than none"
    )
    assert "wake_up_async" in names, (
        "the manager-level wake is the step that makes the interval more than a broadcast; it must be named"
    )

    span_metadata = _dict_value_nodes(resize_node, "wake_after_sync")
    assert len(span_metadata) == 1, "the between list must live on the post-sync span, beside its payload"


def _assert_transition_id_generated_once(node: ast.AST) -> None:
    """The id must be minted once per schedule and every record reuse that local."""
    generated = _single_call(node, ("self", "_transition_id"))
    assignment = _single_local_assignment(node, "transition_id")
    assert assignment.value is generated, "transition_id must be the local bound to the single _transition_id call"
    assert len(generated.args) == 1 and _same_expression(generated.args[0], "k"), (
        "the id must be derived from this schedule's k"
    )

    recorded = _dict_value_nodes(node, "transition_id")
    assert len(recorded) >= 3, f"sleep, wake and the pending slot must all carry the id, found {len(recorded)}"
    for value in recorded:
        assert isinstance(value, ast.Name) and value.id == "transition_id", (
            "every record must reuse the same local transition_id; calling _transition_id again mints a "
            "fresh counter value and silently unpairs wake_mapped from wake_after_sync"
        )


def test_transition_id_pairs_the_two_wake_records():
    init_node = _find_trainer_method("__init__")
    counter_inits = [
        write
        for write in _attribute_writes(init_node, ("self", "_memory_transition_counter"))
        if isinstance(write.value, ast.Constant) and write.value.value == 0
    ]
    assert len(counter_inits) == 1, "__init__ must seed the transition counter at 0"
    pending_inits = [
        write
        for write in _attribute_writes(init_node, ("self", "_pending_memory_transition"))
        if isinstance(write.value, ast.Constant) and write.value.value is None
    ]
    assert len(pending_inits) == 1, "__init__ must seed the pending slot as None"

    _assert_transition_id_generated_once(_find_trainer_method("_apply_sleep_pool_schedule"))

    resize_node = _find_trainer_method("_staged_resize_shared_pool")
    recorded = _dict_value_nodes(resize_node, "transition_id")
    assert len(recorded) == 1, f"staged resize must record exactly one transition_id, found {len(recorded)}"
    assert isinstance(recorded[0], ast.Subscript), "the recorded id must be read out of the pending slot"
    assert isinstance(recorded[0].value, ast.Name) and recorded[0].value.id == "pending", (
        "the post-sync record must reuse the id the schedule handed it, not mint a new one"
    )
    assert isinstance(recorded[0].slice, ast.Constant) and recorded[0].slice.value == "transition_id"

    pending_reads = [
        statement
        for statement in _direct_executable_nodes(resize_node)
        if isinstance(statement, ast.Assign)
        and any(_target_names(target) == ("pending",) for target in statement.targets)
        and _attribute_chain(statement.value) == ("self", "_pending_memory_transition")
    ]
    assert len(pending_reads) == 1, "staged resize must read the pending slot into a local exactly once"
    pending_clears = [
        write
        for write in _attribute_writes(resize_node, ("self", "_pending_memory_transition"))
        if isinstance(write.value, ast.Constant) and write.value.value is None
    ]
    assert len(pending_clears) == 1, "staged resize must clear the slot after consuming it"
    assert pending_reads[0].lineno < pending_clears[0].lineno, (
        "the local must capture the transition before the shared slot is cleared"
    )


def test_transition_id_guard_rejects_regenerating_the_id_for_the_pending_slot():
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    node = _find_trainer_method_in_tree(tree, "_apply_sleep_pool_schedule")
    pending_writes = _attribute_writes(node, ("self", "_pending_memory_transition"))
    pending_payloads = [
        write.value
        for write in pending_writes
        if isinstance(write, ast.Assign)
        and (
            isinstance(write.value, ast.Dict)
            or (isinstance(write.value, ast.Call) and _attribute_chain(write.value.func) == ("dict",))
        )
    ]
    assert len(pending_payloads) == 1, "mutation setup expects one pending-slot payload"
    payload = pending_payloads[0]

    regenerated = ast.parse("self._transition_id(k)", mode="eval").body
    if isinstance(payload, ast.Dict):
        items = _dict_items(payload)
        assert isinstance(items["transition_id"], ast.Name), "mutation setup expects the local Name"
        payload.values = [regenerated if key == "transition_id" else value for key, value in items.items()]
    else:
        keywords = {keyword.arg: keyword for keyword in payload.keywords}
        assert isinstance(keywords["transition_id"].value, ast.Name), "mutation setup expects the local Name"
        keywords["transition_id"].value = regenerated
    ast.fix_missing_locations(tree)
    assert "transition_id" in ast.unparse(node), "mutation must keep the key the weak test looked for"

    try:
        _assert_transition_id_generated_once(node)
    except AssertionError as exc:
        assert "exactly once" in str(exc) or "same local transition_id" in str(exc)
    else:
        raise AssertionError("transition-id guard accepted a regenerated id that unpairs the two wake records")


def test_schedule_guard_rejects_an_added_assert_on_probe_data():
    tree = ast.parse(_RAY_TRAINER.read_text(encoding="utf-8"))
    node = _find_trainer_method_in_tree(tree, "_apply_sleep_pool_schedule")
    added = ast.parse("assert not wake_mapped['error_count']\n").body[0]
    node.body.insert(len(node.body) - 1, added)
    ast.fix_missing_locations(tree)
    mutated_src = ast.unparse(node)
    assert "_memory_samples_complete" not in mutated_src and "raise" not in mutated_src, (
        "mutation must stay invisible to the old literal exclusions"
    )

    assert _probe_path_aborts(node), (
        "schedule guard accepted an assert that aborts a live resize on incomplete probe data"
    )


def test_pending_transition_is_cleared_before_each_schedule():
    node = _find_trainer_method("_apply_sleep_pool_schedule")
    writes = _attribute_writes(node, ("self", "_pending_memory_transition"))
    assert len(writes) >= 2, "schedule must clear the slot on entry and only then set a fresh transition"

    clears = [
        write
        for write in writes
        if isinstance(write, ast.Assign)
        and isinstance(write.value, ast.Constant)
        and write.value.value is None
    ]
    assert len(clears) == 1, "expected exactly one unconditional clear of the pending slot"
    clear = clears[0]

    assert clear in node.body, (
        "the clear must sit in the method's top-level body; nesting it under a branch lets a resize that "
        "raised mid-flight leave a stale transition_id for the next resize to consume"
    )
    for other in writes:
        if other is clear:
            continue
        assert other.lineno > clear.lineno, "the pending slot must be cleared before a new transition is stored"

    sample_calls = _calls_to(node, ("self", "_sample_pool_memory"))
    assert sample_calls, "schedule must sample memory"
    assert all(call.lineno > clear.lineno for call in sample_calls), (
        "clearing on entry must precede all sampling so no probe can observe the previous resize's state"
    )


def test_pending_transition_clear_precedes_the_disabled_pool_early_return():
    """The clear must run even when the pool is disabled, not only on the live path.

    A stale transition_id surviving because pool/layout happened to be None this
    call is exactly the corpse the entry-clear exists to prevent -- disabled-pool
    runs must still scrub state left by an earlier, differently-configured resize.
    """
    node = _find_trainer_method("_apply_sleep_pool_schedule")
    clears = [
        statement
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(_attribute_chain(target) == ("self", "_pending_memory_transition") for target in statement.targets)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is None
    ]
    assert len(clears) == 1, "expected exactly one unconditional clear of the pending slot"
    clear = clears[0]

    early_returns = [statement for statement in node.body if _is_disabled_pool_early_return(statement)]
    assert len(early_returns) == 1, f"expected exactly one disabled-pool early return, found {len(early_returns)}"
    guard = early_returns[0]

    clear_index = node.body.index(clear)
    guard_index = node.body.index(guard)
    assert clear_index < guard_index, (
        "the entry-clear must run before the disabled-pool early return, so a resize with pool=None "
        "still scrubs a stale transition_id left by a previous, differently-configured resize"
    )


def test_pending_transition_clear_survives_a_failing_resize():
    """A resize raising between wake and post-sync must not poison the next one."""
    node = _find_trainer_method("_apply_sleep_pool_schedule")
    clear = [
        statement
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(_attribute_chain(target) == ("self", "_pending_memory_transition") for target in statement.targets)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is None
    ]
    assert len(clear) == 1, "mutation setup expects the top-level clear"

    trainer = type("Trainer", (), {})()
    trainer._pending_memory_transition = {"transition_id": "stale_from_crashed_resize", "groups": [3]}
    module = ast.Module(body=[clear[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(_RAY_TRAINER), "exec"), {"self": trainer})

    assert trainer._pending_memory_transition is None, (
        "entry clear must drop a stale transition regardless of how the previous resize ended"
    )


def test_staged_resize_call_site_clears_the_pending_slot_in_a_finally():
    """A raising resize must scrub its own transition, not leave it for the next one.

    The schedule's entry clear is a backstop: between the raise and the next
    schedule call, metrics or a checkpoint could still read a transition whose
    post-sync probe never happened.
    """
    node = _find_trainer_method("_maybe_dynamic_resize")
    tries = [
        statement
        for statement in _direct_executable_nodes(node)
        if isinstance(statement, ast.Try)
        and any(
            isinstance(part, ast.Call) and _attribute_chain(part.func) == ("self", "_staged_resize_shared_pool")
            for part in ast.walk(statement)
        )
    ]
    assert len(tries) == 1, "the staged resize await must be wrapped exactly once"
    guarded = tries[0]

    assert any(
        isinstance(part, ast.Call) and _attribute_chain(part.func) == ("self", "_staged_resize_shared_pool")
        for statement in guarded.body
        for part in ast.walk(statement)
    ), "the staged resize call must be inside the try body, not the finally"

    assert guarded.finalbody, "the wrapper must use finally so a raise still clears the slot"
    clears = [
        statement
        for statement in guarded.finalbody
        if isinstance(statement, ast.Assign)
        and any(_attribute_chain(target) == ("self", "_pending_memory_transition") for target in statement.targets)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is None
    ]
    assert len(clears) == 1, "finally must clear the pending transition exactly once"
    assert not guarded.handlers, (
        "instrumentation must not swallow the resize's exception; finally-only keeps the failure loud"
    )


class _RecordedSpan:
    """Records one resize_trace_span entry/exit against a shared event log."""

    def __init__(self, events: list, name: str, metadata: dict | None) -> None:
        self.events = events
        self.name = name
        self.metadata = metadata

    def __enter__(self):
        self.events.append(("span_enter", self.name, self.metadata))
        return self

    def __exit__(self, *exc_info):
        self.events.append(("span_exit", self.name))
        return False


def _schedule_trainer(events: list, *, to_sleep: tuple, to_wake: tuple, groups: int = 4):
    """A trainer whose real _apply_sleep_pool_schedule runs against fakes.

    Only _apply_sleep_pool_schedule and _transition_id come from the real source;
    everything the schedule touches is a fake that appends to `events`, so the test
    observes runtime ORDER rather than AST position.
    """

    def _span(_config, name, *, step=None, lane=None, metadata=None):
        return _RecordedSpan(events, name, metadata)

    class _Layout:
        num_elastic_groups = groups

        def schedule_from_rollout_cards(self, cards):
            return int(cards)

    class _Topology:
        @staticmethod
        def from_schedule(**kwargs):
            return "topology"

    class _Pool:
        def handles(self, group_index):
            return (f"h{group_index}",)

        def reconcile(self, topology):
            return list(to_sleep), list(to_wake)

        async def sleep_groups(self, target):
            events.append(("sleep_groups", list(target)))
            return 1.5

        async def wake_groups(self, target):
            events.append(("wake_groups", list(target)))
            return 2.5

    trainer_class = _compile_trainer_class(
        "_apply_sleep_pool_schedule",
        "_transition_id",
        extra_namespace={
            "resize_trace_span": _span,
            "AwakeTopology": _Topology,
            "Role": type("Role", (), {"Rollout": "rollout"}),
            "logger": type("Log", (), {"info": staticmethod(lambda *a, **k: None)}),
        },
    )
    trainer = trainer_class()
    trainer._elastic_pool = _Pool()
    trainer._sleep_pool_layout = _Layout()
    trainer._resize_trace_config = None
    trainer._memory_transition_counter = 0
    trainer._pending_memory_transition = None
    trainer._trace_step = lambda: 7
    trainer._pool_world_size_from_spec = lambda role, spec: 2
    trainer._permanent_rollout_workers = lambda: []
    trainer.actor_wg = type("WG", (), {"workers": []})()
    trainer._update_resize_execution_metrics = lambda **kwargs: events.append(("metrics", dict(kwargs)))
    trainer._get_memory_config_fingerprint = lambda: {"fingerprint": "real"}

    def _sample(group_indices, **kwargs):
        # Unique token per call so a swapped/overwritten payload is detectable.
        token = f"sample#{len([e for e in events if e[0] == 'sample'])}"
        events.append(("sample", list(group_indices), token))
        return {"token": token, "groups": list(group_indices)}

    trainer._sample_pool_memory = _sample
    return trainer


def test_schedule_runtime_order_samples_around_the_real_sleep_and_wake():
    """Runtime proof of ordering: sample, sleep, sample, then wake, sample."""
    import asyncio

    events: list = []
    # Multi-group on BOTH sides: a single-group set cannot distinguish full
    # sampling from a `[:1]`-truncated probe.
    trainer = _schedule_trainer(events, to_sleep=(0, 1), to_wake=(2, 3))
    topology = asyncio.run(trainer._apply_sleep_pool_schedule({"n": 2}))
    assert topology == "topology"

    order = [event[0] for event in events]
    sleep_at = order.index("sleep_groups")
    wake_at = order.index("wake_groups")
    samples = [index for index, name in enumerate(order) if name == "sample"]
    assert len(samples) == 3, f"one sample before/after sleep and one after wake, got {len(samples)}"
    assert samples[0] < sleep_at < samples[1] < wake_at < samples[2], (
        f"sampling must bracket the real sleep and follow the real wake, order={order}"
    )

    sampled_groups = [event[1] for event in events if event[0] == "sample"]
    assert sampled_groups == [[0, 1], [0, 1], [2, 3]], (
        f"probes must sample every target group, not just the first, got {sampled_groups}"
    )
    assert [event[1] for event in events if event[0] == "sleep_groups"] == [[0, 1]]
    assert [event[1] for event in events if event[0] == "wake_groups"] == [[2, 3]]


def test_schedule_trace_metadata_carries_the_real_sample_payloads():
    """Each probe span must publish the payload its own sampling call returned."""
    import asyncio

    events: list = []
    trainer = _schedule_trainer(events, to_sleep=(0, 1), to_wake=(2, 3))
    asyncio.run(trainer._apply_sleep_pool_schedule({"n": 2}))

    spans = {
        event[1]: event[2]
        for event in events
        if event[0] == "span_enter" and str(event[1]).startswith("memory_probe")
    }
    assert set(spans) == {"memory_probe_sleep", "memory_probe_wake_mapped"}, (
        f"schedule must emit exactly the two resize probe spans, got {sorted(spans)}"
    )

    sleep_meta = spans["memory_probe_sleep"]
    assert sleep_meta["sleep_before"] == {"token": "sample#0", "groups": [0, 1]}
    assert sleep_meta["sleep_after"] == {"token": "sample#1", "groups": [0, 1]}
    assert sleep_meta["sleep_before"] != sleep_meta["sleep_after"], (
        "before/after must be two distinct samples, not the same payload recorded twice"
    )
    assert sleep_meta["config_fingerprint"] == {"fingerprint": "real"}
    assert sleep_meta["groups"] == [0, 1] and sleep_meta["schedule_k"] == 2

    wake_meta = spans["memory_probe_wake_mapped"]
    assert wake_meta["wake_mapped_after"] == {"token": "sample#2", "groups": [2, 3]}
    assert wake_meta["config_fingerprint"] == {"fingerprint": "real"}
    assert wake_meta["groups"] == [2, 3]

    assert sleep_meta["transition_id"] == wake_meta["transition_id"], "both records must share one transition id"
    assert trainer._pending_memory_transition == {
        "transition_id": wake_meta["transition_id"],
        "groups": [2, 3],
        "schedule_k": 2,
    }, "the pending slot must hand the post-sync probe the same id and the woken groups"


def test_schedule_skips_probes_for_empty_sleep_or_wake_sets():
    """No sleep set means no sleep probe; the wake side still pairs normally."""
    import asyncio

    events: list = []
    trainer = _schedule_trainer(events, to_sleep=(), to_wake=(0,))
    asyncio.run(trainer._apply_sleep_pool_schedule({"n": 2}))
    span_names = [event[1] for event in events if event[0] == "span_enter"]
    assert "memory_probe_sleep" not in span_names and "pool_sleep" not in span_names
    assert "memory_probe_wake_mapped" in span_names
    assert trainer._pending_memory_transition is not None

    events.clear()
    trainer = _schedule_trainer(events, to_sleep=(0,), to_wake=())
    asyncio.run(trainer._apply_sleep_pool_schedule({"n": 2}))
    span_names = [event[1] for event in events if event[0] == "span_enter"]
    assert "memory_probe_wake_mapped" not in span_names and "pool_wake" not in span_names
    assert "memory_probe_sleep" in span_names
    assert trainer._pending_memory_transition is None, (
        "with nothing woken there is no post-sync probe to pair, so the slot must stay empty"
    )


def test_transition_id_counter_increments_so_ids_never_collide():
    """A constant counter would silently pair a new resize's probe with an old one."""
    trainer_class = _compile_trainer_class("_transition_id")
    trainer = trainer_class()
    trainer._memory_transition_counter = 0
    trainer._trace_step = lambda: 5

    ids = [trainer._transition_id(2), trainer._transition_id(2), trainer._transition_id(3)]
    assert len(set(ids)) == 3, f"every mint must be unique even for the same k, got {ids}"
    assert ids[0] == "step5_k2_t1" and ids[1] == "step5_k2_t2" and ids[2] == "step5_k3_t3"
    assert trainer._memory_transition_counter == 3, "the counter must persist across mints"

    trainer._trace_step = lambda: 6
    assert trainer._transition_id(2) == "step6_k2_t4", "the id must track the live trace step"


def test_startup_gate_raises_for_either_side_being_incomplete():
    """Executes the real gate body: both awake and sleeping results must be checked."""
    node = _find_trainer_method("_finalize_elastic_pool_replicas")
    gate = _completeness_gate(node)
    loops = [
        statement
        for statement in node.body
        if isinstance(statement, ast.For) and any(part is gate for part in ast.walk(statement))
    ]
    assert len(loops) == 1, "the gate must live in one loop over both labelled results"

    trainer_class = _compile_trainer_class("_memory_sample_invalid", "_memory_samples_complete")
    complete = {
        "samples": [_valid_memory_payload(sleep_level=2)],
        "expected_samples": 1,
        "observed_samples": 1,
        "error_count": 0,
        "sampling_rpc_ms": 1.0,
    }
    incomplete = dict(complete, samples=[_valid_memory_payload(sampling_error="boom")], error_count=1)
    empty = {
        "samples": [],
        "expected_samples": 0,
        "observed_samples": 0,
        "error_count": 0,
        "sampling_rpc_ms": 0.0,
    }

    module = ast.Module(body=[loops[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    run_gate = compile(module, str(_RAY_TRAINER), "exec")

    def _outcome(awake_result, sleeping_result, *, awake=(0, 1), sleeping=(2, 3)):
        pool = type("Pool", (), {"awake_indices": awake, "sleeping_indices": sleeping})()
        namespace = {
            "self": trainer_class,
            "pool": pool,
            "awake_result": awake_result,
            "sleeping_result": sleeping_result,
        }
        try:
            exec(run_gate, namespace)
        except RuntimeError as exc:
            return str(exc)
        return None

    assert _outcome(complete, complete) is None, "a healthy startup must not raise"
    both_empty = _outcome(empty, empty)
    assert both_empty is not None and "awake" in both_empty, (
        f"groups were requested and nothing came back: the gate must raise, got {both_empty!r}"
    )
    awake_broken = _outcome(incomplete, complete)
    sleeping_broken = _outcome(complete, incomplete)
    assert awake_broken is not None and "awake" in awake_broken, (
        f"an incomplete awake result must raise and name it, got {awake_broken!r}"
    )
    assert sleeping_broken is not None and "sleeping" in sleeping_broken, (
        f"an incomplete sleeping result must raise and name it, got {sleeping_broken!r}"
    )
    assert _outcome(complete, empty, sleeping=()) is None and _outcome(empty, complete, awake=()) is None, (
        "a side with no groups to sample must not veto a healthy one"
    )


def _memory_result(*, complete: bool = True) -> dict:
    """A two-rank sampling result the real _memory_samples_complete accepts (or rejects)."""
    healthy = [_valid_memory_payload(sleep_level=2), _valid_memory_payload(sleep_level=2)]
    if complete:
        return {
            "samples": healthy,
            "expected_samples": 2,
            "observed_samples": 2,
            "error_count": 0,
            "sampling_rpc_ms": 1.0,
        }
    return {
        "samples": [healthy[0], _valid_memory_payload(sleep_level=2, sampling_error="rpc died")],
        "expected_samples": 2,
        "observed_samples": 2,
        "error_count": 1,
        "sampling_rpc_ms": 1.0,
    }


class _FinalizePool:
    """The elastic pool as _finalize_elastic_pool_replicas sees it.

    awake and sleeping are multi-group and disjoint on purpose: with one group per
    side a `[:1]`-truncated probe is indistinguishable from a complete one.
    """

    def __init__(self, events: list, *, registered: tuple, awake: tuple, sleeping: tuple) -> None:
        self.events = events
        self.all_registered = registered
        self.awake_indices = awake
        self.sleeping_indices = sleeping
        self.replicas: dict = {}

    @property
    def all_replicas(self) -> tuple:
        return tuple(self.replicas.values())

    def register_replica(self, group_index, replica) -> None:
        self.events.append(("register_replica", group_index))
        self.replicas[group_index] = replica

    def handles(self, group_index):
        raise AssertionError("this fixture registers every group up front; handles() must not be reached")


def _finalize_trainer(events: list, results: list, *, awake: tuple = (0, 1), sleeping: tuple = (2, 3)):
    """A trainer whose real _finalize_elastic_pool_replicas runs against fakes.

    The layout reports every group as initially awake, so missing_indices is empty
    and the probe tail is reached without faking replica launching or asyncio.run.
    Only _sample_pool_memory is faked; the completeness gate runs the real
    _memory_samples_complete compiled from the same source.
    """

    def _span(_config, name, *, step=None, lane=None, metadata=None):
        return _RecordedSpan(events, name, metadata)

    class _Layout:
        num_elastic_groups = 4

        @staticmethod
        def awake_elastic_group_indices(k):
            return (0, 1, 2, 3)

    class _UnusedWorkerGroup:
        @staticmethod
        def from_detached(**kwargs):
            raise AssertionError("no replica may be launched: every group is registered up front")

    trainer_class = _compile_trainer_class(
        "_finalize_elastic_pool_replicas",
        "_memory_sample_invalid",
        "_memory_samples_complete",
        extra_namespace={
            "resize_trace_span": _span,
            "logger": type("Log", (), {"info": staticmethod(lambda *a, **k: None)}),
            "RayWorkerGroup": _UnusedWorkerGroup,
        },
    )
    trainer = trainer_class()
    trainer._elastic_pool = _FinalizePool(events, registered=(0, 1, 2, 3), awake=awake, sleeping=sleeping)
    trainer._sleep_pool_layout = _Layout()
    trainer._sleep_pool_initial_k = 2
    trainer._elastic_prebuild_started_at = 0.25
    trainer._resize_trace_config = None
    trainer.async_rollout_manager = type(
        "Manager", (), {"rollout_replicas": [f"replica{rank}" for rank in range(4)]}
    )()
    trainer._trace_step = lambda: 11
    trainer._update_resize_execution_metrics = lambda **kwargs: events.append(("metrics", dict(kwargs)))
    trainer._get_memory_config_fingerprint = lambda: {"fingerprint": "real"}

    def _sample(group_indices, **kwargs):
        result = results[len([event for event in events if event[0] == "sample"])]
        events.append(("sample", list(group_indices), result))
        return result

    trainer._sample_pool_memory = _sample
    return trainer


def test_startup_probe_runtime_samples_both_group_sets_and_publishes_them():
    """Runtime proof that the startup baseline executes, not merely that it parses.

    Every other guard on this method is an AST position check, so an inserted early
    return or a never-true guard in front of the probe tail leaves them all green
    while the startup baseline silently becomes dead code.
    """
    events: list = []
    awake_result = _memory_result()
    sleeping_result = _memory_result()
    trainer = _finalize_trainer(events, [awake_result, sleeping_result])

    trainer._finalize_elastic_pool_replicas()

    sampled_groups = [event[1] for event in events if event[0] == "sample"]
    assert sampled_groups == [[0, 1], [2, 3]], (
        f"startup must sample every awake group and every sleeping group, got {sampled_groups}"
    )

    entered = [event for event in events if event[0] == "span_enter"]
    probe_spans = [event for event in entered if event[1] == "memory_probe_startup"]
    assert len(probe_spans) == 1, (
        f"startup must emit exactly one memory_probe_startup span, got {[event[1] for event in entered]}"
    )
    metadata = probe_spans[0][2]
    assert metadata["awake"] is awake_result and metadata["sleeping"] is sleeping_result, (
        "the span must publish the payloads its own two sampling calls returned, not substitutes"
    )
    assert metadata["awake_groups"] == [0, 1] and metadata["sleeping_groups"] == [2, 3]
    assert metadata["initial_k"] == 2 and metadata["phase"] == "startup_baseline"
    assert metadata["config_fingerprint"] == {"fingerprint": "real"}
    assert ("span_exit", "memory_probe_startup") in events, "the span must close"

    assert [event for event in events if event[0] == "metrics"] == [("metrics", {"pool/prebuild_s": 0.75})], (
        "finalize must still report the real prebuild cost it accounted before probing"
    )
    order = [event[0] for event in events]
    probe_span_at = next(index for index, event in enumerate(events) if event is probe_spans[0])
    assert order.index("metrics") < order.index("sample") < probe_span_at, (
        f"the probe must run after the prebuild accounting and before the memory_probe_startup span opens, "
        f"order={order}"
    )


def test_startup_probe_runtime_gate_raises_only_on_an_incomplete_side():
    """Runtime proof that the fail-fast gate really executes, for both sides."""

    def _run(awake_result, sleeping_result):
        events: list = []
        trainer = _finalize_trainer(events, [awake_result, sleeping_result])
        try:
            trainer._finalize_elastic_pool_replicas()
        except RuntimeError as exc:
            return events, str(exc)
        return events, None

    healthy_events, healthy_error = _run(_memory_result(), _memory_result())
    assert healthy_error is None, f"a complete startup baseline must not raise, got {healthy_error!r}"
    assert [event[0] for event in healthy_events].count("sample") == 2, (
        "the healthy path must still have really sampled both group sets"
    )

    awake_events, awake_error = _run(_memory_result(complete=False), _memory_result())
    assert awake_error is not None and "awake" in awake_error and "sleeping" not in awake_error, (
        f"an incomplete awake result must raise RuntimeError naming the awake side, got {awake_error!r}"
    )
    assert any(event[0] == "span_enter" for event in awake_events), (
        "the trace must record what the probe saw before the gate fails the run"
    )

    _, sleeping_error = _run(_memory_result(), _memory_result(complete=False))
    assert sleeping_error is not None and "sleeping" in sleeping_error, (
        f"an incomplete sleeping result must raise RuntimeError naming the sleeping side, got {sleeping_error!r}"
    )


def _empty_result() -> dict:
    """What a probe returns when the RPC path is broken end to end."""
    return {
        "samples": [],
        "expected_samples": 0,
        "observed_samples": 0,
        "error_count": 0,
        "sampling_rpc_ms": 0.0,
    }


def test_startup_gate_raises_when_a_requested_side_produced_an_empty_payload():
    """The empty-payload signature must be loud, not silent.

    The guard used to read `if result["expected_samples"] and not complete(...)`.
    A probe that reached no worker at all reports expected_samples == 0, so the
    guard short-circuited and a run with zero memory data looked successful.
    """
    events: list = []
    trainer = _finalize_trainer(events, [_empty_result(), _memory_result()])

    try:
        trainer._finalize_elastic_pool_replicas()
    except RuntimeError as exc:
        error = str(exc)
    else:
        error = None

    assert error is not None, "groups were requested and nothing came back; startup must fail fast"
    assert "awake" in error, error
    assert any(event[0] == "span_enter" for event in events), (
        "the empty payload must still reach the trace before the run dies"
    )


def test_startup_gate_stays_silent_when_a_side_has_no_groups_to_sample():
    """No false-fail: k0 == num_groups means nothing sleeps, so an empty sleeping
    payload is the correct answer rather than a broken probe."""
    events: list = []
    trainer = _finalize_trainer(events, [_memory_result(), _empty_result()], awake=(0, 1, 2, 3), sleeping=())

    trainer._finalize_elastic_pool_replicas()

    sampled = [event[1] for event in events if event[0] == "sample"]
    assert sampled == [[0, 1, 2, 3], []], f"both sides must still be sampled, got {sampled}"
