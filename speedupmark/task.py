"""SPEEDUP-MARK task interface — mirrors AlgoTune: generate / solve / verify.

Each ``tasks/<name>/task_spec.py`` defines inputs, correctness, and any
deterministic metric. The sibling ``reference.py`` supplies the baseline and
submission starter. This module contains the shared protocol and helpers.
"""

import ast
import hashlib
import importlib.util
import json
import pathlib
import re
import secrets
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

_TASK_VERSION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class TaskSpecMetadata:
    name: str | None = None
    task_version: str | None = None
    grading_cases: tuple[int, ...] | list[int] | None = None
    family_period: int = 1
    timing_batch: int = 1
    port_todo: bool = False
    external_imports: tuple[str, ...] = ()


def _assignments(statements):
    for statement in statements:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    yield target.id, statement.value


def read_task_metadata(task_file) -> TaskSpecMetadata:
    """Read literal spec metadata without importing the task or its dependencies."""
    path = pathlib.Path(task_file)
    tree = ast.parse(path.read_text(), filename=str(path))
    module_fields = list(_assignments(tree.body))
    dependencies = [value for name, value in module_fields if name == "ALLOWED_EXTERNAL_IMPORTS"]
    if len(dependencies) > 1:
        raise ValueError(f"{path.parent.name}: duplicate ALLOWED_EXTERNAL_IMPORTS")

    def literal(name, value):
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            raise ValueError(f"{path.parent.name}: {name} must be literal") from None

    roots = literal("ALLOWED_EXTERNAL_IMPORTS", dependencies[0]) if dependencies else ()
    if (type(roots) is not tuple
            or any(type(root) is not str or not root.isidentifier() for root in roots)
            or len(set(roots)) != len(roots)):
        raise ValueError(f"{path.parent.name}: ALLOWED_EXTERNAL_IMPORTS must contain distinct module roots")

    port_todo = False
    attributes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is True:
            port_todo |= any(name == "PORT_TODO" for name, _ in _assignments((node,)))
        if isinstance(node, ast.ClassDef) and not attributes:
            fields = dict(_assignments(node.body))
            if "name" in fields or "task_version" in fields:
                attributes = {name: literal(name, fields[name])
                              for name in ("name", "task_version", "grading_cases",
                                           "family_period", "timing_batch") if name in fields}
    return TaskSpecMetadata(**attributes, port_todo=port_todo, external_imports=roots)


def declared_task_version(task: Any) -> str:
    """Return a task's required, canonical ``MAJOR.MINOR.PATCH`` version."""
    version = getattr(task, "task_version", None)
    if not isinstance(version, str) or not _TASK_VERSION_PATTERN.fullmatch(version):
        name = getattr(task, "name", task)
        raise ValueError(f"{name}: task_version must be MAJOR.MINOR.PATCH")
    return version


def _load_solver(task_file: str, filename: str) -> ModuleType:
    """Load a task-local solver without changing sys.path or global imports."""
    path = pathlib.Path(task_file).resolve().with_name(filename)
    name = f"speedupmark_{path.stem}_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load solver from {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and similar introspection require a registered module while
    # the file executes. A failed import must not leave a half-loaded solver.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_reference(task_file: str) -> ModuleType:
    """Load the authoritative baseline used by a task's specification."""
    return _load_solver(task_file, "reference.py")


def load_candidate(task_file: str, *, forbidden_roots=()) -> ModuleType:
    """Load the submission supplied to the grader for this task."""
    from speedupmark.dependencies import checked_candidate

    return checked_candidate(task_file, lambda: _load_solver(task_file, "candidate.py"),
                             forbidden_roots)


@dataclass(frozen=True)
class SolutionEvaluation:
    """A deterministic task-specific score and its correctness result."""

    score: float
    correct: bool


def verification_challenge(problem, programs, *, generator, trials, runtime_input,
                           digest_key, hash_missing_programs=False, replay=None, record=None):
    """Share one private challenge across programs that have already been frozen."""
    def digest(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    receipt = {
        "generator_version": generator,
        "problem_sha256": digest(problem),
        "program_sha256": {role: digest(program) if program is not None or hash_missing_programs else None
                           for role, program in programs.items()},
    }
    if replay is None:
        seed = secrets.token_hex(32)
    else:
        if type(replay) is not dict or set(replay) != set(receipt) | {"seed", digest_key}:
            raise ValueError("invalid verification receipt")
        if any(replay[key] != value for key, value in receipt.items()):
            raise ValueError("replay requires the same problem, frozen programs and generator")
        seed = replay["seed"]
        if (type(seed) is not str or len(seed) != 64 or
                any(character not in "0123456789abcdef" for character in seed)):
            raise ValueError("verification seed must be 256-bit lowercase hexadecimal")
    inputs = tuple(tuple(runtime_input(workload, seed, index, trial) for trial in range(trials))
                   for index, workload in enumerate(problem["workloads"]))
    receipt["seed"], receipt[digest_key] = seed, digest(inputs)
    if replay is not None and replay[digest_key] != receipt[digest_key]:
        raise ValueError("replay runtime input digest does not match")
    # Keep the challenge available even if later simulation rejects a program.
    if record is not None:
        record(receipt)
    return inputs


class SpeedupMarkTask(Protocol):
    """Required task surface.

    Tasks with a deterministic cost model may also expose ``metric_unit`` and
    ``evaluate_solution(problem, proposed) -> SolutionEvaluation``. The harness
    otherwise measures reference and candidate ``solve(problem)`` calls in
    wall-clock milliseconds. The grader supplies the candidate entrypoint;
    references are self-contained and copied verbatim into new run candidates.

    A task needing a shared private correctness challenge can implement
    ``evaluate_pair(problem, outputs, *, replay=None, record=None)`` instead.
    It returns reference/candidate evaluations and records replay context before
    execution; that context must never be supplied to candidate compilation.

    A canonical-output task can declare ``uses_reference_output = True`` and
    accept ``is_solution(problem, proposed, *, reference_output=None)``. The
    harness supplies the already-measured, frozen reference answer as a detached
    copy after both timers stop. Reference correctness is independently tested;
    this comparison does not independently solve each graded input.

    A generator selecting workload families from the seed declares
    ``family_period``; the harness rounds the sample count up to a multiple of
    it. A task whose reference call is too short to time once declares
    ``timing_batch``, the number of same-family problems summed per sample.

    Managed development and final grading both measure ``grading_cases``.
    That attribute is a non-empty sequence of positive problem sizes. The run
    controller does not add sizes of its own.

    A task can supply ``correctness_problem(seed)`` for a small legal input that
    its ``is_solution`` verifies independently. Each sample also has to pass this
    check. It runs before the timed problems and can warm both solvers.

    Outputs must be completed data accepted by ``freeze_output``. Both solvers'
    outputs are detached before the timer stops and before verification begins.
    """

    name: str
    display_name: str
    task_version: str

    def generate_problem(self, n: int, random_seed: int = 0) -> Any: ...
    def solve(self, problem: Any) -> Any: ...
    def is_solution(self, problem: Any, proposed: Any) -> bool: ...


def grading_cases(task: Any) -> tuple[int, ...]:
    """Problem sizes declared by a task spec for managed grading.

    Development and final grading both use this sequence. A missing or invalid
    declaration fails the evaluation; the controller does not invent a second size.
    """
    declared = getattr(task, "grading_cases", None)
    name = getattr(task, "name", task)
    if isinstance(declared, (str, bytes)) or not isinstance(declared, (tuple, list)) or not declared:
        raise ValueError(f"{name}: grading_cases must be a non-empty sequence of positive integers")
    sizes = tuple(declared)
    # bool is an int subclass, and a True size would be a one-element accident.
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError(f"{name}: grading_cases must be a non-empty sequence of positive integers")
    return sizes


def _positive_count(task: Any, attribute: str) -> int:
    value = getattr(task, attribute, 1)
    if type(value) is not int or value < 1:
        raise ValueError(f"{getattr(task, 'name', task)}: {attribute} must be a positive integer")
    return value


def family_period(task: Any) -> int:
    """Consecutive seeds after which a generator's seed-selected families repeat.

    Generators that pick a workload family from ``seed % k`` declare the least
    common multiple of every such ``k``. A spec that puts every family into
    every problem declares nothing and has period one.
    """
    return _positive_count(task, "family_period")


def timing_batch(task: Any) -> int:
    """Independent problems measured together as one sample of a wall-clock task.

    A reference call lasting microseconds cannot be timed once: the ratio is
    then timer noise and fixed call overhead. Such a task declares how many
    distinct same-family problems one sample sums, instead of repeating one
    problem, which a candidate could answer from a cache.
    """
    return _positive_count(task, "timing_batch")


def covering_samples(task: Any, requested: int) -> int:
    """Round a requested sample count up so every seed-selected family is graded.

    Samples use consecutive seeds, so a whole number of periods measures each
    family equally often; fewer would drop a family chosen by the random base.
    """
    period = family_period(task)
    return -(-requested // period) * period


def plain_containers(value: Any) -> bool:
    """Reject sequence subclasses before untimed verification can iterate them.

    Task verifiers still decide which scalar leaf types and values are valid.
    """
    if isinstance(value, (tuple, list)):
        return type(value) in (tuple, list) and all(plain_containers(item) for item in value)
    return True


def same_plain_structure(actual: Any, expected: Any) -> bool:
    """Compare exact integer or None leaves in plain sequence trees."""
    expected_type, actual_type = type(expected), type(actual)
    if expected_type is tuple or expected_type is list:
        return ((actual_type is tuple or actual_type is list) and len(actual) == len(expected)
                and all(same_plain_structure(item, value)
                        for item, value in zip(actual, expected)))
    return (actual is None and expected is None
            or (actual_type is int and expected_type is int and actual == expected))


def plain_numeric(value: Any) -> bool:
    """Check completed real numeric data before NumPy coercion can call hooks.

    Numerical task verifiers use this before asarray, then check shape and
    finiteness. Exact numeric ndarrays and built-in list/tuple trees are valid.
    """
    kind = type(value)
    if kind is int or kind is float:
        return True
    if kind is list or kind is tuple:
        return all(plain_numeric(item) for item in value)
    numpy = sys.modules.get("numpy")
    if numpy is not None and kind is numpy.ndarray:
        return value.dtype.kind in "iuf" and not value.dtype.hasobject
    return False


def freeze_output(value: Any) -> Any:
    """Detach completed output data without invoking candidate-defined methods.

    Exact built-in containers preserve their representation; custom objects,
    subclasses, iterators and cycles are rejected. Already-loaded NumPy may
    supply exact numeric scalars and ndarrays; arrays are copied and made
    read-only. Never use deepcopy, pickle or array coercion here: their hooks
    can execute a candidate's deferred computation. The harness charges this
    copy to both solvers before stopping their timers.
    """
    numpy = sys.modules.get("numpy")
    array_type = getattr(numpy, "ndarray", None) if numpy is not None else None
    numeric_types = (() if numpy is None else (
        numpy.bool_, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
        numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
        numpy.float16, numpy.float32, numpy.float64, numpy.longdouble,
        numpy.complex64, numpy.complex128, numpy.clongdouble,
    ))
    active, copied = set(), {}

    def freeze(item):
        kind = type(item)
        if (item is None or kind is bool or kind is int or kind is float
                or kind is str or kind is bytes):
            return item
        if any(kind is scalar for scalar in numeric_types):
            return item
        identity = id(item)
        if identity in active:
            raise ValueError("output must not contain cycles")
        if identity in copied:
            return copied[identity]
        if kind is bytearray:
            result = bytearray(item)
        elif kind is array_type:
            if item.dtype.kind not in "biufc" or item.dtype.hasobject:
                raise ValueError("output arrays must contain numeric data")
            result = array_type.copy(item, order="K")
            result.setflags(write=False)
        elif kind is list or kind is tuple or kind is dict:
            active.add(identity)
            try:
                if kind is dict:
                    result = {freeze(key): freeze(value) for key, value in item.items()}
                else:
                    values = [freeze(value) for value in item]
                    result = values if kind is list else tuple(values)
            finally:
                active.remove(identity)
        else:
            # Do not format item or its type: even error reporting must not
            # call candidate-controlled repr or metaclass methods.
            raise ValueError("output must contain only completed plain data")
        copied[identity] = result
        return result

    try:
        return freeze(value)
    except RecursionError:
        raise ValueError("output is too deeply nested") from None
