"""SPEEDUP-MARK task interface — mirrors AlgoTune: generate / solve / verify.

Each ``tasks/<name>/task_spec.py`` defines inputs, correctness, and any
deterministic metric. The sibling ``reference.py`` supplies the baseline and
submission starter. This module contains the shared protocol and helpers.
"""

import dis
import importlib.util
import pathlib
import re
import sys
from dataclasses import dataclass
from types import CodeType, ModuleType
from typing import Any, Protocol


_TASK_VERSION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


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


def load_candidate(task_file: str) -> ModuleType:
    """Load the submission supplied to the grader for this task."""
    return _load_solver(task_file, "candidate.py")


@dataclass(frozen=True)
class SolutionEvaluation:
    """A deterministic task-specific score and its correctness result."""

    score: float
    correct: bool


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

    Managed development and final grading both measure ``grading_cases``.
    That attribute is a non-empty sequence of positive problem sizes. The run
    controller does not add sizes of its own.

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


def plain_containers(value: Any) -> bool:
    """Reject sequence subclasses before untimed verification can iterate them.

    Task verifiers still decide which scalar leaf types and values are valid.
    """
    if isinstance(value, (tuple, list)):
        return type(value) in (tuple, list) and all(plain_containers(item) for item in value)
    return True


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


def _code_imports(code, roots, found):
    """Find actual import instructions, including in unused nested functions.

    A function-local ``import re`` binds nothing at module level, so neither the
    namespace scan nor a ``sys.modules`` delta sees it when the module is already
    imported at interpreter start (``re`` always is). Inspect IMPORT_NAME rather
    than arbitrary names or string literals, which also occur in innocent code.
    """
    for instruction in dis.get_instructions(code):
        if instruction.opname == "IMPORT_NAME":
            root = _root_of(instruction.argval, roots)
            if root is not None:
                found.add(root)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            _code_imports(constant, roots, found)


def _root_of(name: str, roots: tuple[str, ...]):
    for root in roots:
        if name == root or name.startswith(f"{root}."):
            return root
    return None


def forbidden_imports(module: ModuleType, roots: tuple[str, ...],
                      imported_since=None, imported_during=None) -> tuple[str, ...]:
    """Forbidden module roots reachable from a candidate module.

    Some tasks ask the candidate to build the engine the task is about, so
    handing the whole job to a library that has that engine inside it (an
    embedded SQL engine, a regex engine) is a contract violation rather than an
    optimization.

    Four signals are combined, because each one alone has a hole:

    * the module attributes the candidate bound at import time;
    * import instructions inside the candidate's code objects, which catch a
      function-local import of a module that is already loaded;
    * ``imported_during``, the roots ``watch_imports`` observed being imported
      while the candidate ran;
    * ``imported_since``, the ``sys.modules`` delta since a snapshot: ``from
      sqlite3 import connect`` binds a function whose ``__module__`` is the C
      extension ``_sqlite3``, so a name scan alone would miss it.

    This is a good-faith policy check for broadly honest submissions, not a
    sandbox: a candidate that smuggles an engine in through ``ctypes`` or a
    vendored copy of one is out of scope for it.
    """
    found = set()
    for value in vars(module).values():
        name = getattr(value, "__module__", None)
        if not isinstance(name, str):
            name = getattr(value, "__name__", None)
        if isinstance(name, str):
            root = _root_of(name, roots)
            if root is not None:
                found.add(root)
        code = getattr(value, "__code__", None)
        if code is not None:
            _code_imports(code, roots, found)
    if imported_during is not None:
        for name in imported_during:
            root = _root_of(name, roots)
            if root is not None:
                found.add(root)
    if imported_since is not None:
        for name in set(sys.modules) - set(imported_since):
            root = _root_of(name, roots)
            if root is not None:
                found.add(root)
    return tuple(sorted(found))


class watch_imports:
    """Record which forbidden roots a block imports, even preloaded ones.

    The ``sys.modules`` delta cannot see ``import re`` inside a candidate
    function, because ``re`` is already imported before the interpreter starts
    running the task. Wrapping ``builtins.__import__`` for the duration of the
    candidate call sees every import statement the candidate executes.
    """

    def __init__(self, roots: tuple[str, ...]) -> None:
        self.roots = tuple(roots)
        self.used: set[str] = set()
        self._original = None

    def __enter__(self):
        import builtins

        self._original = builtins.__import__
        watched_roots, used = self.roots, self.used
        original = self._original

        def watched(name, *args, **kwargs):
            root = _root_of(name, watched_roots)
            if root is not None:
                used.add(root)
            return original(name, *args, **kwargs)

        builtins.__import__ = watched
        return self.used

    def __exit__(self, *_exc):
        import builtins

        if self._original is not None:
            builtins.__import__ = self._original
        return False
