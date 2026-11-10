"""Declared candidate imports and harness-independent Python worker startup.

These checks enforce submission rules for ordinary Python code, not a security
sandbox against code deliberately modifying the interpreter or loading binaries.
"""

import ast
import builtins
import functools
import importlib
import importlib.util
import pathlib
import sys


class DependencyViolation(ValueError):
    code = "dependency_violation"

    def __init__(self, task, module):
        self.module = module
        super().__init__(f"dependency_violation: {module} is not allowed for {task}")


class GradingEnvironmentError(RuntimeError):
    code = "environment_error"


def declared_dependencies(task_file):
    """Read literal module metadata without importing the task or candidate."""
    source = pathlib.Path(task_file)
    tree = ast.parse(source.read_text(), filename=str(source))
    values = [node.value for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "ALLOWED_EXTERNAL_IMPORTS"
                      for target in node.targets)]
    if not values:
        return ()
    if len(values) != 1:
        raise ValueError(f"{source.parent.name}: duplicate ALLOWED_EXTERNAL_IMPORTS")
    try:
        roots = ast.literal_eval(values[0])
    except (ValueError, TypeError):
        raise ValueError(f"{source.parent.name}: ALLOWED_EXTERNAL_IMPORTS must be a literal tuple") from None
    if (type(roots) is not tuple
            or any(type(root) is not str or not root.isidentifier() for root in roots)
            or len(set(roots)) != len(roots)):
        raise ValueError(f"{source.parent.name}: ALLOWED_EXTERNAL_IMPORTS must contain distinct module roots")
    return roots


# A speedup must come from the algorithm and its implementation, not from the
# host's core count or a compiled helper. Threads, worker processes, child
# processes and foreign-function loading are therefore outside every task's
# allowance, although they are standard library. This names the ordinary routes;
# os.fork and similar calls are excluded by the written rules, not detected.
DENIED_STDLIB_ROOTS = frozenset({
    "_ctypes", "_multiprocessing", "_posixsubprocess", "_thread", "concurrent", "ctypes",
    "multiprocessing", "subprocess", "threading",
})


class CandidateImports:
    def __init__(self, task_file):
        self.path = pathlib.Path(task_file).resolve().with_name("candidate.py")
        self.task = self.path.parent.name
        self.external = declared_dependencies(task_file)
        self.allowed = ((sys.stdlib_module_names | set(self.external) | {"__future__"})
                        - DENIED_STDLIB_ROOTS)
        self._stack = []
        self.violation = None

    def check(self, name, level=0):
        root = name.split(".", 1)[0]
        if level or root not in self.allowed:
            self.violation = DependencyViolation(self.task, name or "relative import")
            raise self.violation

    def validate_source(self):
        tree = ast.parse(self.path.read_text(), filename=str(self.path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.check(alias.name)
            elif isinstance(node, ast.ImportFrom):
                self.check(node.module or "", node.level)
        for name in self.external:
            if importlib.util.find_spec(name) is None:
                raise GradingEnvironmentError(
                    f"environment_error: {self.task} requires declared package {name}")

    def __enter__(self):
        if not self._stack:
            self.violation = None
        original_import = builtins.__import__
        original_import_module = importlib.import_module
        self._stack.append((original_import, original_import_module))
        filename = str(self.path)

        def watched(name, globals=None, locals=None, fromlist=(), level=0):
            # Only candidate requests are constrained. An allowed numerical
            # library retains its own runtime dependency closure.
            if sys._getframe(1).f_code.co_filename == filename:
                self.check(name, level)
            return original_import(name, globals, locals, fromlist, level)

        def import_module(name, package=None):
            if sys._getframe(1).f_code.co_filename == filename:
                self.check(name, int(name.startswith(".")))
            return original_import_module(name, package)

        builtins.__import__ = watched
        importlib.import_module = import_module
        return self

    def __exit__(self, *_exc):
        builtins.__import__, importlib.import_module = self._stack.pop()
        if _exc[0] is None and self.violation is not None:
            raise self.violation


def checked_candidate(task_file, loader):
    policy = CandidateImports(task_file)
    policy.validate_source()
    with policy:
        module = loader()
    solve = module.solve

    @functools.wraps(solve)
    def checked_solve(*args, **kwargs):
        with policy:
            return solve(*args, **kwargs)

    module.solve = checked_solve
    return module


def grading_command(arguments, root):
    """Ignore harness Python paths, user site packages, and the working directory."""
    bootstrap = ("import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
                 "runpy.run_module('speedupmark', run_name='__main__')")
    return [sys.executable, "-I", "-B", "-c", bootstrap, str(pathlib.Path(root).resolve()),
            *map(str, arguments)]
