"""Dependency contracts hold with missing, installed, and already loaded packages."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from speedupmark.dependencies import (DependencyViolation, GradingEnvironmentError,
                                     declared_dependencies, grading_command)
from speedupmark.harness import _run_isolated
from speedupmark.task import load_candidate


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DependencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = pathlib.Path(temporary.name)
        self.spec = self.path / 'task_spec.py'
        self.spec.write_text('raise AssertionError("metadata must not execute")\n')

    def candidate(self, source):
        (self.path / 'candidate.py').write_text(source)
        return load_candidate(str(self.spec))

    def test_static_imports_fail_before_candidate_execution(self):
        for source in (
            'import numpy as np\n',
            'from numpy.linalg import solve\n',
            'def unused():\n    import numpy\n',
            'from . import helper\n',
        ):
            with self.subTest(source=source):
                with self.assertRaises(DependencyViolation):
                    self.candidate('raise AssertionError("candidate executed")\n' + source)

    def test_static_rejection_is_identical_with_preloaded_package(self):
        for modules in ({}, {'numpy': types.ModuleType('numpy')}):
            with self.subTest(preloaded=bool(modules)), patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(DependencyViolation, 'numpy is not allowed'):
                    self.candidate('import numpy\ndef solve(problem): return problem\n')

    def test_dynamic_imports_reject_preloaded_packages_at_load_and_call(self):
        expressions = ('__import__(name)', 'importlib.import_module(name)')
        fake = types.ModuleType('external_probe')
        for expression in expressions:
            for at_load in (False, True):
                body = f'import importlib\nname = "external_" + "probe"\n'
                body += (f'value = {expression}\n' if at_load else '')
                body += f'def solve(problem):\n    return {expression}\n'
                with self.subTest(expression=expression, at_load=at_load), patch.dict(
                        sys.modules, external_probe=fake):
                    with self.assertRaisesRegex(DependencyViolation, 'external_probe is not allowed'):
                        module = self.candidate(body)
                        module.solve(None)

    def test_standard_library_and_recursive_calls_work_and_restore_hooks(self):
        original_import = __import__
        module = self.candidate('import math, importlib\n'
                                'def solve(n):\n'
                                '    if n: return solve(n-1)\n'
                                '    return importlib.import_module("math").isqrt(16)\n')
        self.assertEqual(module.solve(3), 4)
        self.assertIs(__import__, original_import)

    def test_catching_dynamic_import_failure_does_not_hide_policy_violation(self):
        module = self.candidate('def solve(problem):\n'
                                '    try: __import__("external_" + "probe")\n'
                                '    except Exception: return 42\n')
        with self.assertRaises(DependencyViolation):
            module.solve(None)

    def test_declared_package_works_and_missing_prerequisite_is_environment_error(self):
        self.spec.write_text('ALLOWED_EXTERNAL_IMPORTS = ("external_probe",)\n'
                             'raise AssertionError("metadata executed")\n')
        package = types.ModuleType('external_probe')
        package.__spec__ = importlib.util.spec_from_loader('external_probe', loader=None)
        package.answer = 42
        with patch.dict(sys.modules, external_probe=package):
            module = self.candidate('import external_probe\n'
                                    'def solve(problem): return external_probe.answer\n')
            self.assertEqual(module.solve(None), 42)
        with self.assertRaisesRegex(GradingEnvironmentError, 'requires declared package external_probe'):
            self.candidate('def solve(problem): return problem\n')

    def test_allowed_package_keeps_its_transitive_dependencies(self):
        self.spec.write_text('ALLOWED_EXTERNAL_IMPORTS = ("external_probe",)\n')
        (self.path / 'external_probe.py').write_text('import transitive_probe\nanswer = transitive_probe.answer\n')
        (self.path / 'transitive_probe.py').write_text('answer = 42\n')
        with patch.object(sys, 'path', [str(self.path), *sys.path]):
            try:
                module = self.candidate('import external_probe\n'
                                        'def solve(problem): return external_probe.answer\n')
                self.assertEqual(module.solve(None), 42)
                with self.assertRaises(DependencyViolation):
                    self.candidate('import transitive_probe\ndef solve(problem): return 42\n')
            finally:
                sys.modules.pop('external_probe', None)
                sys.modules.pop('transitive_probe', None)

    def test_optional_task_allowlists_match_documented_extras(self):
        titles = {'numpy': 'NumPy', 'scipy': 'SciPy', 'cvxpy': 'CVXPY', 'clarabel': 'Clarabel'}
        for spec in (ROOT / 'tasks').glob('*/task_spec.py'):
            readme = spec.with_name('README.md').read_text()
            description = next((line for line in readme.splitlines()
                                if 'optional numerical extra' in line), '')
            expected = tuple(name for name, title in titles.items() if title in description)
            with self.subTest(task=spec.parent.name):
                self.assertEqual(set(declared_dependencies(spec)), set(expected))

    def test_worker_ignores_pythonpath_and_returns_structured_dependency_failure(self):
        (self.path / 'speedupmark.py').write_text('raise AssertionError("harness shadowed grader")\n')
        (self.path / 'numpy.py').write_text('raise AssertionError("external package imported")\n')
        self.spec.write_text('from speedupmark.task import load_reference\n'
                            '_reference = load_reference(__file__)\n'
                            'class Task:\n'
                            '    name = "probe"\n'
                            '    task_version = "1.0.0"\n'
                            '    default_n = 1\n'
                            '    def generate_problem(self,n,random_seed): return 1\n'
                            '    solve = staticmethod(_reference.solve)\n'
                            '    def is_solution(self,p,a): return a == p\n'
                            'TASK = Task()\n')
        (self.path / 'reference.py').write_text('def solve(p): return p\n')
        (self.path / 'candidate.py').write_text('import numpy\ndef solve(p): return p\n')
        env = dict(os.environ, PYTHONPATH=str(self.path), PYTHONUSERBASE=str(self.path))
        result = subprocess.run(grading_command([self.path, '--json', '--samples', '1'], ROOT),
                                env=env, cwd=self.path, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['errors'][0]['code'], 'dependency_violation')
        self.assertNotIn('Traceback', result.stderr)
        with patch.dict(os.environ, PYTHONPATH=str(self.path)):
            with self.assertRaises(DependencyViolation):
                _run_isolated(self.path, 1, 0, 1, 10)
