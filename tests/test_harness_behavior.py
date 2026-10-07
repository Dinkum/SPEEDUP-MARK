"""Regression tests for measurement integrity and the public runner."""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from speedupmark.harness import CandidateNotReady, _run_isolated, discover_tasks, load_task, resolve_seed, run_task
from speedupmark.suites import SMOKE_TASKS
from speedupmark.task import freeze_output, load_candidate


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MeasurementTests(unittest.TestCase):
    def test_fresh_seeds_and_explicit_replay(self):
        path = self.submission(ROOT / 'examples/example_gzip')
        with patch('speedupmark.harness.secrets.randbits', side_effect=[123, 456]) as entropy:
            first = run_task(path, samples=3)
            second = run_task(path, samples=3)
            replay = run_task(path, seed=123, samples=3)
        self.assertEqual(entropy.call_count, 2)
        self.assertEqual([s.seed for s in first.samples], [123, 124, 125])
        self.assertEqual([s.seed for s in second.samples], [456, 457, 458])
        self.assertEqual([s.seed for s in replay.samples], [123, 124, 125])
        self.assertTrue(first.correct and second.correct and replay.correct)
        self.assertEqual(resolve_seed(0), 0)
        for bad in (-1, True, 1.5, '1'):
            with self.assertRaises(ValueError):
                resolve_seed(bad)

    def test_cli_records_random_seed_and_accepts_replay(self):
        path = self.submission(ROOT / 'examples/example_gzip')
        command = [sys.executable, '-B', '-m', 'speedupmark', str(path), '--json']
        first = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
        seed = first['config']['seed']
        self.assertEqual(first['config']['seed_source'], 'random')
        self.assertIs(type(seed), int)
        replay = json.loads(subprocess.check_output(command + ['--seed', str(seed)], cwd=ROOT, text=True))
        self.assertEqual(replay['config']['seed_source'], 'explicit')
        self.assertEqual([s['seed'] for s in first['results'][0]['samples']],
                         [s['seed'] for s in replay['results'][0]['samples']])
        self.assertTrue(first['correct'] and replay['correct'])

    def task(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = pathlib.Path(directory.name)
        (path / "task_spec.py").write_text(textwrap.dedent(source))
        # Synthetic tasks retain stateful hooks for measurement-integrity tests.
        # Import the real candidate file and capture the hook before loader wiring.
        (path / "candidate.py").write_text(
            'import sys\n'
            f'_task = sys.modules["speedupmark_task_{path.name}"].TASK\n'
            '_solve = getattr(_task, "candidate_solve", _task.solve)\n'
            'def solve(problem): return _solve(problem)\n')
        return path

    def submission(self, source):
        path = self.task((source / "task_spec.py").read_text())
        shutil.copyfile(source / "reference.py", path / "reference.py")
        (path / "candidate.py").write_bytes(
            (source / "reference.py").read_bytes() + b'\n# Test submission.\n')
        return path

    def test_missing_and_unchanged_candidates_stop_before_task_import(self):
        path = self.task('raise AssertionError("task imported")')
        reference = 'raise AssertionError("reference imported")\n'
        (path / "reference.py").write_text(reference)
        (path / "candidate.py").rename(path / "saved-candidate.py")
        for code in ("missing_candidate", "unchanged_candidate"):
            if code == "unchanged_candidate":
                (path / "candidate.py").write_text(reference)
            with self.subTest(code=code), patch("speedupmark.harness.load_task") as load:
                with self.assertRaises(CandidateNotReady) as raised:
                    run_task(path)
                self.assertEqual(raised.exception.code, code)
                load.assert_not_called()
            with patch("speedupmark.harness.subprocess.run") as worker:
                with self.assertRaises(CandidateNotReady):
                    _run_isolated(path, None, 0, 1, 5)
                worker.assert_not_called()
            command = [sys.executable, "-m", "speedupmark", str(path)]
            text = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(text.returncode, 1)
            self.assertIn("INFO" if code == "unchanged_candidate" else "ERROR", text.stdout)
            self.assertNotIn("Traceback", text.stderr)
            report = subprocess.run(command + ["--json"], cwd=ROOT,
                                    capture_output=True, text=True)
            payload = json.loads(report.stdout)
            self.assertEqual(payload["results"], [])
            self.assertEqual(payload["errors"][0]["code"], code)
            self.assertFalse(payload["correct"])
            self.assertIsNone(payload["geomean_speedup"])

    def test_each_timed_output_is_checked_on_a_fresh_seed(self):
        path = self.task('''
            class Task:
                name = "bad_second_call"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return random_seed
                def solve(self, problem): return problem
                def candidate_solve(self, problem): return 8
                def is_solution(self, problem, proposed): return proposed == problem
            TASK = Task()
        ''')
        result = run_task(path, seed=8, samples=3)
        self.assertEqual([sample.seed for sample in result.samples], [8, 9, 10])
        self.assertEqual([sample.correct for sample in result.samples], [True, False, False])
        self.assertFalse(result.correct)
        self.assertEqual(result.speedup, 0)

    def test_mutation_is_isolated_from_peer_and_verifier(self):
        path = self.task('''
            class Task:
                name = "mutation"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return [random_seed]
                def solve(self, problem): return problem.pop()
                def candidate_solve(self, problem): return problem.pop()
                def is_solution(self, problem, proposed): return problem == [proposed]
            TASK = Task()
        ''')
        self.assertTrue(run_task(path).correct)

    def test_canonical_tasks_reuse_each_measured_reference_answer(self):
        for name in ("articulation_points", "grouped_analytics_reports"):
            with self.subTest(task=name):
                path = self.submission(ROOT / "tasks" / name)
                benchmark = load_task(path)
                reference_solve = benchmark.solve
                with patch.object(benchmark, "solve", wraps=reference_solve) as reference, \
                        patch.object(benchmark, "candidate_solve", side_effect=reference_solve), \
                        patch("speedupmark.harness.load_task", return_value=benchmark):
                    result = run_task(path, n=12, seed=123, samples=3)
                self.assertTrue(result.correct)
                self.assertEqual(result.metric_unit, "ms")
                self.assertEqual(reference.call_count, 3)
                self.assertEqual([sample.seed for sample in result.samples], [123, 124, 125])

    def test_runtime_grading_function_replacements_abort(self):
        mutations = {
            "solve": "type(self).solve = lambda self, problem: problem",
            "is_solution": "self.is_solution = lambda problem, proposed: True",
            "evaluate_solution": "self.evaluate_solution = lambda problem, proposed: None",
            "evaluate_pair": "self.evaluate_pair = lambda *args, **kwargs: None",
            "timer": "time.perf_counter_ns = lambda: 0",
        }
        for name, mutation in mutations.items():
            # Seed 8 changes after reference-first; seed 9 changes before the
            # reference could run. A worker contains the deliberately changed clock.
            for change_seed in (8, 9):
                with self.subTest(function=name, seed=change_seed):
                    path = self.task(f'''
                        import time
                        from speedupmark.task import SolutionEvaluation
                        class Task:
                            name = "runtime_change"
                            task_version = "1.0.0"
                            def generate_problem(self, n, random_seed): return random_seed
                            def solve(self, problem): return problem
                            def candidate_solve(self, problem):
                                if problem == {change_seed}:
                                    {mutation}
                                return problem
                            def is_solution(self, problem, proposed): return proposed == problem
                            def evaluate_solution(self, problem, proposed):
                                return SolutionEvaluation(1, proposed == problem)
                            def evaluate_pair(self, problem, outputs, **kwargs):
                                return {{role: self.evaluate_solution(problem, value)
                                        for role, value in outputs.items()}}
                        TASK = Task()
                    ''')
                    with self.assertRaisesRegex(RuntimeError, f"grading function changed: {name}"):
                        _run_isolated(path, 1, 8, 2, 5)

    def test_output_freezing_is_inside_both_timers(self):
        path = self.task('''
            class Task:
                name = "freeze_timing"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return random_seed
                def solve(self, problem): return [problem]
                def candidate_solve(self, problem): return [problem]
                def is_solution(self, problem, proposed): return proposed == [problem]
            TASK = Task()
        ''')
        from speedupmark.harness import _check_grading_functions

        events = []

        def check(task, expected):
            events.append("guard")
            _check_grading_functions(task, expected)

        def clock():
            events.append("clock")
            return len(events) * 1_000_000

        def freeze(value):
            events.append("freeze")
            return freeze_output(value)

        with patch("speedupmark.harness.time.perf_counter_ns", side_effect=clock), \
                patch("speedupmark.harness.freeze_output", side_effect=freeze), \
                patch("speedupmark.harness._check_grading_functions", side_effect=check):
            result = run_task(path, samples=1)
        self.assertEqual(events, ["guard", "clock", "freeze", "clock", "guard"] * 2)
        self.assertEqual(result.reference_score, 2)
        self.assertEqual(result.candidate_score, 2)
        self.assertTrue(result.correct)

    def test_outputs_are_detached_before_peer_calls_and_verification(self):
        path = self.task('''
            class Task:
                name = "output_mutation"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return random_seed
                def solve(self, problem):
                    self.reference_output = [problem]
                    return self.reference_output
                def candidate_solve(self, problem):
                    if hasattr(self, "reference_output"):
                        self.reference_output[0] = -1
                    self.candidate_output = [problem]
                    return self.candidate_output
                def is_solution(self, problem, proposed):
                    self.candidate_output[0] = -1
                    return proposed == [problem]
            TASK = Task()
        ''')
        self.assertTrue(run_task(path, seed=7, samples=3).correct)

    def test_lazy_outputs_fail_before_verification_can_execute_them(self):
        for expression in ("Deferred()", "{'rows': [Deferred()]}", "{'U': DeferredArray()}"):
            with self.subTest(output=expression):
                path = self.task(f'''
                    class Deferred(tuple):
                        def __len__(self): raise AssertionError("lazy length ran")
                        def __iter__(self): raise AssertionError("lazy iteration ran")
                        def __repr__(self): raise AssertionError("lazy repr ran")
                    class DeferredArray:
                        def __array__(self, *args, **kwargs):
                            raise AssertionError("lazy array conversion ran")
                    class Task:
                        name = "lazy_output"
                        task_version = "1.0.0"
                        def generate_problem(self, n, random_seed): return random_seed
                        def solve(self, problem): return [problem]
                        def candidate_solve(self, problem): return {expression}
                        def is_solution(self, problem, proposed):
                            raise AssertionError("invalid output reached verification")
                    TASK = Task()
                ''')
                with self.assertRaisesRegex(ValueError, "candidate output:.*completed plain data"):
                    run_task(path, samples=1)

    def test_session_window_audit_exploit_is_rejected(self):
        path = self.task((ROOT / "tasks/out_of_order_session_windows/task_spec.py").read_text())
        (path / "reference.py").write_bytes((ROOT / "tasks/out_of_order_session_windows/reference.py").read_bytes())
        (path / "candidate.py").write_text(textwrap.dedent('''
            class Deferred(tuple):
                def _value(self):
                    raise AssertionError("untimed session computation ran")
                def __len__(self): return len(self._value())
                def __iter__(self): return iter(self._value())
            def solve(problem):
                return Deferred()
        '''))
        with self.assertRaisesRegex(ValueError, "candidate output:.*completed plain data"):
            run_task(path, n=32, seed=39185, samples=1)

    def test_compiler_receipts_replay_only_the_recorded_source_and_seed(self):
        path = self.submission(ROOT / "tasks/layout_aware_pipeline_compiler")
        records = []
        first = run_task(path, n=8, seed=41, samples=1,
                         verification_record=records.append)
        self.assertTrue(first.correct)
        self.assertEqual(records, [first.samples[0].verification])
        replay = run_task(path, n=8, seed=41, samples=1,
                          verification_replay=records)
        self.assertEqual(replay.samples[0].verification, records[0])
        self.assertEqual(replay.speedup, first.speedup)
        for field, value in (("task_revision", "changed"),
                             ("candidate_sha256", "changed"), ("seed", 42)):
            invalid = [records[0] | {field: value}]
            with self.subTest(field=field), patch("speedupmark.harness.load_task") as load:
                with self.assertRaisesRegex(ValueError, "replay source"):
                    run_task(path, n=8, seed=41, samples=1,
                             verification_replay=invalid)
                load.assert_not_called()

    def test_verifier_receipt_survives_worker_simulation_failure(self):
        path = self.task('''
            from speedupmark.task import SolutionEvaluation
            class Task:
                name = "verifier_crash"
                task_version = "1.0.0"
                metric_unit = "cycles"
                def generate_problem(self, n, random_seed): return random_seed
                def solve(self, problem): return problem
                def evaluate_solution(self, problem, proposed):
                    return SolutionEvaluation(1, True)
                def evaluate_pair(self, problem, outputs, *, replay=None, record=None):
                    record({"private_seed": "saved"})
                    raise RuntimeError("simulation failed")
            TASK = Task()
        ''')
        log = path / "verification.jsonl"
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            _run_isolated(path, None, 7, 1, 5, verification_log=log)
        receipt = json.loads(log.read_text())
        self.assertEqual(receipt["seed"], 7)
        self.assertEqual(receipt["context"], {"private_seed": "saved"})

    def test_candidate_loader_supports_dataclasses(self):
        path = self.task('')
        (path / "candidate.py").write_text(textwrap.dedent('''
            from __future__ import annotations
            from dataclasses import dataclass
            @dataclass
            class Answer:
                value: int
            def solve(problem):
                return Answer(problem)
        '''))
        candidate = load_candidate(str(path / "task_spec.py"))
        self.assertEqual(candidate.solve(12).value, 12)

    def test_reference_is_verified_with_runtime_checks(self):
        path = self.task('''
            class Task:
                name = "broken_reference"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return 1
                def solve(self, problem): return 2
                def is_solution(self, problem, proposed): return proposed == problem
            TASK = Task()
        ''')
        with self.assertRaisesRegex(ValueError, "reference failed"):
            run_task(path)

    def test_deterministic_scores_and_invalid_candidate_costs(self):
        for score in ("0", "-1", "float('nan')", "float('inf')"):
            with self.subTest(score=score):
                path = self.task(f'''
                    from speedupmark.task import SolutionEvaluation
                    class Task:
                        name = "bad_metric"
                        task_version = "1.0.0"
                        metric_unit = "cycles"
                        def generate_problem(self, n, random_seed): return random_seed
                        def solve(self, problem): return 100
                        def candidate_solve(self, problem): return {score}
                        def evaluate_solution(self, problem, proposed):
                            return SolutionEvaluation(proposed, True)
                    TASK = Task()
                ''')
                result = run_task(path, samples=1)
                self.assertFalse(result.correct)
                self.assertEqual(result.speedup, 0)
                self.assertEqual(result.candidate_score, 0)

    def test_discovery_does_not_execute_task_code(self):
        path = self.task('raise RuntimeError("must never import during listing")')
        other = path / "todo"
        other.mkdir()
        (other / "task_spec.py").write_text('class Task:\n    PORT_TODO = True\n')
        self.assertEqual(discover_tasks(path), [])
        self.assertEqual(discover_tasks(path, include_todo=True), [other])
        live = path / "live"
        live.mkdir()
        (live / "task_spec.py").write_text('raise RuntimeError("must never execute")')
        self.assertEqual(discover_tasks(path), [live])

    def test_discovery_reports_broken_syntax(self):
        path = self.task('')
        child = path / "broken"
        child.mkdir()
        (child / "task_spec.py").write_text('this is invalid Python!')
        with self.assertRaises(SyntaxError):
            discover_tasks(path)

    def test_worker_timeout_terminates_hung_task(self):
        path = self.task('''
            class Task:
                name = "hang"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return 1
                def solve(self, problem):
                    while True: pass
            TASK = Task()
        ''')
        with self.assertRaises(subprocess.TimeoutExpired):
            _run_isolated(path, None, 0, 1, 0.3)

    def test_worker_stdout_is_not_result_protocol(self):
        path = self.task('''
            print("import chatter")
            class Task:
                name = "chatty"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return 1
                def solve(self, problem):
                    print("solver chatter")
                    return problem
                def is_solution(self, problem, proposed): return proposed == problem
            TASK = Task()
        ''')
        result = _run_isolated(path, None, 0, 1, 5)
        self.assertTrue(result.correct)

    def test_default_cli_lists_exactly_ten_and_includes_simd_traversal_kernel(self):
        completed = subprocess.run([sys.executable, "-m", "speedupmark", "--list", "--json"], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), list(SMOKE_TASKS))
        self.assertEqual(len(set(SMOKE_TASKS)), 10)
        self.assertIn("simd_traversal_kernel", SMOKE_TASKS)

    def test_cli_returns_failure_and_no_aggregate_for_incorrect_candidate(self):
        path = self.task('''
            class Task:
                name = "wrong"
                task_version = "1.0.0"
                def generate_problem(self, n, random_seed): return 1
                def solve(self, problem): return problem
                def candidate_solve(self, problem): return 2
                def is_solution(self, problem, proposed): return proposed == problem
            TASK = Task()
        ''')
        completed = subprocess.run([sys.executable, "-m", "speedupmark", str(path), "--json", "--samples", "1"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        report = json.loads(completed.stdout)
        self.assertFalse(report["correct"])
        self.assertIsNone(report["geomean_speedup"])
        self.assertEqual(report["results"][0]["task_version"], "1.0.0")
        self.assertEqual(report["results"][0]["speedup"], 0)


class OutputFreezingTests(unittest.TestCase):
    def test_plain_types_and_aliases_are_preserved_without_source_references(self):
        shared = [None, True, 3, 1.5, "text", b"bytes", bytearray(b"mutable")]
        source = {("key", 1): [shared, shared], "tuple": (shared,)}
        frozen = freeze_output(source)
        self.assertEqual(frozen, source)
        self.assertIsNot(frozen, source)
        self.assertIs(frozen[("key", 1)][0], frozen[("key", 1)][1])
        self.assertIsNot(frozen[("key", 1)][0], shared)
        shared[-1][0] = 0
        shared.append("changed")
        self.assertEqual(frozen[("key", 1)][0][-1], bytearray(b"mutable"))
        self.assertIs(type(frozen["tuple"]), tuple)

    def test_custom_values_keys_and_subclasses_are_rejected_without_callbacks(self):
        calls = []

        class Deferred:
            def __float__(self): calls.append("float"); return 1.0
            def __hash__(self): calls.append("hash"); return 1
            def __repr__(self): calls.append("repr"); return "deferred"

        class ForgedDict(dict):
            def items(self): calls.append("items"); return super().items()

        class ForgedInt(int):
            def __int__(self): calls.append("int"); return 1

        class Meta(type):
            def __eq__(self, other): calls.append("metaclass equality"); return False

        class Custom(metaclass=Meta):
            pass

        values = [Deferred(), {"value": Deferred()}, {Deferred(): 1},
                  ForgedDict(value=1), ForgedInt(1), Custom(), iter([1])]
        calls.clear()
        for value in values:
            with self.assertRaisesRegex(ValueError, "completed plain data"):
                freeze_output(value)
        self.assertEqual(calls, [])

    def test_cycles_and_excessive_nesting_fail_cleanly(self):
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(ValueError, "cycles"):
            freeze_output(cyclic)
        deep = []
        for _ in range(2000):
            deep = [deep]
        with self.assertRaisesRegex(ValueError, "deeply nested"):
            freeze_output(deep)

    def test_numpy_arrays_are_detached_and_executable_arrays_rejected(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("optional NumPy is not installed")
        source = numpy.arange(12, dtype=float).reshape(3, 4)[:, ::2]
        frozen = freeze_output({"matrix": source})["matrix"]
        numpy.testing.assert_array_equal(frozen, source)
        self.assertFalse(numpy.shares_memory(frozen, source))
        self.assertFalse(frozen.flags.writeable)
        scalars = [numpy.float32(1.5), numpy.int64(3), numpy.bool_(True)]
        detached_scalars = freeze_output(scalars)
        self.assertEqual(detached_scalars, scalars)
        self.assertEqual([type(value) for value in detached_scalars],
                         [type(value) for value in scalars])
        source[0, 0] = -1
        self.assertEqual(frozen[0, 0], 0)

        class Deferred:
            def __float__(self): raise AssertionError("lazy numeric conversion ran")

        class ArraySubclass(numpy.ndarray):
            def __array__(self, *args, **kwargs):
                raise AssertionError("lazy array conversion ran")

        for value in (numpy.array([Deferred()], dtype=object), source.view(ArraySubclass)):
            with self.assertRaises(ValueError):
                freeze_output(value)


if __name__ == "__main__":
    unittest.main()
