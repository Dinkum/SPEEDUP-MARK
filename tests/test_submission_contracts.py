"""Regression checks for untimed-work and engine-delegation loopholes."""

import pathlib
import shutil
import tempfile
import types
import unittest

from speedupmark.harness import load_task
from speedupmark.task import forbidden_imports


ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tasks"


class SubmissionContractTests(unittest.TestCase):
    def test_sqlite_and_log_outputs_reject_forged_nested_sequences(self):
        for name in ("grouped_analytics_reports", "durable_log_recovery"):
            task = load_task(TASK_ROOT / name)
            problem = task.generate_problem(32, 0)
            expected = task.solve(problem)
            self.assertTrue(task.is_solution(problem, expected))
            for base in (list, tuple):
                callbacks = []

                class Forged(base):
                    def __len__(self):
                        callbacks.append("length")
                        return claimed_length

                    def __iter__(self):
                        callbacks.append("iteration")
                        return iter(())

                # Check the result and each nested report/row independently.
                for location in (None, 0):
                    claimed_length = len(expected if location is None else expected[0])
                    proposed = Forged() if location is None else (Forged(), *expected[1:])
                    with self.subTest(task=name, container=base.__name__, location=location):
                        callbacks.clear()
                        self.assertFalse(task.is_solution(problem, proposed))
                        self.assertEqual(callbacks, [])

                if name == "grouped_analytics_reports":
                    claimed_length = len(expected[0][0])
                    proposed = ((Forged(), *expected[0][1:]), *expected[1:])
                    with self.subTest(task=name, container=base.__name__, location="row"):
                        callbacks.clear()
                        self.assertFalse(task.is_solution(problem, proposed))
                        self.assertEqual(callbacks, [])

    def test_literal_replacement_rejects_bytes_with_forged_equality(self):
        task = load_task(TASK_ROOT / "multi_literal_replacement")
        problem = {"chunks": (b"abc",), "replacements": ((b"a", b"x"),)}
        callbacks = []

        class Forged(bytes):
            def __eq__(self, other):
                callbacks.append("equality")
                return True

        self.assertTrue(task.is_solution(problem, b"xbc"))
        self.assertFalse(task.is_solution(problem, b"wrong"))
        self.assertFalse(task.is_solution(problem, Forged(b"wrong")))
        self.assertEqual(callbacks, [])

    def test_graph_mapping_rejects_deferred_search_before_callbacks(self):
        task = load_task(TASK_ROOT / "labeled_graph_isomorphism")
        problem = task.generate_problem(6, 0)
        for base in (list, tuple):
            searches = []

            class Lazy(base):
                def values(self):
                    if not searches:
                        searches.append(task.solve(problem))
                    return searches[0]

                def __len__(self):
                    return len(self.values())

                def __iter__(self):
                    return iter(self.values())

                def __getitem__(self, index):
                    return self.values()[index]

            with self.subTest(container=base.__name__):
                self.assertFalse(task.is_solution(problem, Lazy()))
                self.assertEqual(searches, [])
        mapping = task.solve(problem)
        self.assertTrue(task.is_solution(problem, mapping))
        self.assertTrue(task.is_solution(problem, list(mapping)))

    def test_import_policy_ignores_literals_variables_and_attribute_names(self):
        module = types.ModuleType("candidate")
        exec('def solve(problem):\n'
             '    label = "re"\n'
             '    re = "sqlite3"\n'
             '    return problem.re, label, re\n', module.__dict__)
        self.assertEqual(forbidden_imports(module, ("re", "sqlite3")), ())

    def test_import_policy_finds_aliased_dotted_and_nested_unused_imports(self):
        sources = (
            "def unused():\n    import re as matcher\n",
            "def unused():\n    from re import compile as matcher\n",
            "def unused():\n    import sqlite3.dbapi2 as database\n",
            "def outer():\n    def unused():\n        import re\n    return unused\n",
        )
        for source in sources:
            with self.subTest(source=source):
                module = types.ModuleType("candidate")
                exec(source, module.__dict__)
                expected = ("sqlite3",) if "sqlite3" in source else ("re",)
                self.assertEqual(forbidden_imports(module, ("re", "sqlite3")), expected)

    def test_temporal_join_rejects_forged_results_and_rows_before_callbacks(self):
        task = load_task(TASK_ROOT / "temporal_asof_join")
        problem = {
            "dimensions": (("a", 5, 0, "value"),),
            "events": ((0, "a", 5), (1, "missing", 5)),
        }
        expected = ((0, "value"), (1, None))
        self.assertTrue(task.is_solution(problem, expected))
        self.assertTrue(task.is_solution(problem, [list(row) for row in expected]))
        self.assertFalse(task.is_solution(problem, (expected[0],)))
        self.assertFalse(task.is_solution(problem, ((0, "wrong"), expected[1])))

        for base in (list, tuple):
            callbacks = []

            class Forged(base):
                def __len__(self):
                    callbacks.append("length")
                    return 2

                def __iter__(self):
                    callbacks.append("iteration")
                    return iter(())

            for location, proposed in (
                ("result", Forged()),
                ("row", (Forged(), expected[1])),
            ):
                with self.subTest(container=base.__name__, location=location):
                    callbacks.clear()
                    self.assertFalse(task.is_solution(problem, proposed))
                    self.assertEqual(callbacks, [])

    def test_lazy_container_subclasses_are_rejected_before_iteration(self):
        class Lazy(tuple):
            def __len__(self):
                raise AssertionError("lazy length ran during verification")

            def __iter__(self):
                raise AssertionError("lazy computation ran during verification")

        names = (
            "incremental_spreadsheet_recalculation",
            "near_duplicate_document_clustering",
            "dynamic_document_search",
            "dynamic_shortest_paths",
            "incremental_multiway_join",
            "ranked_bpe_tokenization",
        )
        for name in names:
            with self.subTest(task=name):
                task = load_task(TASK_ROOT / name)
                problem = task.generate_problem(32, 0)
                self.assertTrue(task.is_solution(problem, task.solve(problem)))
                self.assertFalse(task.is_solution(problem, Lazy()))
                if name in ("dynamic_document_search", "dynamic_shortest_paths",
                            "incremental_multiway_join"):
                    self.assertFalse(task.is_solution(problem, (Lazy(),)))

    def test_existing_engine_imports_fail_their_task_contracts(self):
        cases = (
            ("dynamic_document_search", "sqlite3"),
            ("incremental_multiway_join", "sqlite3"),
            ("ranked_bpe_tokenization", "tokenizers"),
            ("adaptive_query_engine", "pandas"),
        )
        for name, root in cases:
            with self.subTest(task=name), tempfile.TemporaryDirectory() as temp:
                task_dir = pathlib.Path(temp) / name
                task_dir.mkdir()
                shutil.copyfile(TASK_ROOT / name / "task_spec.py", task_dir / "task_spec.py")
                (task_dir / "candidate.py").write_text(
                    "def solve(problem, reference_solve):\n"
                    "    return reference_solve(problem)\n\n"
                    "def unused():\n"
                    f"    import {root}\n"
                )
                task = load_task(task_dir)
                problem = task.generate_problem(32, 0)
                self.assertIsNone(task.candidate_solve(problem))
                self.assertIn(root, task.policy_violations)

    def test_clean_reference_delegate_remains_valid(self):
        for name in ("dynamic_document_search", "incremental_multiway_join",
                     "ranked_bpe_tokenization"):
            with self.subTest(task=name):
                task = load_task(TASK_ROOT / name)
                problem = task.generate_problem(32, 0)
                self.assertTrue(task.is_solution(problem, task.candidate_solve(problem)))
                self.assertEqual(task.policy_violations, ())


if __name__ == "__main__":
    unittest.main()
