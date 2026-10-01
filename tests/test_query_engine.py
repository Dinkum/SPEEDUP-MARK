"""Canonical ordering, tie-breaks and engine policy for the query engine task."""

import copy
import pathlib
import shutil
import tempfile
import types
import unittest

from speedupmark.harness import load_task
from speedupmark.task import forbidden_imports


ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = ("selective_filter", "permissive_filter", "skew", "topk", "shared_subplan")


class QueryEngineTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(ROOT / "tasks/adaptive_query_engine")

    def test_hand_built_plan_semantics(self):
        tables = {
            "orders": {"columns": ("o_id", "cust", "region", "amount"),
                       "rows": ((0, 0, 1, 10), (1, 0, 2, 20), (2, 1, 1, 30))},
            "customers": {"columns": ("cust", "tier", "signup"),
                          "rows": ((0, 3, 100), (1, 1, 200))},
        }
        # A filter placed above the join still filters the left input's column.
        plan = ("group", ("filter", ("join", ("scan", "orders"), ("scan", "customers"),
                                     "cust", "cust"), "amount", "ge", 20),
                ("tier",), (("sum", "amount"), ("count", None)))
        self.assertEqual(self.task.solve({"families": ({"family": "x", "tables": tables,
                                                        "queries": (plan,)},)}),
                         ((((1, 30, 1), (3, 20, 1)),),))
        # Duplicates survive projection and the output is sorted ascending.
        projected = ("project", ("scan", "orders"), ("region", "amount"))
        self.assertEqual(self.task.solve({"families": ({"family": "x", "tables": tables,
                                                        "queries": (projected,)},)}),
                         ((((1, 10), (1, 30), (2, 20)),),))
        # topk keeps the semantic order (column descending, whole row ascending)
        # and returns the survivors ascending.
        topk = ("project", ("topk", ("scan", "orders"), "amount", 2), ("o_id", "amount"))
        self.assertEqual(self.task.solve({"families": ({"family": "x", "tables": tables,
                                                        "queries": (topk,)},)}),
                         ((((1, 20), (2, 30)),),))

    def test_topk_breaks_ties_at_the_boundary_by_whole_row(self):
        tables = {"t": {"columns": ("k", "v"), "rows": ((0, 5), (1, 5), (2, 5), (3, 5))}}
        plan = ("topk", ("scan", "t"), "v", 2)
        self.assertEqual(self.task.solve({"families": ({"family": "x", "tables": tables,
                                                        "queries": (plan,)},)}),
                         ((((0, 5), (1, 5)),),))
        wider = ("topk", ("scan", "t"), "v", 3)
        self.assertEqual(self.task.solve({"families": ({"family": "x", "tables": tables,
                                                        "queries": (wider,)},)}),
                         ((((0, 5), (1, 5), (2, 5)),),))

    def test_all_five_families_generate_exact_integer_answers(self):
        for size in (self.task.grading_cases[0],):
            problem = self.task.generate_problem(size, 4)
            self.assertEqual([family["family"] for family in problem["families"]],
                             list(FAMILIES))
            pristine = copy.deepcopy(problem)
            answers = self.task.solve(problem)
            self.assertEqual(problem, pristine)
            self.assertTrue(self.task.is_solution(problem, answers))
            for family_answers in answers:
                for rows in family_answers:
                    for row in rows:
                        for value in row:
                            self.assertIs(type(value), int)

    def test_candidate_and_reference_agree_and_lists_are_rejected_once_mutated(self):
        problem = self.task.generate_problem(320, 1)
        candidate = self.task.candidate_solve(copy.deepcopy(problem))
        self.assertTrue(self.task.is_solution(problem, candidate))
        mutated = [list(family) for family in candidate]
        if mutated[0]:
            mutated[0][0] = list(mutated[0][0]) + [(0,)]
        self.assertFalse(self.task.is_solution(problem, mutated))

    def test_embedded_engine_imports_are_detected(self):
        module = types.ModuleType("candidate")
        module.sqlite3 = types.ModuleType("sqlite3")
        module.helper = lambda: None
        self.assertEqual(forbidden_imports(module, ("sqlite3", "duckdb")), ("sqlite3",))
        # `from sqlite3 import connect` binds a C-extension function, so the name
        # scan alone misses it. The runtime signal has to do the work: a module
        # that appears after the snapshot is a violation even if nothing is bound.
        from sqlite3 import connect
        import sys

        bound = types.ModuleType("candidate")
        bound.connect = connect
        self.assertEqual(forbidden_imports(bound, ("sqlite3",)), ())
        before = set(sys.modules)
        sys.modules["duckdb"] = types.ModuleType("duckdb")
        try:
            self.assertEqual(forbidden_imports(bound, ("sqlite3", "duckdb"), before),
                             ("duckdb",))
            clean = types.ModuleType("candidate")
            clean.helper = lambda: None
            self.assertEqual(forbidden_imports(clean, ("sqlalchemy",), before), ())
        finally:
            del sys.modules["duckdb"]


class QueryEnginePolicyTests(unittest.TestCase):
    """Delegating to an embedded SQL engine is the one thing the task forbids."""

    HOSTILE = '''"""Hostile probe: run the whole query through an embedded engine."""


def solve(problem, reference_solve):
    import sqlite3
    _ = sqlite3.connect
    return reference_solve(problem)
'''

    CLEAN = '''"""Honest probe: dictionaries and tuples only."""


def solve(problem, reference_solve):
    scored = {family["family"]: len(family["queries"]) for family in problem["families"]}
    assert all(value > 0 for value in scored.values())
    return reference_solve(problem)
'''

    def grade(self, source):
        directory = pathlib.Path(tempfile.mkdtemp()) / "hostile_query_policy"
        directory.mkdir()
        shutil.copy(ROOT / "tasks/adaptive_query_engine/task_spec.py",
                    directory / "task_spec.py")
        (directory / "candidate.py").write_text(source)
        task = load_task(directory)
        problem = task.generate_problem(task.grading_cases[0], 0)
        task.candidate_solve(problem)
        return task.policy_violations

    def test_function_local_engine_import_is_reported(self):
        violations = self.grade(self.HOSTILE)
        self.assertIn("sqlite3", violations)

    def test_an_honest_candidate_reports_nothing(self):
        self.assertEqual(self.grade(self.CLEAN), ())


if __name__ == "__main__":
    unittest.main()
