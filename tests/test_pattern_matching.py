"""Independent `re` oracle, chunk seams and non-saturation for pattern matching."""

import copy
import importlib.util
import pathlib
import pickle
import re
import shutil
import sys
import tempfile
import types
import unittest

from speedupmark.harness import load_task
from speedupmark.task import forbidden_imports


ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = ("literal_heavy", "automata_heavy", "many_patterns_few_streams",
            "few_patterns_many_streams", "boundary_crossing")


def to_regex(pattern):
    """Translate the task's pattern IR into a bytes regex for cross-checking."""
    kind = pattern[0]
    if kind == "lit":
        return re.escape(pattern[1])
    if kind == "class":
        return b"[" + re.escape(bytes(sorted(pattern[1]))) + b"]"
    if kind == "cat":
        return b"(?:" + b"".join(to_regex(child) for child in pattern[1:]) + b")"
    if kind == "alt":
        return b"(?:" + b"|".join(to_regex(child) for child in pattern[1:]) + b")"
    if kind == "rep":
        _, child, low, high = pattern
        inner = to_regex(child)
        if high is None:
            return b"(?:" + inner + b"){%d,}" % low
        if low == high:
            return b"(?:" + inner + b"){%d}" % low
        return b"(?:" + inner + b"){%d,%d}" % (low, high)
    raise ValueError(f"unsupported pattern node {kind!r}")


class PatternMatchingTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(ROOT / "tasks/multi_pattern_matching")

    def test_oracle_agrees_with_python_regex_engine(self):
        """The strongest available check on the task's own matcher."""
        for size in (self.task.grading_cases[0],):
            problem = self.task.generate_problem(size, 5)
            self.assertEqual([family["family"] for family in problem["families"]],
                             list(FAMILIES))
            answers = self.task.solve(problem)
            checked = 0
            for family, family_answers in zip(problem["families"], answers):
                regexes = [re.compile(to_regex(pattern)) for pattern in family["patterns"]]
                for chunks, mask in zip(family["streams"], family_answers):
                    data = b"".join(chunks)
                    for index, expression in enumerate(regexes):
                        expected = expression.search(data) is not None
                        self.assertEqual(bool((mask >> index) & 1), expected,
                                         f"{family['family']}/{index} disagrees with re")
                        checked += 1
            self.assertGreater(checked, 100)

    def test_matches_across_chunk_boundaries_are_required(self):
        """Restarting the matcher at every chunk boundary must give a wrong answer."""
        problem = self.task.generate_problem(self.task.grading_cases[0], 0)
        family = problem["families"][4]
        self.assertEqual(family["family"], "boundary_crossing")
        answers = self.task.solve({"families": (family,)})[0]
        resetting = []
        for chunks in family["streams"]:
            mask = 0
            for index, pattern in enumerate(family["patterns"]):
                program = self.task_module()._compile(pattern)
                if any(self.task_module()._matches(chunk, program) for chunk in chunks):
                    mask |= 1 << index
            resetting.append(mask)
        self.assertNotEqual(tuple(resetting), tuple(answers))

    def task_module(self):
        import importlib.util
        import sys

        name = "speedupmark_pattern_helpers"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, ROOT / "tasks/multi_pattern_matching/task_spec.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    def test_no_stream_matches_every_pattern(self):
        """A constant 'everything matched' answer must never be correct."""
        for size in self.task.grading_cases:
            for seed in (0, 1, 2):
                problem = self.task.generate_problem(size, seed)
                answers = self.task.solve(problem)
                for family, family_answers in zip(problem["families"], answers):
                    full = (1 << len(family["patterns"])) - 1
                    for stream_answers in family_answers:
                        self.assertNotEqual(stream_answers, full)

    def test_every_pattern_needs_at_least_one_byte(self):
        module = self.task_module()
        for size in (self.task.grading_cases[0],):
            problem = self.task.generate_problem(size, 7)
            for family in problem["families"]:
                for pattern in family["patterns"]:
                    self.assertGreaterEqual(module._reference._min_length(pattern), 1)

    def test_nullable_whole_patterns_are_outside_the_contract(self):
        module = self.task_module()
        patterns = (
            ("rep", ("lit", b"a"), 0, 1),
            ("rep", ("lit", b"a"), 0, None),
            ("lit", b""),
            ("alt", ("lit", b""), ("lit", b"a")),
        )
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, "positive minimum match length"):
                    module._compile(pattern)
                for data in (b"", b"b", b"a"):
                    problem = {"families": ({"patterns": (pattern,),
                                             "streams": ((data,),)},)}
                    with self.assertRaises(ValueError):
                        self.task.solve(problem)
                    for mask in (0, 1):
                        self.assertFalse(self.task.is_solution(problem, ((mask,),)))

    def test_nullable_subpatterns_preserve_nonempty_matching(self):
        patterns = (
            ("cat", ("rep", ("lit", b"a"), 0, 1), ("lit", b"b")),
            ("cat", ("rep", ("rep", ("lit", b"a"), 0, 1), 0, None),
             ("lit", b"b")),
            ("cat", ("alt", ("lit", b""), ("lit", b"a")), ("lit", b"b")),
            ("cat", ("lit", b"a"), ("rep", ("lit", b"b"), 0, 1),
             ("lit", b"c")),
        )
        streams = ((b"",), (b"x",), (b"b",), (b"a", b"b"),
                   (b"aa", b"ab"), (b"a", b"c"), (b"a", b"b", b"c"))
        problem = {"families": ({"patterns": patterns, "streams": streams},)}
        expected = tuple(
            sum(1 << index for index, pattern in enumerate(patterns)
                if re.search(to_regex(pattern), b"".join(chunks)) is not None)
            for chunks in streams
        )
        self.assertEqual(self.task.solve(problem), (expected,))
        self.assertTrue(self.task.is_solution(problem, (expected,)))

    def test_generation_is_deterministic_picklable_and_candidate_agrees(self):
        problem = self.task.generate_problem(self.task.grading_cases[0], 2)
        self.assertEqual(problem, self.task.generate_problem(self.task.grading_cases[0], 2))
        self.assertEqual(pickle.loads(pickle.dumps(problem)), problem)
        pristine = copy.deepcopy(problem)
        reference = self.task.solve(problem)
        self.assertEqual(problem, pristine)
        candidate = self.task.solve(copy.deepcopy(problem))
        self.assertTrue(self.task.is_solution(problem, candidate))
        self.assertFalse(self.task.is_solution(problem, None))
        self.assertFalse(self.task.is_solution(problem, "nope"))
        broken = [list(family) for family in reference]
        broken[0][0] = True
        self.assertFalse(self.task.is_solution(problem, broken))


class EnginePolicyTests(unittest.TestCase):
    """The regex engine is the task, so importing it is a violation.

    A function-local ``import re`` binds nothing at import time, and ``re`` is
    already in ``sys.modules`` before the task runs, so neither a namespace scan
    nor a ``sys.modules`` delta catches it. It must still be reported, and the
    honest candidate must stay clean.
    """

    HOSTILE = '''"""Hostile probe: delegate to the forbidden engine from inside solve()."""


def solve(problem):
    import re
    _ = re.escape
    return ()
'''

    CLEAN = '''"""Honest probe: ordinary structures, no engine."""


def solve(problem):
    seen = set()
    for family in problem["families"]:
        seen.update(str(len(family["streams"])).encode())
    return ()
'''

    def grade(self, source):
        directory = pathlib.Path(tempfile.mkdtemp()) / "hostile_pattern_policy"
        directory.mkdir()
        shutil.copy(ROOT / "tasks/multi_pattern_matching/task_spec.py",
                    directory / "task_spec.py")
        shutil.copy(ROOT / "tasks/multi_pattern_matching/reference.py", directory / "reference.py")
        (directory / "candidate.py").write_text(source)
        task = load_task(directory)
        problem = task.generate_problem(task.grading_cases[0], 0)
        task.candidate_solve(problem)
        return task.policy_violations

    def test_function_local_regex_import_is_reported(self):
        self.assertIn("re", self.grade(self.HOSTILE))

    def test_an_honest_candidate_reports_nothing(self):
        self.assertEqual(self.grade(self.CLEAN), ())

    def test_preloaded_roots_are_still_detected_by_name(self):
        """A snapshot that already contains the root must not hide the import."""
        spec = importlib.util.spec_from_file_location(
            "policy_scan_probe", ROOT / "tasks/multi_pattern_matching/task_spec.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hostile = types.ModuleType("hostile")
        exec(compile(self.HOSTILE, "candidate.py", "exec"), hostile.__dict__)
        self.assertIn("re", forbidden_imports(hostile, ("re", "_sre"), imported_since=set(sys.modules)))


if __name__ == "__main__":
    unittest.main()
