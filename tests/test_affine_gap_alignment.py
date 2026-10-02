"""Exact alignment objectives checked against exhaustive tiny alignment paths."""

import itertools
import pathlib
import unittest
from unittest.mock import patch

from speedupmark.harness import load_task

ROOT = pathlib.Path(__file__).resolve().parents[1]


def exhaustive(left, right, mismatch, opened, extended):
    def visit(i, j, previous):
        if i == len(left) and j == len(right):
            return 0
        choices = []
        if i < len(left) and j < len(right):
            choices.append((0 if left[i] == right[j] else mismatch) + visit(i + 1, j + 1, 0))
        if i < len(left):
            choices.append((extended if previous == 1 else opened) + visit(i + 1, j, 1))
        if j < len(right):
            choices.append((extended if previous == 2 else opened) + visit(i, j + 1, 2))
        return min(choices)
    return visit(0, 0, 0)


class AffineGapAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task(ROOT / 'tasks/affine_gap_sequence_alignment')
        cls.module = __import__(cls.task.__class__.__module__, fromlist=['_reference_cost'])

    def test_every_tiny_alignment_matches_exhaustive_paths(self):
        strings = [bytes(items) for length in range(4)
                   for items in itertools.product(b'AB', repeat=length)]
        for left, right in itertools.product(strings, repeat=2):
            for penalties in ((2, 5, 1), (9, 2, 1), (4, 1, 3)):
                case = (left, right, *penalties)
                expected = exhaustive(*case)
                with self.subTest(case=case):
                    self.assertEqual(self.module._reference_cost(*case), expected)
                    self.assertEqual(self.module._checked_cost(*case), expected)

    def test_boundary_gaps_and_opposite_gaps(self):
        for case, expected in (((b'', b'', 4, 5, 2), 0),
                               ((b'AAAA', b'', 4, 5, 2), 11),
                               ((b'', b'AAA', 4, 5, 2), 9),
                               ((b'A', b'B', 9, 2, 1), 4),
                               ((b'ABC', b'ABC', 4, 5, 2), 0)):
            self.assertEqual(self.module._reference_cost(*case), expected)
            self.assertTrue(self.task.is_solution({'alignments': (case,)}, (expected,)))

    def test_checker_does_not_trust_reference(self):
        problem = {'alignments': ((b'A', b'B', 9, 2, 1),)}
        with patch.object(self.module, '_reference_cost', return_value=0):
            self.assertFalse(self.task.is_solution(problem, self.task.solve(problem)))
            self.assertTrue(self.task.is_solution(problem, (4,)))

    def test_wrong_values_shapes_and_lazy_outputs_fail(self):
        problem = {'alignments': ((b'A', b'B', 3, 5, 1),)}
        class Lazy(tuple):
            def __iter__(self):
                raise AssertionError('deferred work ran')
        for value in ((), (0,), (-1,), (True,), (3.0,), (3, 3), Lazy((3,)), iter((3,))):
            self.assertFalse(self.task.is_solution(problem, value))
        self.assertTrue(self.task.is_solution(problem, [3]))

    def test_generated_families_and_sizes_pass_independent_checker(self):
        for size in (1, 16, *self.task.grading_cases):
            for seed in (0, 1):
                problem = self.task.generate_problem(size, seed)
                self.assertEqual(len(problem['alignments']), 4)
                self.assertEqual(problem, self.task.generate_problem(size, seed))
                self.assertTrue(self.task.is_solution(problem, self.task.solve(problem)))
        for size in (0, True, 513, 1.5):
            with self.assertRaises(ValueError):
                self.task.generate_problem(size)
