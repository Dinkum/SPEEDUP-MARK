"""Completed outputs, numerical objectives, and complete planar witnesses."""

import copy
import importlib.util
import pathlib
import unittest
from unittest.mock import Mock, patch

from speedupmark.harness import load_task
from speedupmark.task import freeze_output

ROOT = pathlib.Path(__file__).resolve().parents[1]
NUMERICAL = all(importlib.util.find_spec(name) for name in ('numpy', 'scipy', 'cvxpy'))


def task(name):
    return load_task(ROOT / 'tasks' / name)


class CompletedOutputTests(unittest.TestCase):
    def test_output_freezing_rejects_hooks_and_cycles_without_callbacks(self):
        class Lazy(tuple):
            def __iter__(self):
                raise AssertionError('deferred work ran')

        class ArrayHook:
            def __array__(self, *args, **kwargs):
                raise AssertionError('deferred work ran')

        cyclic = []
        cyclic.append(cyclic)
        for proposed in (Lazy(), {'nested': Lazy()}, ArrayHook(), cyclic, iter(())):
            with self.subTest(kind=type(proposed).__name__), self.assertRaises(ValueError):
                freeze_output(proposed)
        original = {'values': [1, (2.0, None)], 'bytes': bytearray(b'data')}
        frozen = freeze_output(original)
        original['values'][0] = 9
        original['bytes'][0] = 0
        self.assertEqual(frozen, {'values': [1, (2.0, None)], 'bytes': bytearray(b'data')})

    def test_session_verifier_rejects_lazy_nested_containers(self):
        class Lazy(tuple):
            def __len__(self):
                raise AssertionError('deferred work ran')

        benchmark = task('out_of_order_session_windows')
        problem = benchmark.generate_problem(12, 0)
        expected = benchmark.solve(problem)
        self.assertTrue(benchmark.is_solution(problem, expected))
        self.assertFalse(benchmark.is_solution(problem, Lazy()))
        self.assertFalse(benchmark.is_solution(problem, (Lazy(), *expected[1:])))

    def test_facility_supported_minimum_is_feasible(self):
        benchmark = task('capacitated_facility_location')
        for size in (0, 1, 2):
            with self.assertRaisesRegex(ValueError, 'at least 3'):
                benchmark.generate_problem(size)
        for seed in range(9):
            problem = benchmark.generate_problem(3, seed)
            self.assertTrue(benchmark.is_solution(problem, benchmark.solve(problem)))


@unittest.skipUnless(NUMERICAL, 'requires pinned numerical extra')
class NumericalWitnessTests(unittest.TestCase):
    def test_tiny_delaunay_cluster_cannot_collapse_to_one_line(self):
        benchmark = task('delaunay')
        rng = Mock()
        rng.randint.side_effect = [0] * 6 + [0, 0, 1, 0, 2, 0]
        with patch('random.Random', return_value=rng):
            problem = benchmark.generate_problem(3, 0)
        self.assertEqual(len({tuple(point) for point in problem['points']}), 3)
        self.assertTrue(benchmark.is_solution(problem, benchmark.solve(problem)))

    def test_sparse_eigenvalue_starter_survives_arpack_nonconvergence(self):
        import numpy
        from scipy.sparse.linalg import ArpackNoConvergence
        benchmark = task('sparse_lowest_eigenvalues_posdef')
        problem = benchmark.generate_problem(60, 1)
        failure = ArpackNoConvergence('no convergence', numpy.array([]), numpy.empty((60, 0)))
        with patch('scipy.sparse.linalg.eigsh', side_effect=failure):
            answer = benchmark.solve(problem)
        self.assertTrue(benchmark.is_solution(problem, answer))

    def test_conversion_hooks_are_rejected_before_coercion(self):
        class Deferred:
            def __array__(self, *args, **kwargs):
                raise AssertionError('candidate work ran in verifier')

        cases = {
            'randomized_svd': (8, ('U', 'S', 'V')),
            'matrix_completion': (3, ('B',)),
            'robust_kalman_filter': (4, ('x_hat', 'w_hat', 'v_hat')),
            'rocket_landing_optimization': (8, ('position', 'velocity', 'thrust')),
        }
        for name, (size, fields) in cases.items():
            benchmark = task(name)
            problem = benchmark.generate_problem(size, 0)
            answer = benchmark.solve(problem)
            self.assertTrue(benchmark.is_solution(problem, answer))
            for field in fields:
                with self.subTest(task=name, field=field):
                    self.assertFalse(benchmark.is_solution(problem, answer | {field: Deferred()}))

    def test_nonfinite_objective_fields_are_rejected(self):
        for name, size, field in (('matrix_completion', 3, 'optimal_value'),
                                  ('rocket_landing_optimization', 8, 'fuel_consumption')):
            benchmark = task(name)
            problem = benchmark.generate_problem(size, 0)
            answer = benchmark.solve(problem)
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(task=name, value=value):
                    self.assertFalse(benchmark.is_solution(problem, answer | {field: value}))

    def test_delaunay_mesh_requires_complete_nonoverlapping_hull(self):
        benchmark = task('delaunay')
        # Six vertices on one circle. Two disjoint triangles pass empty-circle
        # checks but do not form a triangulation of the convex hull.
        points = [[5, 0], [3, 4], [-3, 4], [-5, 0], [-3, -4], [3, -4]]
        problem = {'points': points}
        forged = {'simplices': [[0, 1, 2], [3, 4, 5]],
                  'convex_hull': [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]}
        self.assertFalse(benchmark.is_solution(problem, forged))
        valid = benchmark.solve(problem)
        self.assertTrue(benchmark.is_solution(problem, valid))
        duplicate = copy.deepcopy(valid)
        duplicate['simplices'].append(valid['simplices'][0])
        self.assertFalse(benchmark.is_solution(problem, duplicate))
        duplicate = copy.deepcopy(valid)
        duplicate['convex_hull'].append(valid['convex_hull'][0])
        self.assertFalse(benchmark.is_solution(problem, duplicate))
        # Collinear hull points must split the boundary edge.
        problem = {'points': [[0, 0], [2, 0], [4, 0], [4, 4], [0, 4], [2, 2]]}
        self.assertTrue(benchmark.is_solution(problem, benchmark.solve(problem)))
        # Both cocircular diagonals remain legal; orientation is immaterial.
        problem = {'points': [[0, 0], [1, 0], [1, 1], [0, 1]]}
        for triangles in ([[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]):
            self.assertTrue(benchmark.is_solution(problem, {
                'simplices': triangles, 'convex_hull': [[0, 1], [1, 2], [2, 3], [3, 0]]}))


if __name__ == '__main__':
    unittest.main()
