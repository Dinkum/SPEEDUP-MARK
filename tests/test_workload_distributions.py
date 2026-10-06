"""Distribution documentation and a generator termination regression."""

import importlib.util
import pathlib
import subprocess
import sys
import unittest

from speedupmark.harness import discover_tasks, load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]


class WorkloadDistributionTests(unittest.TestCase):
    def test_every_runnable_task_documents_its_distribution(self):
        catalog = (ROOT / 'GUIDE.md').read_text()
        for path in discover_tasks():
            with self.subTest(task=path.name):
                text = (path / 'README.md').read_text()
                self.assertIn('## Workload distribution\n', text)
                section = text.split('## Workload distribution\n', 1)[1].split('\n## ', 1)[0]
                for field in ('Size', 'Selection', 'Randomized', 'Fixed structure'):
                    self.assertIn(f'**{field}:** ', section)
                self.assertIn('task_spec.py', section)
                self.assertIn(f'tasks/{path.name}/README.md#workload-distribution', catalog)

    def test_discrete_log_real_moduli_and_distinct_order_regimes(self):
        task = load_task(ROOT / 'tasks' / 'discrete_log')
        for size in (12, *task.grading_cases):
            for base in (0, 1, 2, 17, 2**63 - 3):
                families = set()
                for seed in range(base, base + 3):
                    problem = task.generate_problem(size, seed)
                    self.assertEqual(problem, task.generate_problem(size, seed))
                    self.assertEqual(set(problem), {'p', 'g', 'h'})
                    self.assertEqual(problem['p'].bit_length(), size)
                    answer = task.solve(problem)
                    self.assertTrue(task.is_solution(problem, answer))
                    families.add(task.workload_family(size, seed))
                self.assertEqual(families, {'smooth', 'medium_subgroup', 'large_subgroup'})
        for family in range(3):
            problems = [task.generate_problem(50, seed) for seed in range(family, 18, 3)]
            self.assertEqual(len({tuple(sorted(p.items())) for p in problems}), 6)

    def test_factorization_real_sizes_and_four_structural_families(self):
        task = load_task(ROOT / 'tasks' / 'integer_factorization')
        for size in (24, *task.grading_cases):
            families = set()
            for seed in range(8):
                problem = task.generate_problem(size, seed)
                self.assertEqual(problem, task.generate_problem(size, seed))
                self.assertEqual(set(problem), {'composite'})
                self.assertIn(problem['composite'].bit_length(), (size - 1, size))
                answer = task.solve(problem)
                self.assertTrue(task.is_solution(problem, answer))
                families.add(task.workload_family(size, seed))
                if seed % 4 == 1:
                    self.assertLess(answer['q'] - answer['p'], 1500)
                if seed % 4 == 3:
                    self.assertLessEqual(answer['p'].bit_length(), 20)
            self.assertEqual(families, {'balanced', 'close', 'smooth_factor', 'unbalanced'})
        for family in range(4):
            values = [task.generate_problem(64, seed)['composite']
                      for seed in range(family, 24, 4)]
            self.assertEqual(len(set(values)), 6)

    def test_number_theory_generation_is_bounded_at_smallest_sizes(self):
        program = """
from speedupmark.harness import load_task
from pathlib import Path
for name, sizes in [('integer_factorization', (24, 25, 26)), ('discrete_log', (12, 13, 14))]:
    task = load_task(Path('tasks') / name)
    for size in sizes:
        for seed in range(24):
            problem = task.generate_problem(size, seed)
            assert task.is_solution(problem, task.solve(problem))
"""
        subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT,
                       check=True, timeout=10, capture_output=True, text=True)

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),
                         'requires optional numerical dependencies')
    def test_delaunay_clustered_declared_sizes_terminate_with_unique_points(self):
        # The original 9x9 jitter boxes could supply at most 243 points,
        # so the declared 256-point case never completed. Bound the regression.
        program = '''
from speedupmark.harness import load_task
from pathlib import Path
task = load_task(Path('tasks/delaunay'))
for n in task.grading_cases:
    for seed in (0, 3):
        problem = task.generate_problem(n, seed)
        assert len(problem['points']) == n
        assert len({tuple(p) for p in problem['points']}) == n
        assert task.is_solution(problem, task.solve(problem))
'''
        subprocess.run([sys.executable, '-B', '-c', program], cwd=ROOT,
                       check=True, timeout=30, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
