"""Distribution documentation and a generator termination regression."""

import importlib.util
import pathlib
import subprocess
import sys
import unittest

from speedupmark.harness import discover_tasks


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
