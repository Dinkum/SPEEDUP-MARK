"""Independent adversarial checks for the frontier task contracts."""

import pathlib
import random
import unittest

from speedupmark.harness import load_task


TASK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tasks"


def task(name):
    return load_task(TASK_ROOT / name)


class FrontierSemanticValidationTests(unittest.TestCase):
    def test_join_replacements_deletions_and_cancellation(self):
        benchmark = task("incremental_multiway_join")
        problem = {
            "scenarios": ({
                "family": "adversarial",
                "relations": {
                    "R": ((0, 0, 2), (1, 0, -2)),
                    "S": ((0, 0, 3),),
                    "T": ((0, 0, 5), (0, 1, 5)),
                },
                "operations": (
                    ("query",),
                    ("set", "T", 0, 1, 7),
                    ("query",),
                    ("delete", "R", 0, 0),
                    ("query",),
                    ("set", "R", 0, 0, 0),
                    ("query",),
                ),
            },),
        }
        expected = ((0, -12, -42, -42),)
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertTrue(benchmark.is_solution(problem, expected))
        self.assertFalse(benchmark.is_solution(problem, ((0, -12, -42, True),)))

    def test_shortest_paths_directed_updates_self_loop_and_disconnection(self):
        benchmark = task("dynamic_shortest_paths")
        problem = {
            "scenarios": ({
                "family": "adversarial",
                "node_count": 3,
                "edges": ((0, 0, 1), (0, 1, 5), (1, 2, 7), (0, 2, 20)),
                "operations": (
                    ("query", 0, 2),
                    ("set", 0, 2, 3),
                    ("query", 0, 2),
                    ("delete", 0, 2),
                    ("query", 0, 2),
                    ("delete", 1, 2),
                    ("query", 0, 2),
                    ("query", 2, 2),
                ),
            },),
        }
        expected = ((12, 3, 12, None, 0),)
        self.assertEqual(benchmark.solve(problem), expected)
        self.assertTrue(benchmark.is_solution(problem, expected))
        self.assertFalse(benchmark.is_solution(problem, ((12, 3, 12, None, False),)))

    def test_queens_clique_search_against_bruteforce_graphs(self):
        benchmark = task("queens_with_obstacles")
        maximum_clique = benchmark.solve.__func__.__globals__["maximum_clique"]
        rng = random.Random(9173)
        for size in range(1, 10):
            for _ in range(20):
                neighbors = [0] * size
                for left in range(size):
                    for right in range(left + 1, size):
                        if rng.randrange(2):
                            neighbors[left] |= 1 << right
                            neighbors[right] |= 1 << left
                expected = max(
                    subset.bit_count()
                    for subset in range(1 << size)
                    if all(
                        not (subset >> left & 1) or
                        subset & ~neighbors[left] & ~(1 << left) == 0
                        for left in range(size)
                    )
                )
                self.assertEqual(len(maximum_clique(neighbors)), expected)


if __name__ == "__main__":
    unittest.main()
