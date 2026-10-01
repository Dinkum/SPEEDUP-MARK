"""Small exhaustive and differential checks using independent algorithms."""

import itertools
import pathlib
import random
import re
import unittest

from speedupmark.harness import load_task


TASK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tasks"


class IndependentOracleTests(unittest.TestCase):
    def test_all_three_vertex_graph_pairs_against_permutations(self):
        task = load_task(TASK_ROOT / "labeled_graph_isomorphism")
        possible_edges = list(itertools.combinations(range(3), 2))
        labels = (0, 1, 0)
        for left_mask, right_mask in itertools.product(range(8), repeat=2):
            left = tuple(edge for i, edge in enumerate(possible_edges) if left_mask >> i & 1)
            right = tuple(edge for i, edge in enumerate(possible_edges) if right_mask >> i & 1)
            problem = {
                "left": {"labels": labels, "edges": left},
                "right": {"labels": labels, "edges": right},
            }
            expected = any(
                all(labels[i] == labels[mapping[i]] for i in range(3))
                and {tuple(sorted((mapping[x], mapping[y]))) for x, y in left} == set(right)
                for mapping in itertools.permutations(range(3))
            )
            output = task.solve(problem)
            self.assertEqual(output is not None, expected)
            self.assertTrue(task.is_solution(problem, output))

    def test_bpe_binary_strings_against_one_merge_at_a_time(self):
        task = load_task(TASK_ROOT / "ranked_bpe_tokenization")
        merges = ((97, 97), (256, 97), (97, 98), (258, 258))
        for length in range(9):
            for data in itertools.product((97, 98), repeat=length):
                tokens = list(data)
                while True:
                    matches = [
                        (rank, position)
                        for position in range(len(tokens) - 1)
                        for rank, pair in enumerate(merges)
                        if tuple(tokens[position : position + 2]) == pair
                    ]
                    if not matches:
                        break
                    rank, position = min(matches)
                    tokens[position : position + 2] = [256 + rank]
                problem = {"data": bytes(data), "merges": merges}
                self.assertEqual(task.solve(problem), tuple(tokens))

    def test_replacement_against_literal_regex_alternation(self):
        task = load_task(TASK_ROOT / "streaming_literal_replacement")
        rng = random.Random(873)
        for _ in range(100):
            data = bytes(rng.choice(b"abc") for _ in range(40))
            patterns = tuple(
                (
                    bytes(rng.choice(b"abc") for _ in range(rng.randrange(1, 4))),
                    bytes([65 + index]),
                )
                for index in range(12)
            )
            first = {}
            for needle, value in patterns:
                first.setdefault(needle, value)
            expression = b"|".join(
                re.escape(needle) for needle in sorted(first, key=lambda item: -len(item))
            )
            expected = re.sub(expression, lambda match: first[match.group()], data)
            problem = {
                "chunks": tuple(data[i : i + 3] for i in range(0, len(data), 3)),
                "replacements": patterns,
            }
            self.assertEqual(task.solve(problem), expected)

    def test_sqlite_and_python_agree_on_small_null_heavy_inputs(self):
        task = load_task(TASK_ROOT / "sqlite_analytics_reports")
        for seed in range(20):
            for size in (0, 1, 2, 5, 30):
                with self.subTest(seed=seed, size=size):
                    problem = task.generate_problem(size, seed)
                    self.assertTrue(task.is_solution(problem, task.solve(problem)))


if __name__ == "__main__":
    unittest.main()
