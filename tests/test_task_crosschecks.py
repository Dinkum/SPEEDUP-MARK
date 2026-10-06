"""Small exhaustive and differential checks using independent algorithms."""

import itertools
import pathlib
import random
import re
import unittest
from unittest.mock import patch

from speedupmark.harness import load_task


TASK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tasks"


def _routing_cost(problem, routes):
    return sum(problem["D"][left][right]
               for route in routes for left, right in zip(route, route[1:]))


def _exhaustive_routing_options(problem):
    # Enumerate customer orders and nonempty route cuts directly, without the
    # reference's subset DP, reconstruction helpers, or optimality checker.
    depot = problem["depot"]
    customers = [node for node in range(len(problem["D"])) if node != depot]
    for order in itertools.permutations(customers):
        for cuts in itertools.combinations(range(1, len(customers)), problem["K"] - 1):
            boundaries = (0, *cuts, len(customers))
            routes = [[depot, *order[start:end], depot]
                      for start, end in zip(boundaries, boundaries[1:])]
            yield _routing_cost(problem, routes), routes


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
        task = load_task(TASK_ROOT / "multi_literal_replacement")
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

    def test_vehicle_routing_matches_exhaustive_orders_and_splits(self):
        task = load_task(TASK_ROOT / "vehicle_routing")
        rng = random.Random(142607)
        for customers in range(1, 6):
            size = customers + 1
            for trial in range(8):
                distance = [[0] * size for _ in range(size)]
                for left in range(size):
                    for right in range(left + 1, size):
                        distance[left][right] = distance[right][left] = rng.randint(1, 30)
                depot = rng.randrange(size)
                for vehicles in range(1, customers + 1):
                    with self.subTest(customers=customers, trial=trial, vehicles=vehicles,
                                      depot=depot):
                        problem = {"D": distance, "K": vehicles, "depot": depot}
                        options = list(_exhaustive_routing_options(problem))
                        optimum, best_routes = min(options, key=lambda option: option[0])
                        worst_cost, worst_routes = max(options, key=lambda option: option[0])
                        reference_routes = task.solve(problem)
                        self.assertEqual(_routing_cost(problem, reference_routes), optimum)
                        self.assertTrue(task.is_solution(problem, reference_routes))
                        self.assertTrue(task.is_solution(problem, best_routes))
                        if worst_cost > optimum:
                            self.assertFalse(task.is_solution(problem, worst_routes))

    def test_vehicle_routing_accepts_optimal_routes_and_rejects_feasible_slow_routes(self):
        task = load_task(TASK_ROOT / "vehicle_routing")
        problem = {"D": [[0, 1, 1, 1], [1, 0, 1, 7],
                         [1, 1, 0, 9], [1, 7, 9, 0]], "K": 2, "depot": 0}
        optimal = [[0, 1, 2, 0], [0, 3, 0]]
        alternative = [[0, 3, 0], [0, 2, 1, 0]]
        suboptimal = [[0, 1, 3, 0], [0, 2, 0]]
        self.assertEqual(_routing_cost(problem, optimal), 5)
        self.assertEqual(_routing_cost(problem, suboptimal), 11)
        self.assertTrue(task.is_solution(problem, optimal))
        self.assertTrue(task.is_solution(problem, alternative))
        self.assertFalse(task.is_solution(problem, suboptimal))
        for cost, routes in _exhaustive_routing_options(problem):
            self.assertEqual(task.is_solution(problem, routes), cost == 5)

    def test_vehicle_routing_accepts_all_tied_route_orders(self):
        task = load_task(TASK_ROOT / "vehicle_routing")
        distance = [[int(left != right) for right in range(5)] for left in range(5)]
        for vehicles in range(1, 5):
            problem = {"D": distance, "K": vehicles, "depot": 2}
            # Each valid route has one extra depot leg: four customers + K.
            for cost, routes in _exhaustive_routing_options(problem):
                self.assertEqual(cost, 4 + vehicles)
                self.assertTrue(task.is_solution(problem, routes))

    def test_sqlite_reference_on_null_duplicate_and_tie_fixture(self):
        task = load_task(TASK_ROOT / "grouped_analytics_reports")
        problem = {
            "rows": ((0, None, None, None), (0, None, None, None),
                     (1, "a", 5, "note"), (1, "a", 5, "note"), (2, "b", 10, None),
                     (3, "c", -2, "x"), (3, "c", None, "x"), (4, "z", None, None)),
            "minimum_total": 10,
        }
        expected = (
            (("a", 2, 2, 10, 1), ("b", 1, 1, 10, 0), ("c", 2, 1, -2, 1),
             (None, 2, 0, None, 0), ("z", 1, 0, None, 0)),
            ((0, None, 2), (1, "a", 2), (3, "c", 2)),
            ((1, 10), (2, 10)),
        )
        self.assertEqual(task.solve(problem), expected)
        self.assertEqual(task.solve({"rows": (), "minimum_total": 0}), ((), (), ()))
        self.assertEqual(task.solve(problem | {"minimum_total": 11}), (*expected[:2], ()))
        lists = [[list(row) for row in report] for report in expected]
        wrong_sum = ((expected[0][0][:3] + (11, 1), *expected[0][1:]), *expected[1:])
        wrong_type = ((("a", True, 2, 10, 1), *expected[0][1:]), *expected[1:])
        with patch.object(task, "solve", side_effect=AssertionError("verifier reran reference")):
            self.assertTrue(task.is_solution(problem, expected, reference_output=expected))
            self.assertTrue(task.is_solution(problem, lists, reference_output=expected))
            for bad in (expected[:2], (tuple(reversed(expected[0])), *expected[1:]),
                        wrong_sum, wrong_type):
                with self.subTest(output=bad):
                    self.assertFalse(task.is_solution(problem, bad, reference_output=expected))


if __name__ == "__main__":
    unittest.main()
