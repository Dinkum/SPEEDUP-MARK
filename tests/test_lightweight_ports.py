"""Independent small oracles and witness attacks for the stdlib adaptations."""

import copy
import importlib.util
import itertools
import random
import unittest
from unittest.mock import patch
import zlib
from pathlib import Path


NAMES = (
    "integer_factorization", "minimum_spanning_tree", "articulation_points",
    "min_weight_assignment", "kd_tree", "max_flow_min_cost", "gzip_compression",
    "matrix_multiplication", "queens_with_obstacles",
)


def load(name):
    path = Path(__file__).resolve().parents[1] / "tasks" / name / "task_spec.py"
    spec = importlib.util.spec_from_file_location("port_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASK


class LightweightPortsTests(unittest.TestCase):
    def test_seeded_baselines_and_strict_outputs(self):
        for name in NAMES:
            task = load(name)
            size = {"integer_factorization": 1000, "gzip_compression": 2000,
                    "queens_with_obstacles": 4}.get(name, 8)
            for seed in range(6):
                with self.subTest(name=name, seed=seed):
                    problem = task.generate_problem(size, seed)
                    self.assertEqual(problem, task.generate_problem(size, seed))
                    original = copy.deepcopy(problem)
                    answer = task.solve(problem)
                    self.assertEqual(original, problem)
                    self.assertTrue(task.is_solution(problem, answer))
                    self.assertTrue(task.is_solution(problem, task.candidate_solve(problem)))
                    for bad in [None, True, (), iter([]), {"junk": 0}]:
                        self.assertFalse(task.is_solution(problem, bad))

    def test_factor_primes_not_just_product(self):
        task = load("integer_factorization")
        self.assertTrue(task.is_solution({"composite": 15}, {"p": 3, "q": 5}))
        for bad in [{"p": 1, "q": 15}, {"p": True, "q": 15}, {"p": 5, "q": 3}]:
            self.assertFalse(task.is_solution({"composite": 15}, bad))
        self.assertFalse(task.is_solution({"composite": 60}, {"p": 4, "q": 15}))

    def test_mst_exhaustive_small(self):
        task = load("minimum_spanning_tree")
        for seed in range(8):
            problem = task.generate_problem(5, seed)
            edges = problem["edges"]
            best = None
            for chosen in itertools.combinations(range(len(edges)), 4):
                reached = {0}
                for _ in range(5):
                    for i in chosen:
                        u, v, _ = edges[i]
                        if u in reached or v in reached:
                            reached.update([u, v])
                if len(reached) == 5:
                    cost = sum(edges[i][2] for i in chosen)
                    best = cost if best is None else min(best, cost)
            answer = task.solve(problem)
            self.assertEqual(sum(edges[i][2] for i in answer["edge_indices"]), best)
            self.assertTrue(task.is_solution(problem, answer))
        problem = {"num_nodes": 3, "edges": [[0, 1, 1], [1, 2, 1], [0, 2, 9]]}
        self.assertFalse(task.is_solution(problem, {"edge_indices": [0, 2]}))
        self.assertFalse(task.is_solution(problem, {"edge_indices": [False, 1]}))

    def test_articulation_all_graphs_four_nodes(self):
        task = load("articulation_points")
        edges = list(itertools.combinations(range(4), 2))
        for mask in range(1 << len(edges)):
            problem = {"num_nodes": 4, "edges": [list(e) for i, e in enumerate(edges) if mask >> i & 1]}
            answer = task.solve(problem)
            self.assertTrue(task.is_solution(problem, answer))
            wrong = {"articulation_points": sorted(set(answer["articulation_points"]) ^ {0})}
            self.assertFalse(task.is_solution(problem, wrong))
        self.assertFalse(task.is_solution({"num_nodes": 3, "edges": [[0, 1], [1, 2]]},
                                          {"articulation_points": [True]}))

    def test_assignment_all_permutations(self):
        task = load("min_weight_assignment")
        for seed in range(8):
            problem = task.generate_problem(5, seed)
            costs = problem["costs"]
            score = lambda p: sum(costs[i][j] for i, j in enumerate(p))
            optimum = min(map(score, itertools.permutations(range(5))))
            self.assertEqual(score(task.solve(problem)["assignment"]), optimum)
            for permutation in itertools.permutations(range(5)):
                self.assertEqual(task.is_solution(problem, {"assignment": list(permutation)}),
                                 score(permutation) == optimum)
        self.assertFalse(task.is_solution({"costs": [[0, 0], [0, 0]]}, {"assignment": [False, True]}))

    def test_knn_duplicates_and_ties(self):
        task = load("kd_tree")
        problem = {"points": [[1, 0], [-1, 0], [1, 0], [10, 0]], "queries": [[0, 0]], "k": 2}
        self.assertEqual(task.solve(problem), {"indices": [[0, 1]]})
        for bad in [[[1, 0]], [[0, 2]], [[0, 3]], [[0, True]], [[0, 0]]]:
            self.assertFalse(task.is_solution(problem, {"indices": bad}))

    def test_flow_exhaustive_capacities(self):
        task = load("max_flow_min_cost")
        rng = random.Random(42)
        for _ in range(12):
            edges = [[u, v, rng.randrange(1, 3), rng.randrange(4)]
                     for u, v in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]]
            problem = {"num_nodes": 4, "source": 0, "sink": 3, "edges": edges}
            feasible = []
            for flow in itertools.product(*(range(e[2] + 1) for e in edges)):
                if flow[0] != flow[2] + flow[3] or flow[1] + flow[2] != flow[4]:
                    continue
                objective = (-(flow[0] + flow[1]), sum(f * e[3] for f, e in zip(flow, edges)))
                feasible.append((objective, list(flow)))
            optimum = min(obj for obj, _ in feasible)
            self.assertTrue(task.is_solution(problem, task.solve(problem)))
            for objective, flow in feasible:
                self.assertEqual(task.is_solution(problem, {"flow": flow}), objective == optimum)
        self.assertFalse(task.is_solution({"num_nodes": 2, "source": 0, "sink": 1,
                                           "edges": [[0, 1, 1, 0]]}, {"flow": [True]}))

    def test_gzip_size_and_format(self):
        task = load("gzip_compression")
        problem = {"plaintext": b"repeated data\n" * 2000}
        good = task.solve(problem)
        self.assertTrue(task.is_solution(problem, good))
        compressor = zlib.compressobj(level=0, wbits=31)
        uncompressed = compressor.compress(problem["plaintext"]) + compressor.flush()
        for data in [uncompressed, good["compressed_data"] + b"junk", good["compressed_data"][:-1],
                     bytearray(good["compressed_data"]), zlib.compress(problem["plaintext"])]:
            self.assertFalse(task.is_solution(problem, {"compressed_data": data}))
        self.assertTrue(task.is_solution({"plaintext": b""}, task.solve({"plaintext": b""})))

    def test_integer_matrix_known_products(self):
        task = load("matrix_multiplication")
        problem = {"A": [[1, 2, -3], [0, 4, 5]], "B": [[2, 0], [1, -1], [3, 2]]}
        expected = [[-5, -8], [19, 6]]
        self.assertEqual(task.solve(problem), expected)
        self.assertTrue(task.is_solution(problem, expected))
        self.assertFalse(task.is_solution(problem, [[-5.0, -8], [19, 6]]))
        self.assertFalse(task.is_solution({"A": [[1]], "B": [[1]]}, [[True]]))

    def test_queens_exhaustive_small(self):
        task = load("queens_with_obstacles")
        for seed in range(12):
            problem = task.generate_problem(3, seed)
            board = problem["obstacles"]
            squares = [(r, c) for r in range(len(board)) for c in range(len(board[0])) if not board[r][c]]
            def safe(placed):
                for (r, c), (rr, cc) in itertools.combinations(placed, 2):
                    if r != rr and c != cc and abs(r - rr) != abs(c - cc):
                        continue
                    steps = max(abs(rr - r), abs(cc - c))
                    dr, dc = (rr - r) // steps, (cc - c) // steps
                    if not any(board[r + k * dr][c + k * dc] for k in range(1, steps)):
                        return False
                return True
            optimum = max(len(placed) for count in range(len(squares) + 1)
                          for placed in itertools.combinations(squares, count) if safe(placed))
            answer = task.solve(problem)
            self.assertEqual(len(answer["queens"]), optimum)
            self.assertTrue(task.is_solution(problem, answer))
            if optimum:
                self.assertFalse(task.is_solution(problem, {"queens": answer["queens"][:-1]}))
        problem = {"obstacles": [[False, True, False]]}
        self.assertTrue(task.is_solution(problem, {"queens": [[0, 0], [0, 2]]}))
        self.assertFalse(task.is_solution(problem, {"queens": [[False, 0], [0, 2]]}))
        self.assertFalse(task.is_solution({"obstacles": [[False, False]]}, {"queens": [[0, 0], [0, 1]]}))
        self.assertTrue(task.is_solution({"obstacles": [[True]]}, {"queens": []}))

    def test_queens_optimality_does_not_trust_reference_search(self):
        task = load("queens_with_obstacles")
        problem = {"obstacles": [[False]]}
        with patch.dict(task.solve.__globals__, {"maximum_clique": lambda _: []}):
            self.assertEqual(task.solve(problem), {"queens": []})
            self.assertFalse(task.is_solution(problem, {"queens": []}))
            self.assertTrue(task.is_solution(problem, {"queens": [[0, 0]]}))


if __name__ == "__main__":
    unittest.main()
