"""Cross-check packed gate faults against direct single-pattern simulation."""

import operator
import pathlib
import random
import unittest
from unittest.mock import patch

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]


def scalar_faults(problem):
    operations = {"and": operator.and_, "or": operator.or_, "xor": operator.xor}
    gates = problem["gates"]
    rows = [[0, 0] for _ in gates]
    for bit in range(problem["pattern_count"]):
        inputs = [(value >> bit) & 1 for value in problem["inputs"]]

        def observe(site, stuck):
            values = inputs.copy()
            for i, (op, a, b) in enumerate(gates):
                values.append(stuck if i == site else operations[op](values[a], values[b]))
            return tuple(values[o] for o in problem["outputs"])

        normal = observe(None, None)
        for i, row in enumerate(rows):
            for stuck in (0, 1):
                if observe(i, stuck) != normal:
                    row[stuck] |= 1 << bit
    return [tuple(row) for row in rows]


class CircuitFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task(ROOT / "tasks" / "circuit_fault_simulation", with_candidate=False)

    def test_split_reconverge_observe_and_dead_gate(self):
        problem = {"pattern_count": 4, "inputs": [0b1100, 0b1010],
                   "gates": [("xor", 0, 1), ("and", 2, 2), ("or", 2, 2),
                             ("xor", 3, 4), ("and", 0, 1)], "outputs": [5]}
        expected = [(0, 0), (6, 9), (6, 9), (0, 15), (0, 0)]
        self.assertEqual(self.task.solve(problem), expected)
        problem["outputs"].append(2)
        expected[0] = (6, 9)
        self.assertEqual(self.task.solve(problem), expected)
        self.assertEqual(scalar_faults(problem), expected)

    def test_small_circuits_against_independent_oracle(self):
        for seed in range(64):
            rng = random.Random(seed)
            width = rng.randrange(1, 13)
            inputs = [rng.getrandbits(width) for _ in range(5)]
            gates = [(rng.choice(("and", "or", "xor")), rng.randrange(i), rng.randrange(i))
                     for i in range(5, 38)]
            problem = {"pattern_count": width, "inputs": inputs, "gates": gates,
                       "outputs": rng.sample(range(38), 9)}
            with self.subTest(seed=seed):
                self.assertEqual(self.task.solve(problem), scalar_faults(problem))

    def test_generator_replay_topology_and_packed_batch_boundaries(self):
        for seed in range(3):
            for n in (1, 250):
                with self.subTest(seed=seed, n=n):
                    p = self.task.generate_problem(n, seed)
                    self.assertEqual(p, self.task.generate_problem(n, seed))
                    self.assertEqual(p["pattern_count"], (64, 256, 512)[seed])
                    self.assertGreaterEqual(len(p["gates"]), n)
                    for wire, (_, left, right) in enumerate(p["gates"], len(p["inputs"])):
                        self.assertTrue(0 <= left < wire and 0 <= right < wire)
                    # Check bits at both ends of the packed word with a scalar oracle.
                    indices = (0, p["pattern_count"] - 1)
                    def project(value):
                        return sum(((value >> bit) & 1) << i for i, bit in enumerate(indices))
                    small = dict(p, pattern_count=2, inputs=[project(v) for v in p["inputs"]])
                    answers = self.task.solve(p)
                    self.assertEqual([(project(a), project(b)) for a, b in answers],
                                     scalar_faults(small))

    def test_grader_uses_measured_reference_and_rejects_invalid_masks(self):
        problem = {"pattern_count": 1, "inputs": [1], "gates": [("xor", 0, 0)], "outputs": [1]}
        expected = [(0, 1)]
        with patch.object(self.task, "solve", side_effect=AssertionError("reference ran twice")):
            self.assertTrue(self.task.is_solution(problem, ([0, 1],), reference_output=expected))
            for wrong in ([], [(0, True)], [(0, 2)], [(1, 0)], [(-1, 1)], [(0.0, 1)], [(0, 1, 0)]):
                self.assertFalse(self.task.is_solution(problem, wrong, reference_output=expected))
