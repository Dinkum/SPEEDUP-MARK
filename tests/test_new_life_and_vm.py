"""Reference correctness and generator guarantees for the VM and Life tasks."""

import copy
import pathlib
import sys
import unittest

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]


def task(name):
    return load_task(ROOT / "tasks" / name, with_candidate=False)


class LifeTests(unittest.TestCase):
    def setUp(self):
        self.task = task("sparse_life_generations")

    def advance(self, cells, generations):
        return self.task.solve({"patterns": [{"cells": [list(c) for c in cells],
                                              "generations": generations}]})["patterns"][0]

    def test_still_life_oscillator_and_glider(self):
        block = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.assertEqual(self.advance(block, 5), [list(c) for c in sorted(block)])
        blinker = [(0, 0), (1, 0), (2, 0)]
        self.assertEqual(self.advance(blinker, 2), [list(c) for c in sorted(blinker)])
        self.assertEqual(self.advance(blinker, 1), [[1, -1], [1, 0], [1, 1]])
        glider = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)]
        moved = sorted((x + 1, y + 1) for x, y in glider)
        self.assertEqual(self.advance(glider, 4), [list(c) for c in moved])

    def test_gun_grows_five_cells_every_thirty_generations(self):
        spec = sys.modules["speedupmark_task_sparse_life_generations"]
        sizes = [len(self.advance(spec.GUN, 30 * k)) for k in range(5)]
        self.assertEqual(sizes, [36, 41, 46, 51, 56])

    def test_generator_covers_every_family_and_is_deterministic(self):
        problem = self.task.generate_problem(8, 3)
        self.assertEqual(problem, self.task.generate_problem(8, 3))
        self.assertEqual(len(problem["patterns"]), 8)
        out = self.task.solve(copy.deepcopy(problem))
        self.assertTrue(self.task.is_solution(problem, out, reference_output=out))

    def test_verifier_rejects_wrong_shape_order_and_values(self):
        problem = self.task.generate_problem(4, 1)
        out = self.task.solve(copy.deepcopy(problem))
        bad = copy.deepcopy(out)
        bad["patterns"][0].reverse()
        self.assertFalse(self.task.is_solution(problem, bad, reference_output=out))
        bad = copy.deepcopy(out)
        bad["patterns"][0] = bad["patterns"][0][:-1]
        self.assertFalse(self.task.is_solution(problem, bad, reference_output=out))
        bad = copy.deepcopy(out)
        bad["patterns"][0][0] = [True, 0]
        self.assertFalse(self.task.is_solution(problem, bad, reference_output=out))
        self.assertFalse(self.task.is_solution(problem, {"patterns": out["patterns"][:-1]},
                                               reference_output=out))


class VmTests(unittest.TestCase):
    def setUp(self):
        self.task = task("register_vm_execution")

    def run_code(self, code, registers=None, memory=None):
        program = {"code": code, "registers": registers or [0] * 16, "memory": memory or [0] * 4}
        problem = {"programs": [program]}
        out = self.task.solve(copy.deepcopy(problem))
        self.assertTrue(self.task.is_solution(problem, out))
        return out["states"][0]

    def test_arithmetic_wraps_and_memory_addresses_wrap(self):
        registers, memory = self.run_code([
            [1, 0, 0xFFFFFFFF, 0], [9, 1, 0, 2], [5, 2, 0, 0], [15, 1, 0, 6], [0, 0, 0, 0]])
        self.assertEqual(registers[1], 1)
        self.assertEqual(registers[2], 1)
        self.assertEqual(memory, [0, 1, 0, 0])  # (0xFFFFFFFF + 6) % 4 == 1

    def test_loop_call_and_return(self):
        # r0 += 3, five times, through a called routine.
        registers, _ = self.run_code([
            [1, 1, 5, 0], [19, 5, 0, 0], [9, 1, 1, 0xFFFFFFFF], [18, 1, 1, 0], [0, 0, 0, 0],
            [9, 0, 0, 3], [20, 0, 0, 0]])
        self.assertEqual(registers[0], 15)

    def test_generated_programs_terminate_and_verifier_agrees(self):
        for seed in range(3):
            problem = self.task.generate_problem(8, seed)
            out = self.task.solve(copy.deepcopy(problem))
            self.assertTrue(self.task.is_solution(problem, out))
            broken = copy.deepcopy(out)
            broken["states"][0][0][0] ^= 1
            self.assertFalse(self.task.is_solution(problem, broken))

    def test_cold_programs_dominate_program_count_not_work(self):
        problem = self.task.generate_problem(1500, 0)
        cold = [p for p in problem["programs"] if len(p["memory"]) == 16]
        self.assertGreater(len(cold), 2000)
        self.assertGreater(len(problem["programs"]) - len(cold), 6)


if __name__ == "__main__":
    unittest.main()
