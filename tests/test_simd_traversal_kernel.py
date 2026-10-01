"""SIMD Traversal Kernel semantics, machine hazards, private inputs, and measured runway."""

import copy
import pathlib
import sys
import unittest
from unittest import mock

from speedupmark.harness import load_task, run_task
from simd_traversal_kernel_schedules import compile_workload


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SIMDTraversalKernelTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(ROOT / "tasks/simd_traversal_kernel")
        self.spec = sys.modules[type(self.task).__module__]

    def machine(self, memory=None):
        memory = tuple(range(64)) if memory is None else memory
        return self.spec.Machine({"memory": len(memory)}, memory)

    def test_generator_is_descriptor_only_deterministic_and_covers_all_families(self):
        with mock.patch.object(self.spec, "_runtime_memory", side_effect=AssertionError("early data")):
            for size in (1, 13, *self.task.grading_cases):
                problem = self.task.generate_problem(size, 91)
                self.assertEqual(problem, self.task.generate_problem(size, 91))
                self.assertEqual(tuple(w["family"] for w in problem["workloads"]), self.spec.FAMILIES)
                self.assertTrue(all(type(value) in (str, int) for w in problem["workloads"] for value in w.values()))
                self.task.candidate_solve(problem)
        for invalid in (0, -1, True, 4097, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.task.generate_problem(invalid)

    def test_scalar_vector_interleaved_and_cached_programs_match_oracle(self):
        for size in (1, 7, 13, 32):
            problem = self.task.generate_problem(size, 12)
            for index, workload in enumerate(problem["workloads"]):
                memory = self.spec._runtime_memory(workload, "ab" * 32, index, 0)
                expected = self.spec._oracle(workload, memory)
                for lanes, streams, cached in ((1, 1, False), (8, 1, False),
                                               (8, 2, False), (8, 4, True)):
                    with self.subTest(size=size, family=workload["family"], lanes=lanes, streams=streams):
                        program = compile_workload(workload, lanes=lanes, streams=streams,
                                                   cached=cached, scheduled=True)
                        machine = self.spec.Machine(workload, memory)
                        self.assertGreater(machine.run(program), 0)
                        actual = tuple(tuple(machine.memory[workload[name]:workload[name] + workload["batch"]])
                                       for name in ("out_positions", "out_values"))
                        self.assertEqual(actual, expected)

    def test_region_padding_varies_relative_bank_alignment(self):
        residues = {family: [set() for _ in range(5)] for family in self.spec.FAMILIES}
        for seed in range(32):
            for workload in self.task.generate_problem(32, seed)["workloads"]:
                regions = [(name, workload[name], length) for name, length in (
                    ("payload", workload["nodes"]), ("edges", 2 * workload["nodes"]),
                    ("positions", workload["batch"]), ("values", workload["batch"]),
                    ("out_positions", workload["batch"]), ("out_values", workload["batch"]))]
                for index, ((_, left, length), (_, right, _)) in enumerate(zip(regions, regions[1:])):
                    self.assertIn(right - left - length, range(self.spec.BANKS))
                    residues[workload["family"]][index].add((right - left) % self.spec.BANKS)
        for family, pairs in residues.items():
            with self.subTest(family=family):
                self.assertTrue(all(values == set(range(self.spec.BANKS)) for values in pairs))

    def test_arithmetic_wrap_shifts_rotate_and_partial_width(self):
        machine = self.machine()
        machine.run((("const", 8, 0, 0xffffffff), ("const", 8, 1, 1),
                     ("add", 8, 2, 0, 1), ("muli", 8, 3, 0, 2),
                     ("rotli", 8, 4, 1, 31), ("shri", 8, 5, 4, 31),
                     ("andi", 8, 6, 0, 255), ("xor", 8, 7, 6, 1),
                     ("rotli", 8, 8, 0, 0), ("const", 1, 0, 7)))
        self.assertEqual(machine.registers[0], [7] + [0xffffffff] * 7)
        for reg, value in ((2, 0), (3, 0xfffffffe), (4, 0x80000000),
                           (5, 1), (6, 255), (7, 254), (8, 0xffffffff)):
            self.assertEqual(machine.registers[reg], [value] * 8)

    def test_one_round_uses_both_edges_and_the_old_value_for_rotation(self):
        workload = self.task.generate_problem(2)["workloads"][0] | {"rounds": 1}
        memory = [0] * workload["memory"]
        memory[workload["payload"] + 2] = 1
        memory[workload["edges"] + 4:workload["edges"] + 6] = [3, 4]
        memory[workload["positions"]:workload["positions"] + 2] = [2, 2]
        memory[workload["values"]:workload["values"] + 2] = [0, 1]
        expected = ((4, 3), (2654435761, 128))
        self.assertEqual(self.spec._oracle(workload, memory), expected)
        for program in (self.spec._compile(workload), compile_workload(workload, cached=True)):
            machine = self.spec.Machine(workload, memory)
            machine.run(program)
            actual = tuple(tuple(machine.memory[workload[key]:workload[key] + 2])
                           for key in ("out_positions", "out_values"))
            self.assertEqual(actual, expected)

    def test_bank_coalescing_and_conflicts_have_exact_costs(self):
        for indexes, expected in (([0] * 8, 6), (list(range(8)), 7),
                                   ([4 * i for i in range(8)], 13)):
            machine = self.machine()
            machine.registers[0] = indexes
            self.assertEqual(machine.run((("gather", 8, 1, 0, 0),)), expected)
            self.assertEqual(machine.stats["load_words"], len(set(indexes)))

    def test_register_memory_and_write_after_write_hazards(self):
        machine = self.machine()
        # A scalar load completes at 6; dependent add at 7; store at 13;
        # the aliased reload must wait for the store and finishes at 19.
        self.assertEqual(machine.run((("load", 1, 0, 3), ("add", 1, 1, 0, 0),
                                      ("store", 1, 1, 20), ("load", 1, 2, 20))), 19)
        self.assertEqual(machine.registers[2][0], 6)
        machine = self.machine()
        self.assertEqual(machine.run((("load", 1, 0, 3), ("const", 1, 0, 9))), 7)
        self.assertEqual(machine.registers[0][0], 9)

    def test_issue_capacity_and_independent_work_overlap(self):
        machine = self.machine()
        self.assertEqual(machine.run((("const", 8, 0, 1), ("const", 8, 1, 2),
                                      ("const", 8, 2, 3))), 2)
        machine = self.machine()
        self.assertEqual(machine.run((("load", 8, 0, 0), ("const", 8, 1, 1))), 7)

    def test_invalid_opcodes_bounds_registers_and_types_fail(self):
        workload = self.task.generate_problem(1)["workloads"][0]
        for instruction in (("sleep", 1, 0, 1), ("load", 9, 0, 0),
                            ("load", 1, -1, 0), ("load", 1, 20, 0),
                            ("load", 1, 0, -1), ("load", 8, 0, workload["memory"] - 1),
                            ("const", 1, 0, -1), ("shri", 1, 0, 0, 32),
                            ("add", 1, 0, 1), ("const", True, 0, 1),
                            ("load", 1, 0, 0, 0)):
            with self.subTest(instruction=instruction):
                self.assertFalse(self.task.evaluate_solution({"workloads": (workload,)}, ((instruction,),)).correct)
        machine = self.machine()
        with self.assertRaises(ValueError):
            machine.run((("const", 8, 0, 64), ("gather", 8, 1, 0, 0)))
        machine = self.machine()
        with self.assertRaises(ValueError):
            machine.run((("const", 8, 0, 16), ("pick", 8, 1, 2, 3, 0)))

    def test_lazy_forged_and_oversized_submissions_never_execute(self):
        class Forged(list):
            def __iter__(self):
                raise AssertionError("candidate callback ran")
        class Number(int):
            pass
        problem = self.task.generate_problem(1)
        for output in (Forged(), [Forged()] * 4, [[("const", 1, 0, Number(0))]] * 4,
                       [[("const", 1, 0, 1 << 20000)]] * 4,
                       [[("const", 1, 0, 0)] * 10000] * 4, (), (object(),) * 4):
            self.assertFalse(self.task.evaluate_solution(problem, output).correct)

    def test_pair_freezes_programs_before_drawing_inputs_and_replays(self):
        problem = self.task.generate_problem(3, 19)
        outputs = {role: [[list(inst) for inst in program] for program in self.task.solve(problem)]
                   for role in ("reference", "candidate")}
        pristine = copy.deepcopy(outputs)
        receipts = []
        def entropy(_):
            outputs["candidate"][0][0][0] = "broken_after_freeze"
            return "12" * 32
        with mock.patch.object(self.spec.secrets, "token_hex", side_effect=entropy):
            result = self.task.evaluate_pair(problem, outputs, record=receipts.append)
        self.assertTrue(all(e.correct for e in result.values()))
        self.assertEqual(result["reference"].score, result["candidate"].score)
        self.assertEqual(result, self.task.evaluate_pair(problem, pristine, replay=receipts[0]))
        for key, replacement in (("seed", "bad"), ("memory_sha256", "0" * 64),
                                 ("generator_version", "other"), ("problem_sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.task.evaluate_pair(problem, pristine, replay=receipts[0] | {key: replacement})
        with self.assertRaises(ValueError):
            self.task.evaluate_pair(problem, outputs, replay=receipts[0])

    def test_precomputed_answers_and_wrong_final_positions_fail(self):
        problem = self.task.generate_problem(2, 0)
        workload = problem["workloads"][0]
        old = self.spec._runtime_memory(workload, "00" * 32, 0, 0)
        expected = self.spec._oracle(workload, old)
        program = []
        for name, values in zip(("out_positions", "out_values"), expected):
            for offset, value in enumerate(values):
                program.extend((("const", 1, 0, value), ("store", 1, 0, workload[name] + offset)))
        with mock.patch.object(self.spec.secrets, "token_hex", return_value="ff" * 32):
            self.assertFalse(self.task.evaluate_solution({"workloads": (workload,)}, (program,)).correct)
        program = list(self.task.solve({"workloads": (workload,)})[0])
        program.extend((("const", 1, 0, 0xffffffff), ("store", 1, 0, workload["out_positions"])))
        self.assertFalse(self.task.evaluate_solution({"workloads": (workload,)}, (program,)).correct)

    def test_correctness_inputs_are_all_scored_and_failure_keeps_receipt(self):
        problem = self.task.generate_problem(2)
        outputs = {"reference": self.task.solve(problem), "candidate": (("invalid",),) * 4}
        receipts = []
        result = self.task.evaluate_pair(problem, outputs, record=receipts.append)
        self.assertTrue(result["reference"].correct)
        self.assertFalse(result["candidate"].correct)
        total = 0
        for index, workload in enumerate(problem["workloads"]):
            for trial in range(3):
                memory = self.spec._runtime_memory(workload, receipts[0]["seed"], index, trial)
                total += self.spec.Machine(workload, memory).run(outputs["reference"][index])
        self.assertEqual(result["reference"].score, total / 3)

    def test_optimization_ladder_has_gains_beyond_vectorization(self):
        for seed in (8, 91):
            for index, workload in enumerate(self.task.generate_problem(32, seed)["workloads"]):
                memory = self.spec._runtime_memory(workload, "37" * 32, index, 0)
                scalar = self.spec.Machine(workload, memory).run(self.spec._compile(workload))
                vector = self.spec.Machine(workload, memory).run(compile_workload(workload))
                interleaved = self.spec.Machine(workload, memory).run(
                    compile_workload(workload, streams=4, scheduled=True))
                self.assertLess(vector, scalar / 2)
                self.assertLess(interleaved, vector)
                if workload["family"] == "shared":
                    cached = self.spec.Machine(workload, memory).run(
                        compile_workload(workload, streams=4, cached=True, scheduled=True))
                    self.assertLess(cached, interleaved)

    def test_harness_cycle_score_and_replay_at_declared_sizes(self):
        for size in self.task.grading_cases:
            result = run_task(ROOT / "tasks/simd_traversal_kernel", n=size, seed=123, samples=1)
            self.assertTrue(result.correct)
            self.assertEqual(result.metric_unit, "cycles")
            self.assertEqual(result.speedup, 1.0)
            replay = run_task(ROOT / "tasks/simd_traversal_kernel", n=size, seed=123, samples=1,
                              verification_replay=[result.samples[0].verification])
            self.assertEqual(result, replay)

    def test_register_reuse_adds_streams_with_workload_tradeoffs(self):
        wins, losses = set(), set()
        for size in (32, 64, 128):
            for index, workload in enumerate(self.task.generate_problem(size, 17)["workloads"]):
                memory = self.spec._runtime_memory(workload, "37" * 32, index, 0)
                expected = self.spec._oracle(workload, memory)
                baseline = self.spec.Machine(workload, memory).run(
                    compile_workload(workload, streams=4, scheduled=True))
                for cached in (False, True):
                    machine = self.spec.Machine(workload, memory)
                    cycles = machine.run(compile_workload(workload, streams=4 if cached else 5,
                                                         compact=True, cached=cached))
                    actual = tuple(tuple(machine.memory[workload[key]:workload[key] + workload["batch"]])
                                   for key in ("out_positions", "out_values"))
                    self.assertEqual(actual, expected)
                    self.assertLessEqual(machine.stats["scratch_registers"], 20)
                    if not cached:
                        if cycles < baseline:
                            wins.add(workload["family"])
                        if cycles > baseline:
                            losses.add(workload["family"])
        self.assertIn("crowded", wins)
        self.assertTrue(losses, "extra streams should expose partially filled tile costs")

    def test_compact_cached_streams_improve_shared_fixture(self):
        workload = self.task.generate_problem(32, 17)["workloads"][0]
        memory = self.spec._runtime_memory(workload, "37" * 32, 0, 0)
        costs = []
        for compact in (False, True):
            machine = self.spec.Machine(workload, memory)
            costs.append(machine.run(compile_workload(workload, streams=4, cached=True,
                                                       scheduled=True, compact=compact)))
            actual = tuple(tuple(machine.memory[workload[key]:workload[key] + workload["batch"]])
                           for key in ("out_positions", "out_values"))
            self.assertEqual(actual, self.spec._oracle(workload, memory))
        self.assertLess(costs[1], costs[0])


if __name__ == "__main__":
    unittest.main()
