"""Oracle, hidden-failure and adversarial checks for the layout-aware compiler."""

import copy
import pathlib
import sys
import unittest
from unittest import mock

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = ("chain", "shared_layout", "pressure", "bandwidth")


class LayoutCompilerTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(ROOT / "tasks/layout_aware_pipeline_compiler")
        self.helpers = sys.modules[type(self.task).__module__]

    def score(self, workload, submission):
        return self.task.evaluate_solution({"workloads": (workload,)}, (submission,))

    def test_output_address_checks_allow_column_major_and_reversed_layouts(self):
        workload = self.task._workload(
            "chain", 4, 4, 0.0, (("x", (8, 8)), ("y", (8, 8))),
            (("out", ("ew", "add", "x", "y")),), ("out",))
        reference = self.task.solve({"workloads": (workload,)})[0]
        base = reference["placements"]["out"][0]
        for placement in ((base, 1, 8), (base + 63, -1, -8)):
            with self.subTest(placement=placement):
                emitter = self.helpers._Emitter(workload["machine"])
                for row in range(8):
                    for column in range(8):
                        emitter.issue(("sload", 0, 0, row * 12 + column))
                        emitter.issue(("sload", 1, 0, 96 + row * 12 + column))
                        emitter.issue(("vadd", 2, 0, 1))
                        address = placement[0] + row * placement[1] + column * placement[2]
                        emitter.issue(("sstore", 2, 0, address))
                self.assertTrue(self.score(workload, {
                    "program": emitter.program(), "placements": {"out": placement},
                }).correct)
        for placement in ((base, 1, 1), (base, 0, 1), (0, -1, 8)):
            with self.subTest(invalid=placement), self.assertRaises(ValueError):
                self.helpers._placements(workload, {"out": placement})

    def test_reduction_columns_compose_with_all_pipeline_operations(self):
        pipeline = (
            ("total", ("reduce", "x")),
            ("added", ("ew", "add", "total", "total")),
            ("product", ("ew", "mul", "added", "total")),
            ("window", ("stencil", "product", (2, 3))),
            ("flipped", ("transpose", "window")),
            ("grand", ("reduce", "flipped")),
            ("identity", ("reduce", "window")),
        )
        workload = self.task._workload(
            "chain", 4, 4, 0.0, (("x", (8, 8)),), pipeline,
            ("total", "added", "product", "window", "flipped", "grand", "identity"))
        self.helpers._check_pipeline(workload)
        inputs = {"x": tuple(tuple(self.helpers.MASK if row == 0 else
                                  (row + 1) * 100 + column for column in range(8))
                             for row in range(8))}
        totals = tuple(sum(row) & self.helpers.MASK for row in inputs["x"])
        windows = tuple((4 * value * value) & self.helpers.MASK for value in totals)
        expected = self.helpers._evaluate(workload, inputs)
        self.assertEqual(expected["total"], tuple((value,) for value in totals))
        self.assertEqual(expected["window"], tuple((value,) for value in windows))
        self.assertEqual(expected["flipped"], (windows,))
        self.assertEqual(expected["grand"], ((sum(windows) & self.helpers.MASK,),))
        self.assertEqual(expected["identity"], expected["window"])
        submission = self.task.solve({"workloads": (workload,)})[0]
        with mock.patch.object(self.helpers, "_runtime_inputs", return_value=inputs):
            self.assertTrue(self.score(workload, submission).correct)

    def test_single_column_reference_does_not_transfer_neighboring_cells(self):
        workload = self.task._workload(
            "chain", 4, 4, 0.0, (("x", (1, 1)),),
            (("out", ("ew", "add", "x", "x")),), ("out",))
        submission = self.task.solve({"workloads": (workload,)})[0]
        with mock.patch.object(self.helpers, "_runtime_inputs", return_value={"x": ((7,),)}):
            self.assertTrue(self.score(workload, submission).correct)

    def test_generator_covers_every_family_and_consumes_every_stage(self):
        for size in (self.task.default_n, self.task.grading_cases[-1]):
            problem = self.task.generate_problem(size, 3)
            self.assertEqual([workload["family"] for workload in problem["workloads"]],
                             list(FAMILIES))
            for workload in problem["workloads"]:
                consumed = set()
                for _, expression in workload["pipeline"]:
                    consumed.update(expression[2:] if expression[0] == "ew"
                                    else expression[1:2])
                outputs = set(workload["outputs"])
                for name, _ in workload["pipeline"]:
                    self.assertIn(name, consumed | outputs,
                                  f"{workload['family']}/{name} is dead code")

    def test_reference_verifies_and_is_deterministic(self):
        for seed in range(3):
            first = self.task.generate_problem(self.task.default_n, seed)
            second = self.task.generate_problem(self.task.default_n, seed)
            self.assertEqual(first, second)
            pristine = copy.deepcopy(first)
            submission = self.task.solve(first)
            self.assertEqual(first, pristine)
            evaluation = self.task.evaluate_solution(first, submission)
            self.assertTrue(evaluation.correct)
            self.assertGreater(evaluation.score, 0)
            self.assertTrue(self.task.is_solution(first, submission))

    def test_compilation_receives_descriptors_and_draws_no_runtime_values(self):
        with mock.patch.object(self.helpers, "_runtime_inputs",
                               side_effect=AssertionError("data requested during compilation")):
            problem = self.task.generate_problem(8, 0)
            for workload in problem["workloads"]:
                self.assertEqual(set(workload),
                                 {"family", "inputs", "pipeline", "outputs", "machine"})
                for name, shape, address in workload["inputs"]:
                    self.assertIs(type(name), str)
                    self.assertEqual(len(shape), 2)
                    self.assertTrue(all(type(dimension) is int for dimension in shape))
                    self.assertIs(type(address), int)
            # Exercise the candidate-facing entrypoint and its reference callback.
            with mock.patch.object(self.helpers._candidate, "solve",
                                   side_effect=lambda public, reference: reference(public)) as candidate:
                submission = self.task.candidate_solve(problem)
            self.assertEqual(candidate.call_args.args[0], problem)
        self.assertTrue(self.task.evaluate_solution(problem, submission).correct)

    def test_one_compilation_handles_three_runtime_inputs_at_each_grading_size(self):
        for size in self.task.grading_cases:
            problem = self.task.generate_problem(size, 7)
            submission = self.task.solve(problem)
            seen = []

            def inputs(workload, _seed, _index, _trial):
                # Include zeros, large unsigned values and lane-varying values.
                trial = len(seen) % 3
                result = {name: tuple(tuple((0 if trial == 0 else
                                            (self.helpers.MASK if trial == 1 else
                                             (row * columns + column + index)))
                                           for column in range(columns)) for row in range(rows))
                          for index, (name, (rows, columns), _) in enumerate(workload["inputs"])}
                seen.append(result)
                return result

            with mock.patch.object(self.helpers, "_runtime_inputs", side_effect=inputs):
                evaluation = self.task.evaluate_solution(problem, submission)
            self.assertTrue(evaluation.correct)
            self.assertEqual(len(seen), 3 * len(problem["workloads"]))
            once = sum(self.helpers._Simulator(workload, seen[3 * index]).run(item["program"])
                       for index, (workload, item) in enumerate(zip(problem["workloads"], submission)))
            self.assertEqual(evaluation.score, once)
            self.assertEqual(evaluation.score,
                             self.task.evaluate_solution(problem, submission).score)

    def test_precomputed_answers_fail_when_runtime_values_change(self):
        workload = self.task._workload(
            "chain", 4, 4, 0.0, (("x", (8, 8)), ("y", (8, 8))),
            (("out", ("ew", "add", "x", "y")),), ("out",))
        reference = self.task.solve({"workloads": (workload,)})[0]
        first = {name: tuple((value,) * 8 for _ in range(8))
                 for name, value in (("x", 7), ("y", 9))}
        second = {name: tuple((value,) * 8 for _ in range(8))
                  for name, value in (("x", 11), ("y", 13))}
        expected = self.helpers._evaluate(workload, first)["out"]
        emitter = self.helpers._Emitter(workload["machine"])
        base, stride, _ = reference["placements"]["out"]
        for row in range(8):
            for column in range(8):
                emitter.issue(("vconst", 0, expected[row][column]))
                emitter.issue(("sstore", 0, 0, base + row * stride + column))
        submission = {"program": emitter.program(), "placements": reference["placements"]}
        # This is a legal store-only program and is correct on its precomputed data.
        simulator = self.helpers._Simulator(workload, first)
        simulator.run(submission["program"])
        self.helpers._check_outputs(workload, first, submission["placements"], simulator)
        # The evaluator reuses the same instructions, rather than recompiling.
        with mock.patch.object(self.helpers, "_runtime_inputs", side_effect=(first, second, first)) as draw:
            self.assertFalse(self.score(workload, submission).correct)
        self.assertEqual(draw.call_count, 3)

    def test_lazy_program_objects_are_rejected_before_runtime_data_is_drawn(self):
        class LazyProgram(list):
            def __iter__(self):
                raise AssertionError("candidate code ran during verification")

        problem = self.task.generate_problem(8, 0)
        submission = list(self.task.solve(problem))
        submission[0]["program"] = LazyProgram(submission[0]["program"])
        with mock.patch.object(self.helpers, "_runtime_inputs") as draw:
            self.assertFalse(self.task.evaluate_solution(problem, submission).correct)
        draw.assert_not_called()

    def test_paired_receipt_precedes_simulation_and_replays_the_same_tensors(self):
        problem = self.task.generate_problem(8, 2)
        outputs = {role: self.task.solve(problem) for role in ("reference", "candidate")}
        receipts = []
        seen = []
        original = self.helpers._Simulator

        def simulator(workload, inputs):
            self.assertEqual(len(receipts), 1, "receipt must be saved before simulation")
            seen.append(copy.deepcopy(inputs))
            return original(workload, inputs)

        with mock.patch.object(self.helpers, "_Simulator", side_effect=simulator):
            evaluations = self.task.evaluate_pair(problem, outputs, record=receipts.append)
        self.assertTrue(all(value.correct for value in evaluations.values()))
        self.assertEqual(seen[:12], seen[12:])
        receipt = receipts[0]
        self.assertEqual(len(receipt["seed"]), 64)
        self.assertEqual(receipt["program_sha256"]["reference"],
                         receipt["program_sha256"]["candidate"])
        replayed = []
        with mock.patch.object(self.helpers.secrets, "token_hex",
                               side_effect=AssertionError("replay requested fresh entropy")):
            repeated = self.task.evaluate_pair(problem, outputs, replay=receipt,
                                               record=replayed.append)
        self.assertEqual(repeated, evaluations)
        self.assertEqual(replayed, receipts)

    def test_both_programs_are_frozen_before_verification_seed_is_drawn(self):
        problem = self.task.generate_problem(8, 0)
        outputs = {role: self.task.solve(problem) for role in ("reference", "candidate")}

        def entropy(byte_count):
            self.assertEqual(byte_count, 32)
            for output in outputs.values():
                output[0]["program"] = ()
            return "01" * 32

        with mock.patch.object(self.helpers.secrets, "token_hex", side_effect=entropy):
            evaluations = self.task.evaluate_pair(problem, outputs)
        self.assertTrue(all(value.correct for value in evaluations.values()))

    def test_replay_rejects_changed_program_problem_or_generator_before_tensor_expansion(self):
        problem = self.task.generate_problem(8, 0)
        outputs = {role: self.task.solve(problem) for role in ("reference", "candidate")}
        receipts = []
        self.task.evaluate_pair(problem, outputs, record=receipts.append)
        for changed in ("program", "problem", "generator", "seed"):
            modified_problem = copy.deepcopy(problem)
            modified_outputs = copy.deepcopy(outputs)
            receipt = copy.deepcopy(receipts[0])
            if changed == "program":
                modified_outputs["candidate"][0]["program"] = ()
            elif changed == "problem":
                modified_problem["workloads"][0]["machine"]["scratch"] += 1
            elif changed == "generator":
                receipt["generator_version"] = "unknown"
            else:
                receipt["seed"] = "invalid"
            with self.subTest(changed=changed), \
                    mock.patch.object(self.helpers, "_runtime_inputs") as expand, \
                    mock.patch.object(self.helpers, "_Simulator") as simulator, \
                    self.assertRaises(ValueError):
                self.task.evaluate_pair(modified_problem, modified_outputs, replay=receipt)
            expand.assert_not_called()
            simulator.assert_not_called()
        receipt = dict(receipts[0], tensor_sha256="0" * 64)
        with mock.patch.object(self.helpers, "_Simulator") as simulator, self.assertRaises(ValueError):
            self.task.evaluate_pair(problem, outputs, replay=receipt)
        simulator.assert_not_called()

    def test_failed_program_receipt_is_recorded_and_replayable(self):
        problem = self.task.generate_problem(8, 0)
        outputs = {role: self.task.solve(problem) for role in ("reference", "candidate")}
        outputs["candidate"][0]["program"] = ()
        receipts = []
        evaluations = self.task.evaluate_pair(problem, outputs, record=receipts.append)
        self.assertTrue(evaluations["reference"].correct)
        self.assertFalse(evaluations["candidate"].correct)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(self.task.evaluate_pair(problem, outputs, replay=receipts[0]), evaluations)

    def test_lazy_candidate_does_not_skip_reference_verification(self):
        class LazyProgram(list):
            def __iter__(self):
                raise AssertionError("candidate callback executed")

        problem = self.task.generate_problem(8, 0)
        reference = self.task.solve(problem)
        candidate = copy.deepcopy(reference)
        candidate[0]["program"] = LazyProgram(candidate[0]["program"])
        receipts = []
        evaluations = self.task.evaluate_pair(
            problem, {"reference": reference, "candidate": candidate}, record=receipts.append)
        self.assertTrue(evaluations["reference"].correct)
        self.assertFalse(evaluations["candidate"].correct)
        self.assertIsNone(receipts[0]["program_sha256"]["candidate"])

    def test_rejects_wrong_values_conflicts_and_out_of_range_programs(self):
        problem = self.task.generate_problem(self.task.default_n, 0)
        workload = problem["workloads"][0]
        reference = self.task.solve({"workloads": (workload,)})[0]

        dropped = list(reference["program"])
        # Remove the last write of the program: the tail of the output is unset.
        for index in range(len(dropped) - 1, -1, -1):
            if any(instruction[0] in ("vstore", "sstore") for instruction in dropped[index]):
                dropped[index] = tuple(instruction for instruction in dropped[index]
                                       if instruction[0] not in ("vstore", "sstore"))
                break
        self.assertFalse(self.score(workload, {"program": tuple(dropped),
                                               "placements": reference["placements"]}).correct)

        overlapping = dict(reference["placements"])
        first_name = workload["outputs"][0]
        base, row_stride, column_stride = overlapping[first_name]
        overlapping[first_name] = (base, row_stride, column_stride)
        if len(workload["outputs"]) > 1:
            other = workload["outputs"][1]
            overlapping[other] = (base, overlapping[other][1], overlapping[other][2])
            self.assertFalse(self.score(workload, {"program": reference["program"],
                                                   "placements": overlapping}).correct)

        outside = dict(reference["placements"])
        outside[first_name] = (workload["machine"]["memory"], row_stride, column_stride)
        self.assertFalse(self.score(workload, {"program": reference["program"],
                                               "placements": outside}).correct)

        self.assertFalse(self.score(workload, {"program": (), "placements": reference["placements"]}).correct)
        self.assertFalse(self.task.evaluate_solution(problem, [(None, None)] * len(problem["workloads"])).correct)
        self.assertFalse(self.task.is_solution(problem, "not a submission"))

    def test_slot_hazards_and_capacity_limits_are_enforced(self):
        problem = self.task.generate_problem(self.task.default_n, 0)
        workload = problem["workloads"][0]
        lanes = workload["machine"]["lanes"]
        # Reading a slot in the next cycle after its producer still counts as
        # reading before the result exists.
        program = (("vload", 0, 0), (("vadd", 1, 0, 0),))
        self.assertFalse(self.score(workload, {"program": program,
                                               "placements": self.task.solve({"workloads": (workload,)})[0]["placements"]}).correct)
        # Too many same-resource instructions in one packet.
        crowded = (tuple(("vload", 0, 0) for _ in range(workload["machine"]["load"] + 1)),)
        self.assertFalse(self.score(workload, {"program": crowded,
                                               "placements": self.task.solve({"workloads": (workload,)})[0]["placements"]}).correct)
        # Bank conflict: two transfers that map to the same bank in one cycle.
        banks = workload["machine"]["banks"]
        first = 0
        second = first + lanes * banks
        conflict = ((("vload", 0, first), ("vload", 1, second)),)
        if second + lanes <= workload["machine"]["memory"]:
            self.assertFalse(self.score(workload, {"program": conflict,
                                                   "placements": self.task.solve({"workloads": (workload,)})[0]["placements"]}).correct)

    def test_fusing_an_elementwise_chain_beats_materializing_it(self):
        """The measured accessible improvement, kept as a regression guard."""
        # This hand-written lowering targets this specific program. Keep the
        # fixture explicit; generated programs must be free to vary by seed.
        pipeline = (("s0", ("ew", "add", "x", "y")),
                    ("s1", ("ew", "mul", "s0", "x")),
                    ("s2", ("ew", "add", "s1", "y")),
                    ("s3", ("stencil", "s2", (1, 2, 1))),
                    ("s4", ("ew", "mul", "s3", "s3")))
        workload = self.task._workload(
            "chain", 8, 4, 0.25,
            (("x", (32, 16)), ("y", (32, 16))), pipeline, ("s4",))
        reference = self.score(workload, self.task.solve({"workloads": (workload,)})[0])
        program, placements = _fused_chain(self.task, workload)
        fused = self.score(workload, {"program": program, "placements": placements})
        self.assertTrue(fused.correct)
        self.assertLess(fused.score, reference.score / 2.0)


def _fused_chain(task, workload):
    """Row-wise lowering of the chain family that never materializes intermediates."""
    import importlib.util
    import sys

    module_name = "speedupmark_layout_helpers"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "tasks/layout_aware_pipeline_compiler/task_spec.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    helpers = sys.modules[module_name]

    machine = workload["machine"]
    lanes = machine["lanes"]
    shapes = helpers._shapes(workload)
    rows, columns = shapes["x"]
    address = {name: address for name, _, address in workload["inputs"]}
    x_stride = helpers._row_stride(shapes["x"])
    y_stride = helpers._row_stride(shapes["y"])
    out = workload["outputs"][0]
    out_stride = helpers._row_stride(shapes[out])
    out_base = rows * x_stride + rows * y_stride
    emitter = helpers._Emitter(machine)
    emitter.issue(("vconst", 7, 0))
    for row in range(rows):
        for vector in range(2):
            offset = vector * lanes
            emitter.issue(("vload", vector, address["x"] + row * x_stride + offset))
            emitter.issue(("vload", 2 + vector, address["y"] + row * y_stride + offset))
            emitter.issue(("vadd", 4 + vector, vector, 2 + vector))
            emitter.issue(("vmul", 4 + vector, 4 + vector, vector))
            emitter.issue(("vadd", 4 + vector, 4 + vector, 2 + vector))
        for vector in range(2):
            current = 4 + vector
            following = 5 if vector == 0 else 7
            emitter.issue(("vadd", 6, current, 7))
            emitter.issue(("vperm", 2, current, following, tuple(range(1, lanes + 1))))
            emitter.issue(("vconst", 3, 2))
            emitter.issue(("vmadd", 6, 2, 3, 6))
            emitter.issue(("vperm", 2, current, following, tuple(range(2, lanes + 2))))
            emitter.issue(("vadd", 6, 6, 2))
            emitter.issue(("vmul", 6, 6, 6))
            emitter.issue(("vstore", 6, out_base + row * out_stride + vector * lanes))
    return emitter.program(), {out: (out_base, out_stride, 1)}


if __name__ == "__main__":
    unittest.main()
