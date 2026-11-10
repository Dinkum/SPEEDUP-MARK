"""Every wall-clock sample must run long enough for its ratio to mean something.

A sample is one workload family at one size: the summed reference time of a
task's ``timing_batch`` problems. When that sum is microseconds, the measured
speedup is timer noise plus fixed call overhead, and it carries the same weight
in the geometric mean as every other sample. Sizes and batches are therefore
chosen so each family clears a floor, and seed-selected families must be
declared so the harness samples all of them.

Run ``python3 -m tests.test_reference_floor`` to print the measured table when
sizing a task.
"""

import copy
import math
import re
import time
import unittest

from speedupmark.harness import discover_tasks, load_task
from speedupmark.task import family_period, freeze_output, grading_cases, timing_batch


# Sizes target at least 20 ms on the development laptop. The asserted floor is
# lower so a faster host does not fail the suite; load can only lengthen a
# measurement, so a lower bound is not flaky.
FLOOR_MS = 5.0

# These need the pinned numerical environment and have not been sized against
# the floor. Remove a name once its families are measured there.
UNMEASURED = frozenset({
    "delaunay", "earth_movers_distance", "group_lasso", "low_rank_approximation",
    "ode_stiff_robertson", "pde_burgers1d", "rbf_interpolation", "robust_kalman_filter",
    "rocket_landing_optimization", "sinkhorn", "smallest_eigenvalues_sparse_spd",
    "spectral_radius_matrix_completion",
})


def wall_clock_tasks():
    for path in discover_tasks():
        if path.name in UNMEASURED:
            continue
        task = load_task(path, with_candidate=False)
        if not hasattr(task, "evaluate_solution") and not hasattr(task, "evaluate_pair"):
            yield task


def reference_ms(task, size, residue):
    """Summed reference time of the batch the harness would form for one family."""
    period, total = family_period(task), 0.0
    for member in range(timing_batch(task)):
        problem = copy.deepcopy(task.generate_problem(size, residue + member * period))
        start = time.perf_counter_ns()
        freeze_output(task.solve(problem))
        total += (time.perf_counter_ns() - start) / 1_000_000
    return total


def sizes(task):
    return sorted({getattr(task, "default_n", 1000), *grading_cases(task)})


class ReferenceFloorTests(unittest.TestCase):
    def test_every_family_at_every_size_clears_the_floor(self):
        for task in wall_clock_tasks():
            for size in sizes(task):
                for residue in range(family_period(task)):
                    with self.subTest(task=task.name, n=size, residue=residue):
                        self.assertGreaterEqual(reference_ms(task, size, residue), FLOOR_MS)

    def test_declared_period_covers_every_seed_modulus_in_the_spec(self):
        for path in discover_tasks():
            source = (path / "task_spec.py").read_text()
            moduli = {int(value) for value in re.findall(r"seed\s*%\s*(\d+)", source)}
            declared = re.findall(r"^\s+family_period\s*=\s*(\d+)\s*$", source, re.MULTILINE)
            with self.subTest(task=path.name):
                self.assertEqual([int(value) for value in declared],
                                 [math.lcm(*moduli)] if moduli else [])


if __name__ == "__main__":
    for task in wall_clock_tasks():
        for size in sizes(task):
            row = [reference_ms(task, size, residue) for residue in range(family_period(task))]
            flag = "  LOW" if min(row) < 20 else ""
            print(f"{task.name:40} n={size:<7} batch={timing_batch(task):<4} "
                  + " ".join(f"{value:8.2f}" for value in row) + flag, flush=True)
