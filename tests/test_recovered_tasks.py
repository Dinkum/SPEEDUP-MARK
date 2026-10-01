import itertools
import math
import pathlib
import unittest

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]


def task(slug):
    return load_task(ROOT / "tasks" / slug)


class RecoveredTaskTests(unittest.TestCase):
    def test_multi_dim_knapsack_matches_subset_enumeration(self):
        benchmark = task("multi_dim_knapsack")
        for seed in range(5):
            problem = benchmark.generate_problem(11, seed)
            answer = benchmark.solve(problem)
            candidate_answer = benchmark.candidate_solve(problem)
            optimum = 0
            for mask in range(1 << len(problem["profits"])):
                used = [0, 0, 0]
                profit = 0
                for item in range(len(problem["profits"])):
                    if mask >> item & 1:
                        profit += problem["profits"][item]
                        for d in range(3):
                            used[d] += problem["weights"][item][d]
                if all(used[d] <= problem["capacities"][d] for d in range(3)):
                    optimum = max(optimum, profit)
            self.assertEqual(answer["profit"], optimum)
            self.assertEqual(candidate_answer["profit"], optimum)
            self.assertTrue(benchmark.is_solution(problem, answer))
            self.assertFalse(benchmark.is_solution(problem, {"profit": True}))

    def test_tsp_matches_permutation_oracle(self):
        benchmark = task("tsp")
        for n in range(3, 8):
            for seed in range(2):
                problem = benchmark.generate_problem(n, seed)
                answer = benchmark.solve(problem)
                distances = problem["distances"]
                optimum = min(
                    sum(distances[a][b] for a, b in zip((0, *middle, 0), (*middle, 0)))
                    for middle in itertools.permutations(range(1, n))
                )
                self.assertEqual(answer["cost"], optimum)
                candidate_answer = benchmark.candidate_solve(problem)
                self.assertEqual(candidate_answer["cost"], optimum)
                self.assertTrue(benchmark.is_solution(problem, candidate_answer))
                self.assertTrue(benchmark.is_solution(problem, answer))
                broken = {"tour": answer["tour"][:-2] + [0, answer["tour"][-2], 0], "cost": answer["cost"]}
                self.assertFalse(benchmark.is_solution(problem, broken))

    def test_pagerank_sparse_candidate_has_residual_certificate(self):
        benchmark = task("pagerank")
        for seed in range(3):
            problem = benchmark.generate_problem(48, seed)
            answer = benchmark.candidate_solve(problem)
            self.assertTrue(benchmark.is_solution(problem, answer))
            bad = {"scores": [1.0 / len(answer["scores"])] * len(answer["scores"])}
            self.assertFalse(benchmark.is_solution(problem, bad))
            self.assertAlmostEqual(math.fsum(answer["scores"]), 1.0, places=10)

    def test_job_shop_reference_is_feasible_and_optimal(self):
        benchmark = task("job_shop_scheduling")
        problem = benchmark.generate_problem(3, 4)
        answer = benchmark.solve(problem)
        self.assertTrue(benchmark.is_solution(problem, answer))
        delayed = {
            "start_times": [[time + 1 for time in row] for row in answer["start_times"]],
            "makespan": answer["makespan"] + 1,
        }
        self.assertFalse(benchmark.is_solution(problem, delayed))
        overlapping = {
            "start_times": [[0] * problem["machines"] for _ in problem["jobs"]],
            "makespan": max(duration for job in problem["jobs"] for _, duration in job),
        }
        self.assertFalse(benchmark.is_solution(problem, overlapping))

    def test_declared_grading_cases_pass_their_verifiers(self):
        for slug in ("multi_dim_knapsack", "tsp", "pagerank", "job_shop_scheduling"):
            benchmark = task(slug)
            for size in benchmark.grading_cases:
                with self.subTest(task=slug, size=size):
                    problem = benchmark.generate_problem(size, 3)
                    answer = benchmark.solve(problem)
                    self.assertTrue(benchmark.is_solution(problem, answer))


if __name__ == "__main__":
    unittest.main()
