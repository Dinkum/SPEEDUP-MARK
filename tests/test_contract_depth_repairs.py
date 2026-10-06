"""Oracle independence, variable programs, and cheap optimality certificates."""

import copy
import importlib.util
import itertools
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from speedupmark.harness import load_task


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL = all(importlib.util.find_spec(name) for name in ('numpy', 'scipy'))


def task(name):
    return load_task(ROOT / 'tasks' / name)


def corrupt(value):
    """Change a leaf while preserving the completed-output representation."""
    if type(value) is int:
        return value + 1
    if type(value) is bytes:
        return value + b'\x00'
    if type(value) is str:
        return value + ':wrong'
    if type(value) in (tuple, list):
        values = list(value)
        for index, child in enumerate(values):
            if child not in ((), []):
                values[index] = corrupt(child)
                return tuple(values) if type(value) is tuple else values
    raise ValueError('fixture must contain a mutable semantic leaf')


class ContractDepthRepairTests(unittest.TestCase):
    def test_verifiers_survive_reference_corruption(self):
        cases = (
            ('adaptive_query_engine', 64, '_answer'),
            ('ranked_bpe_tokenization', 200, '_encode'),
            ('temporal_asof_join', 80, '_join'),
            ('incremental_spreadsheet_recalculation', 80, '_run'),
            ('dynamic_document_search', 80, '_search'),
            ('multi_literal_replacement', 200, '_replace'),
            ('durable_log_recovery', 80, '_recover'),
            ('dynamic_exact_ray_queries', 24, '_query'),
            ('multi_pattern_matching', 128, '_scan'),
            ('out_of_order_session_windows', 80, '_sessionize'),
        )
        for name, size, reference in cases:
            with self.subTest(task=name):
                benchmark = task(name)
                problem = benchmark.generate_problem(size, 5)
                good = benchmark.solve(problem)
                bad = corrupt(good)
                original = copy.deepcopy(problem)
                with patch.dict(benchmark.solve.__globals__, {reference: lambda _: bad}):
                    self.assertEqual(benchmark.solve(problem), bad)
                    self.assertTrue(benchmark.is_solution(problem, good))
                    self.assertFalse(benchmark.is_solution(problem, bad))
                self.assertEqual(problem, original)

    def test_query_grammar_and_names_change_within_every_family(self):
        benchmark = task('adaptive_query_engine')
        programs, shapes = {}, {}
        def shape(node):
            if node[0] == 'scan':
                return ('scan',)
            if node[0] == 'join':
                return ('join', shape(node[1]), shape(node[2]))
            return node[0], shape(node[1])
        for seed in range(12):
            problem = benchmark.generate_problem(64, seed)
            self.assertEqual(problem, benchmark.generate_problem(64, seed))
            self.assertTrue(benchmark.is_solution(problem, benchmark.solve(problem)))
            for family in problem['families']:
                name = family['family']
                programs.setdefault(name, set()).add(family['queries'])
                shapes.setdefault(name, set()).add(tuple(shape(q) for q in family['queries']))
                self.assertTrue(all(name.startswith('t') for name in family['tables']))
                self.assertTrue(all(column.startswith('c') for table in family['tables'].values()
                                    for column in table['columns']))
        self.assertTrue(all(len(variants) == 12 for variants in programs.values()))
        self.assertTrue(all(len(variants) > 6 for variants in shapes.values()))

    def test_discrete_log_matches_exhaustive_small_groups(self):
        benchmark = task('discrete_log')
        for modulus, base in ((7, 3), (17, 3), (23, 5), (97, 5)):
            for exponent in range(modulus - 1):
                problem = {'p': modulus, 'g': base, 'h': pow(base, exponent, modulus)}
                self.assertEqual(benchmark.solve(problem), {'x': exponent})
                self.assertTrue(benchmark.is_solution(problem, {'x': exponent}))
                self.assertFalse(benchmark.is_solution(problem, {'x': (exponent + 1) % (modulus - 1)}))

    def test_primality_is_exact_and_not_a_product_only_check(self):
        benchmark = task('integer_factorization')
        prime = benchmark.solve.__globals__['prime']
        for value in range(10000):
            expected = value >= 2 and all(value % d for d in range(2, math.isqrt(value) + 1))
            self.assertEqual(prime(value), expected, value)
        for composite in (341550071728321, 3825123056546413051, 2**64 - 1):
            self.assertFalse(prime(composite))
        self.assertTrue(prime(2**64 - 59))
        for seed in range(8):
            problem = benchmark.generate_problem(24, seed)
            divisor = next(p for p in range(2, math.isqrt(problem['composite']) + 1)
                           if problem['composite'] % p == 0)
            self.assertEqual(benchmark.solve(problem), {'p': divisor, 'q': problem['composite'] // divisor})

    def test_large_tsp_dual_bound_is_checked_not_trusted(self):
        benchmark = task('asymmetric_tsp')
        for n in (48, 96):
            for seed in range(3):
                problem = benchmark.generate_problem(n, seed)
                answer = benchmark.solve(problem)
                self.assertTrue(benchmark.is_solution(problem, answer))
                self.assertTrue(benchmark.is_solution(problem, benchmark.candidate_solve(problem)))
                # Keep a feasible tour but swap two cities; most such tours
                # violate the bound because they introduce positive slack.
                tour = answer['tour']
                bad = next(candidate for i in range(1, n - 1)
                           if (candidate := tour[:i] + [tour[i+1], tour[i]] + tour[i+2:])
                           and sum(problem['distances'][a][b] for a, b in zip(candidate, candidate[1:])) > answer['cost'])
                bad_cost = sum(problem['distances'][a][b] for a, b in zip(bad, bad[1:]))
                self.assertFalse(benchmark.is_solution(problem, {'tour': bad, 'cost': bad_cost}))
                # A broken assignment solver cannot invent a certificate for a
                # suboptimal tour: dual feasibility must still hold everywhere.
                fake_rows = [bad_cost] + [0] * (n - 1)
                with patch.dict(benchmark.solve.__globals__,
                                {'_assignment_dual': lambda _: ([0] * n, fake_rows, [0] * n)}):
                    self.assertFalse(benchmark.is_solution(problem, {'tour': bad, 'cost': bad_cost}))

    def test_assignment_bound_agrees_with_small_exhaustive_tours_when_tight(self):
        benchmark = task('asymmetric_tsp')
        dual = benchmark.solve.__globals__['_dual_certifies']
        for seed in range(3):
            # Restrict a large constructed matrix to a small planted-tight
            # fixture, so exact enumeration tests the dual certificate itself.
            costs = [[0 if i == j else (i+1) + (j+2) + (0 if j == (i+1) % 6 else seed+1)
                      for j in range(6)] for i in range(6)]
            optimum = min(sum(costs[a][b] for a, b in zip((0, *order), (*order, 0)))
                          for order in itertools.permutations(range(1, 6)))
            self.assertTrue(dual(costs, optimum))
            self.assertFalse(dual(costs, optimum + 1))

    def test_bounded_search_verifiers_do_not_call_reference_algorithms(self):
        for name, size, helper in [('capacitated_facility_location', 6, '_solve_dp'),
                                   ('vehicle_routing', 6, '_solve_vrp')]:
            benchmark = task(name)
            problem = benchmark.generate_problem(size, 3)
            good = benchmark.solve(problem)
            with patch.dict(benchmark.solve.__globals__, {helper: lambda *a: (_ for _ in ()).throw(AssertionError('reference called'))}):
                self.assertTrue(benchmark.is_solution(problem, good))

    def test_facility_checker_matches_all_small_feasible_assignments(self):
        benchmark = task('capacitated_facility_location')
        problem = {'fixed_costs': [3, 7, 4], 'capacities': [4, 5, 3],
                   'demands': [2, 3, 1],
                   'transportation_costs': [[1, 8, 2], [3, 1, 4], [6, 2, 1]]}
        feasible = []
        for owners in itertools.product(range(3), repeat=3):
            assignment = [[int(owner == i) for owner in owners] for i in range(3)]
            if any(sum(d * bit for d, bit in zip(problem['demands'], row)) > capacity
                   for row, capacity in zip(assignment, problem['capacities'])):
                continue
            status = [any(row) for row in assignment]
            cost = sum(f for f, opened in zip(problem['fixed_costs'], status) if opened)
            cost += sum(c * bit for costs, row in zip(problem['transportation_costs'], assignment)
                        for c, bit in zip(costs, row))
            feasible.append({'objective_value': cost, 'facility_status': status, 'assignments': assignment})
        optimum = min(answer['objective_value'] for answer in feasible)
        for answer in feasible:
            self.assertEqual(benchmark.is_solution(problem, answer), answer['objective_value'] == optimum)

    @unittest.skipUnless(NUMERICAL, 'requires pinned numerical dependencies')
    def test_transport_dual_rejects_corrupted_primal_reference(self):
        benchmark = task('earth_movers_distance')
        problem = {'source_weights': [0.5, 0.5], 'target_weights': [0.5, 0.5],
                   'cost_matrix': [[0.0, 4.0], [4.0, 0.0]]}
        good = {'transport_plan': [[0.5, 0.0], [0.0, 0.5]]}
        bad = {'transport_plan': [[0.0, 0.5], [0.5, 0.0]]}
        with patch.dict(benchmark.solve.__globals__, {'_optimal_cost': lambda *a: (4.0, bad['transport_plan'])}):
            self.assertTrue(benchmark.is_solution(problem, good))
            self.assertFalse(benchmark.is_solution(problem, bad))


if __name__ == '__main__':
    unittest.main()
