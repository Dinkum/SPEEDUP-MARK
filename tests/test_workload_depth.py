"""Workload diversity, independent oracles, and bounded numerical validity."""

import collections
import importlib.util
import itertools
import pathlib
import random
import unittest

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]
NUMERICAL = all(importlib.util.find_spec(name) for name in ('numpy', 'scipy', 'cvxpy'))


def task(name):
    return load_task(ROOT / 'tasks' / name)


def connected(graph):
    neighbors = [set() for _ in graph['labels']]
    for a, b in graph['edges']:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen, pending = {0}, [0]
    while pending:
        for vertex in neighbors[pending.pop()] - seen:
            seen.add(vertex)
            pending.append(vertex)
    return len(seen) == len(neighbors)


class WorkloadDepthTests(unittest.TestCase):
    def test_isomorphism_real_sizes_connected_pairs_and_small_oracle(self):
        benchmark = task('labeled_graph_isomorphism')
        for n in (6, 10, 20):
            for seed in range(12):
                problem = benchmark.generate_problem(n, seed)
                self.assertEqual(problem, benchmark.generate_problem(n, seed))
                self.assertEqual(len(problem['left']['labels']), n)
                self.assertEqual(len(problem['right']['labels']), n)
                for graph in problem.values():
                    self.assertTrue(connected(graph))
                degrees = []
                for graph in problem.values():
                    counts = collections.Counter(v for edge in graph['edges'] for v in edge)
                    degrees.append(sorted(counts.values()))
                self.assertEqual(degrees[0], degrees[1])
                self.assertEqual(sorted(problem['left']['labels']), sorted(problem['right']['labels']))
                answer = benchmark.solve(problem)
                self.assertTrue(benchmark.is_solution(problem, answer))
                self.assertEqual(answer is None, seed % 3 == 2)
                if n == 6:
                    left, right = problem['left'], problem['right']
                    right_edges = set(right['edges'])
                    exists = any(
                        all(left['labels'][i] == right['labels'][p[i]] for i in range(n))
                        and {tuple(sorted((p[a], p[b]))) for a, b in left['edges']} == right_edges
                        for p in itertools.permutations(range(n)))
                    self.assertEqual(exists, answer is not None)

    def test_symmetric_coloring_varies_graph_and_optimum(self):
        benchmark = task('graph_coloring')
        graphs, optima = set(), set()
        for seed in range(2, 50, 4):
            matrix = benchmark.generate_problem(14, seed)
            graphs.add(tuple(tuple(row) for row in matrix))
            answer = benchmark.solve(matrix)
            self.assertTrue(benchmark.is_solution(matrix, answer))
            optima.add(len(set(answer)))
            # At least one false-twin pair survives the blow-up construction.
            self.assertLess(len({tuple(row) for row in matrix}), len(matrix))
        self.assertGreater(len(graphs), 8)
        self.assertGreater(len(optima), 1)

    def test_lattice_routing_changes_distances_and_costs(self):
        benchmark = task('vehicle_routing')
        distances, costs = set(), set()
        for seed in range(2, 20, 3):
            problem = benchmark.generate_problem(8, seed)
            distances.add(tuple(tuple(row) for row in problem['D']))
            routes = benchmark.solve(problem)
            self.assertTrue(benchmark.is_solution(problem, routes))
            costs.add(sum(problem['D'][a][b] for route in routes for a, b in zip(route, route[1:])))
        self.assertEqual(len(distances), 6)
        self.assertGreater(len(costs), 1)

    def test_compiler_changes_programs_and_keeps_every_stage_live(self):
        benchmark = task('layout_aware_pipeline_compiler')
        programs = collections.defaultdict(set)
        for seed in range(8):
            problem = benchmark.generate_problem(16, seed)
            self.assertEqual(problem, benchmark.generate_problem(16, seed))
            self.assertTrue(benchmark.is_solution(problem, benchmark.solve(problem)))
            for workload in problem['workloads']:
                programs[workload['family']].add(workload['pipeline'])
                consumed = set(workload['outputs'])
                for _, expression in workload['pipeline']:
                    consumed.update(expression[2:] if expression[0] == 'ew' else expression[1:2])
                self.assertTrue(all(name in consumed for name, _ in workload['pipeline']))
        self.assertEqual(len(programs), 4)
        self.assertTrue(all(len(variants) >= 6 for variants in programs.values()))

    def test_bpe_varied_tables_match_independent_one_pair_oracle(self):
        benchmark = task('ranked_bpe_tokenization')
        tables = set()
        for seed in range(12):
            problem = benchmark.generate_problem(220, seed)
            tables.add(problem['merges'])
            for rank, pair in enumerate(problem['merges']):
                self.assertTrue(all(0 <= token < 256 + rank for token in pair))
            tokens = list(problem['data'])
            ranks = {pair: rank for rank, pair in enumerate(problem['merges'])}
            while True:
                matches = [(ranks[pair], i) for i in range(len(tokens) - 1)
                           if (pair := tuple(tokens[i:i + 2])) in ranks]
                if not matches:
                    break
                rank, position = min(matches)
                tokens[position:position + 2] = [256 + rank]
            self.assertEqual(benchmark.solve(problem), tuple(tokens))
            # Consistent byte relabeling preserves ranked-tokenization semantics.
            permutation = list(range(256))
            random.Random(seed).shuffle(permutation)
            rename = lambda t: permutation[t] if t < 256 else t
            renamed = {'data': bytes(rename(b) for b in problem['data']),
                       'merges': tuple(tuple(rename(t) for t in pair) for pair in problem['merges'])}
            self.assertEqual(benchmark.solve(renamed), tuple(rename(t) for t in tokens))
        self.assertEqual(len(tables), 12)

    def test_bpe_families_differ_in_dependency_depth(self):
        benchmark = task('ranked_bpe_tokenization')
        depths = []
        for seed in range(3):
            depth = [0] * 256
            for a, b in benchmark.generate_problem(3000, seed)['merges']:
                depth.append(1 + max(depth[a], depth[b]))
            depths.append(max(depth))
        self.assertEqual(depths[0], 1)
        self.assertGreater(depths[1], 5)
        self.assertGreater(depths[2], 1)

    def test_search_phases_lengths_and_independent_text_scoring(self):
        benchmark = task('dynamic_document_search')
        for seed in range(3):
            problem = benchmark.generate_problem(800, seed)
            operations = problem['operations']
            self.assertEqual(len(operations), 800)
            self.assertEqual(problem, benchmark.generate_problem(800, seed))
            phases = [operations[100 + 175*i:100 + 175*(i+1)] for i in range(4)]
            queries = [sum(op[0] == 'query' for op in phase) for phase in phases]
            self.assertGreater(queries[0], 2 * queries[1])
            self.assertGreater(queries[2], 1.4 * queries[3])
            lengths = [len(op[2].split()) for op in operations if op[0] == 'add']
            self.assertLess(min(lengths), 25)
            self.assertGreater(max(lengths), 100)
            documents, expected = {}, []
            for op in operations:
                if op[0] == 'add':
                    documents[op[1]] = op[2].split()
                elif op[0] == 'delete':
                    documents.pop(op[1], None)
                else:
                    hits = [(identifier, sum(words.count(term) for term in op[1]))
                            for identifier, words in documents.items()]
                    expected.append(tuple(sorted((hit for hit in hits if hit[1]),
                                                 key=lambda hit: (-hit[1], hit[0]))[:op[2]]))
            self.assertEqual(benchmark.solve(problem), tuple(expected))

    @unittest.skipUnless(NUMERICAL, 'requires pinned numerical dependencies')
    def test_numerical_families_vary_and_reference_verifies(self):
        for name, size in [('delaunay', 32), ('ode_stiff_robertson', 4),
                           ('rocket_landing_optimization', 30)]:
            benchmark = task(name)
            for family in range(3):
                examples = []
                for seed in (family, family + 3):
                    with self.subTest(task=name, seed=seed):
                        problem = benchmark.generate_problem(size, seed)
                        self.assertEqual(problem, benchmark.generate_problem(size, seed))
                        self.assertTrue(benchmark.is_solution(problem, benchmark.solve(problem)))
                        examples.append(problem)
                self.assertNotEqual(examples[0], examples[1])


if __name__ == '__main__':
    unittest.main()
