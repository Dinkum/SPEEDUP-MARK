"""Independent small-problem checks for the stateful frontier tasks."""

import copy
import itertools
import pathlib
import random
import unittest

from speedupmark.harness import load_task


TASK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tasks"


def brute_triangles(scenario):
    relations = {
        name: {(a, b): weight for a, b, weight in rows}
        for name, rows in scenario["relations"].items()
    }
    answers = []
    for operation in scenario["operations"]:
        kind = operation[0]
        if kind == "query":
            total = 0
            # Triple cross-product deliberately shares neither the reference's
            # adjacency intersections nor the checker's two-edge path index.
            for ((a, b), r), ((bb, c), s), ((cc, aa), t) in itertools.product(
                relations["R"].items(), relations["S"].items(), relations["T"].items()
            ):
                if a == aa and b == bb and c == cc:
                    total += r * s * t
            answers.append(total)
        elif kind == "set":
            _, name, source, target, weight = operation
            relations[name][source, target] = weight
        else:
            _, name, source, target = operation
            relations[name].pop((source, target), None)
    return tuple(answers)


def floyd_answers(scenario):
    edges = {(a, b): weight for a, b, weight in scenario["edges"]}
    answers = []
    size = scenario["node_count"]
    for operation in scenario["operations"]:
        kind, source, target, *value = operation
        if kind == "set":
            edges[source, target] = value[0]
        elif kind == "delete":
            edges.pop((source, target), None)
        else:
            distance = [[None] * size for _ in range(size)]
            for vertex in range(size):
                distance[vertex][vertex] = 0
            for (a, b), weight in edges.items():
                if a != b:
                    distance[a][b] = weight
            for middle in range(size):
                for a in range(size):
                    for b in range(size):
                        if distance[a][middle] is None or distance[middle][b] is None:
                            continue
                        candidate = distance[a][middle] + distance[middle][b]
                        if distance[a][b] is None or candidate < distance[a][b]:
                            distance[a][b] = candidate
            answers.append(distance[source][target])
    return tuple(answers)


class MultiwayJoinTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_ROOT / "incremental_multiway_join")

    def test_signed_replacements_zero_and_all_three_delta_directions(self):
        scenario = {
            "relations": {"R": ((1, 2, 3),), "S": ((2, 4, -2),), "T": ((4, 1, 5),)},
            "operations": (
                ("query",),
                ("set", "R", 1, 2, -1),
                ("query",),
                ("set", "S", 2, 4, 7),
                ("query",),
                ("set", "T", 4, 1, -3),
                ("query",),
                ("delete", "R", 9, 9),
                ("set", "S", 2, 4, 0),
                ("query",),
                ("set", "S", 2, 4, 7),
                ("delete", "T", 4, 1),
                ("query",),
                ("set", "T", 4, 1, 2 ** 70),
                ("query",),
            ),
        }
        problem = {"scenarios": (scenario,)}
        expected = ((-30, 10, -35, 21, 0, 0, -7 * 2 ** 70),)
        self.assertEqual(self.task.solve(problem), expected)
        self.assertEqual((brute_triangles(scenario),), expected)
        self.assertTrue(self.task.is_solution(problem, expected))
        self.assertTrue(self.task.is_solution(problem, [list(expected[0])]))

    def test_random_streams_against_triple_cross_product(self):
        rng = random.Random(704)
        for _ in range(25):
            relations = {
                name: tuple((a, b, rng.randrange(-3, 4)) for a in range(3) for b in range(3))
                for name in ("R", "S", "T")
            }
            operations = []
            for _ in range(35):
                name, a, b = rng.choice(("R", "S", "T")), rng.randrange(3), rng.randrange(3)
                if rng.random() < 0.3:
                    operations.append(("delete", name, a, b))
                else:
                    operations.append(("set", name, a, b, rng.randrange(-3, 4)))
                operations.append(("query",))
            scenario = {"relations": relations, "operations": tuple(operations)}
            problem = {"scenarios": (scenario,)}
            original = copy.deepcopy(problem)
            expected = (brute_triangles(scenario),)
            self.assertEqual(self.task.solve(problem), expected)
            self.assertTrue(self.task.is_solution(problem, expected))
            self.assertEqual(problem, original)

    def test_empty_relations_and_strict_materialized_answers(self):
        problem = {"scenarios": ({
            "relations": {"R": (), "S": (), "T": ()},
            "operations": (("query",),),
        },)}
        self.assertEqual(self.task.solve(problem), ((0,),))
        for answer in (None, (), ((),), ((False,),), ((0.0,),), ((1,),), ((0, 0),), ({0},)):
            self.assertFalse(self.task.is_solution(problem, answer))
        self.assertTrue(self.task.is_solution({"scenarios": ()}, ()))


class DynamicShortestPathsTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK_ROOT / "dynamic_shortest_paths")

    def test_increase_decrease_closure_reopening_and_directed_disconnection(self):
        scenario = {
            "node_count": 4,
            "edges": ((0, 1, 4), (1, 2, 5), (0, 2, 20), (2, 2, 1)),
            "operations": (
                ("query", 0, 2), ("query", 2, 0), ("query", 3, 3),
                ("set", 0, 1, 30), ("query", 0, 2),
                ("set", 0, 2, 2), ("query", 0, 2),
                ("delete", 0, 2), ("query", 0, 2),
                ("delete", 1, 2), ("query", 0, 2),
                ("delete", 1, 2), ("set", 0, 2, 2 ** 80), ("query", 0, 2),
            ),
        }
        problem = {"scenarios": (scenario,)}
        expected = ((9, None, 0, 20, 2, 35, None, 2 ** 80),)
        self.assertEqual(self.task.solve(problem), expected)
        self.assertEqual((floyd_answers(scenario),), expected)
        self.assertTrue(self.task.is_solution(problem, expected))
        self.assertTrue(self.task.is_solution(problem, [list(expected[0])]))

    def test_random_dynamic_graphs_against_floyd_warshall(self):
        rng = random.Random(804)
        for _ in range(35):
            size = rng.randrange(2, 8)
            edges = tuple(
                (a, b, rng.randrange(1, 25))
                for a in range(size) for b in range(size) if rng.random() < 0.25
            )
            operations = []
            for _ in range(20):
                source, target = rng.randrange(size), rng.randrange(size)
                if rng.random() < 0.35:
                    operations.append(("delete", source, target))
                else:
                    operations.append(("set", source, target, rng.randrange(1, 40)))
                operations.extend(("query", a, b) for a in range(size) for b in range(size))
            scenario = {"node_count": size, "edges": edges, "operations": tuple(operations)}
            problem = {"scenarios": (scenario,)}
            original = copy.deepcopy(problem)
            expected = (floyd_answers(scenario),)
            self.assertEqual(self.task.solve(problem), expected)
            self.assertTrue(self.task.is_solution(problem, expected))
            self.assertEqual(problem, original)

    def test_strict_answers(self):
        problem = {"scenarios": ({"node_count": 1, "edges": (), "operations": (("query", 0, 0),)},)}
        for answer in (None, (), ((),), ((False,),), ((0.0,),), ((None,),), ((1,),), ({0},)):
            self.assertFalse(self.task.is_solution(problem, answer))
        self.assertTrue(self.task.is_solution({"scenarios": ()}, ()))


class FrontierDataGeneratorTests(unittest.TestCase):
    def test_every_size_contains_every_family_and_is_seeded(self):
        cases = {
            "incremental_multiway_join": ("uniform", "hub_skew", "dense_bursts"),
            "dynamic_shortest_paths": ("grid", "hubs", "bottleneck"),
        }
        for slug, families in cases.items():
            task = load_task(TASK_ROOT / slug)
            for size in (0, 10, 35):
                problem = task.generate_problem(size, 44)
                self.assertEqual(problem, task.generate_problem(size, 44))
                self.assertNotEqual(problem, task.generate_problem(size, 45))
                self.assertEqual(tuple(s["family"] for s in problem["scenarios"]), families)
                self.assertTrue(task.is_solution(problem, task.solve(problem)))
            for size in (-1, True, 3.5):
                with self.assertRaises(ValueError):
                    task.generate_problem(size, 0)


if __name__ == "__main__":
    unittest.main()
