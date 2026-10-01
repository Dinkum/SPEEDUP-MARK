"""Independent physical-state checks for constrained fleet tours."""

import heapq
import pathlib
import random
import unittest

from speedupmark.harness import load_task


TASK_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tasks"


def physical_tour_options(node_count, edges, vehicle, jobs):
    """Search roads, battery, and service actions in one tiny state graph."""
    origin, capacity, skills, battery = vehicle
    adjacency = [[] for _ in range(node_count)]
    for start, end, travel_time, energy in edges:
        adjacency[start].append((end, travel_time, energy))
    best = {(origin, 0, 0): 0}
    queue = [(0, origin, 0, 0)]
    tours = {0: 0}
    while queue:
        elapsed, vertex, used, served = heapq.heappop(queue)
        if elapsed != best[vertex, used, served]:
            continue
        if vertex == origin:
            tours[served] = min(tours.get(served, float("inf")), elapsed)
        actions = [(neighbor, used + energy, served, elapsed + travel_time)
                   for neighbor, travel_time, energy in adjacency[vertex]
                   if used + energy <= battery]
        if served.bit_count() < capacity:
            actions.extend((vertex, used, served | (1 << index), elapsed)
                           for index, (location, required, _) in enumerate(jobs)
                           if vertex == location and skills & required
                           and not served & (1 << index))
        for neighbor, next_energy, next_served, next_time in actions:
            key = neighbor, next_energy, next_served
            if next_time < best.get(key, float("inf")):
                best[key] = next_time
                heapq.heappush(queue, (next_time, *key))
    return tours


def physical_fleet_optimum(node_count, edges, vehicles, jobs):
    options = [physical_tour_options(node_count, edges, vehicle, jobs)
               for vehicle in vehicles]
    best = float("inf")

    def visit(vehicle, served, cost):
        nonlocal best
        if vehicle == len(vehicles):
            penalty = sum(job[2] for index, job in enumerate(jobs)
                          if not served & (1 << index))
            best = min(best, cost + penalty)
            return
        for mask, tour_cost in options[vehicle].items():
            if not mask & served:
                visit(vehicle + 1, served | mask, cost + tour_cost)

    visit(0, 0, 0)
    return best


def problem(edges, vehicles, jobs, node_count):
    operations = tuple(("telemetry", identifier, 0, *vehicle)
                       for identifier, vehicle in enumerate(vehicles))
    return {"scenarios": ({"family": "manual", "node_count": node_count,
                           "edges": tuple(edges),
                           "operations": operations + (
                               ("dispatch", 0, tuple(range(len(vehicles))),
                                tuple(jobs)),)},)}


class LiveFleetDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task(TASK_ROOT / "live_fleet_dispatch")

    def test_three_legs_need_mixed_path_profiles_and_return(self):
        # Each leg has a fast (time 1, energy 4) and an eco
        # (time 4, energy 2) path. Battery 8 permits one fast leg.
        edges = ((0, 1, 1, 4), (1, 2, 1, 4), (2, 0, 1, 4),
                 (0, 3, 2, 1), (3, 1, 2, 1),
                 (1, 4, 2, 1), (4, 2, 2, 1),
                 (2, 5, 2, 1), (5, 0, 2, 1))
        vehicles = ((0, 2, 1, 8),)
        jobs = ((1, 1, 100), (2, 1, 100))
        case = problem(edges, vehicles, jobs, 6)
        self.assertEqual(physical_fleet_optimum(6, edges, vehicles, jobs), 9)
        self.assertEqual(self.task.solve(case), ((9,),))
        self.assertTrue(self.task.is_solution(case, ((9,),)))

    def test_unsupported_middle_label_is_needed_at_budget(self):
        edges = ((0, 1, 10, 1), (0, 2, 4, 1), (2, 1, 4, 1),
                 (0, 3, 1, 1), (3, 4, 1, 1), (4, 1, 1, 1),
                 (1, 0, 1, 1))
        vehicles = ((0, 1, 1, 3),)
        jobs = ((1, 1, 100),)
        case = problem(edges, vehicles, jobs, 5)
        self.assertEqual(self.task.solve(case), ((9,),))
        self.assertTrue(self.task.is_solution(case, ((9,),)))

    def test_fleet_choice_beats_independent_vehicle_choices(self):
        edges = ((0, 1, 1, 1), (1, 0, 1, 1),
                 (0, 3, 3, 1), (3, 0, 3, 1),
                 (2, 1, 1, 1), (1, 2, 1, 1))
        vehicles = ((0, 1, 3, 4), (2, 1, 1, 4))
        jobs = ((1, 1, 100), (3, 2, 100))
        case = problem(edges, vehicles, jobs, 4)
        self.assertEqual(physical_fleet_optimum(4, edges, vehicles, jobs), 8)
        self.assertEqual(self.task.solve(case), ((8,),))
        self.assertTrue(self.task.is_solution(case, ((8,),)))

    def test_closure_reopening_and_late_telemetry(self):
        case = {"scenarios": ({
            "family": "manual", "node_count": 3,
            "edges": ((0, 1, 2, 1), (1, 0, 2, 1)),
            "operations": (
                ("telemetry", 4, 10, 0, 1, 1, 2),
                ("telemetry", 4, 5, 1, 1, 1, 2),
                ("dispatch", 6, (4,), ((1, 1, 20),)),
                ("telemetry", 4, 5, 0, 1, 1, 2),
                ("dispatch", 6, (4,), ((1, 1, 20),)),
                ("delete", 1, 0),
                ("dispatch", 10, (4,), ((1, 1, 20),)),
                ("set", 1, 0, 2, 1),
                ("dispatch", 10, (4,), ((1, 1, 20),)),
                ("dispatch", 4, (4,), ((1, 1, 20),)),
                ("dispatch", 10, (4,), ()),
            ),
        },)}
        self.assertEqual(self.task.solve(case), ((0, 4, 20, 4, 20, 0),))
        self.assertTrue(self.task.is_solution(case, ((0, 4, 20, 4, 20, 0),)))
        self.assertFalse(self.task.is_solution(case, ((0, 4, True, 4, 20, 0),)))

    def test_random_tiny_cases_against_physical_states(self):
        rng = random.Random(1717)
        for _ in range(35):
            node_count = 5
            edges = tuple((start, end, rng.randrange(1, 5), rng.randrange(1, 4))
                          for start in range(node_count) for end in range(node_count)
                          if start != end and rng.random() < 0.35)
            vehicles = tuple((rng.randrange(node_count), rng.randrange(1, 3),
                              rng.choice((1, 2, 3)), rng.randrange(3, 7))
                             for _ in range(2))
            jobs = tuple((rng.randrange(node_count), rng.choice((1, 2)),
                          rng.randrange(5, 21)) for _ in range(3))
            case = problem(edges, vehicles, jobs, node_count)
            expected = physical_fleet_optimum(node_count, edges, vehicles, jobs)
            self.assertEqual(self.task.solve(case), ((expected,),))
            self.assertTrue(self.task.is_solution(case, ((expected,),)))

    def test_four_job_tour_beats_every_three_job_tour(self):
        edges = tuple((start, (start + 1) % 5, 1, 1) for start in range(5))
        jobs = tuple((location, 1, 100) for location in range(1, 5))
        for capacity, expected in ((3, 105), (4, 5)):
            vehicles = ((0, capacity, 1, 5),)
            case = problem(edges, vehicles, jobs, 5)
            self.assertEqual(physical_fleet_optimum(5, edges, vehicles, jobs), expected)
            self.assertEqual(self.task.solve(case), ((expected,),))
            self.assertTrue(self.task.is_solution(case, ((expected,),)))

    def test_four_job_random_cases_against_physical_states(self):
        rng = random.Random(2927)
        for _ in range(12):
            edges = tuple((start, end, rng.randrange(1, 5), rng.randrange(1, 3))
                          for start in range(4) for end in range(4)
                          if start != end and rng.random() < 0.6)
            vehicles = ((0, 4, 3, 8), (2, 3, 1, 6))
            jobs = tuple((rng.randrange(4), rng.choice((1, 2)), rng.randrange(5, 21))
                         for _ in range(4))
            case = problem(edges, vehicles, jobs, 4)
            expected = physical_fleet_optimum(4, edges, vehicles, jobs)
            self.assertEqual(self.task.solve(case), ((expected,),))
            self.assertTrue(self.task.is_solution(case, ((expected,),)))

    def test_generated_families_and_bottleneck_reopening(self):
        case = self.task.generate_problem(20, 5)
        self.assertEqual(case, self.task.generate_problem(20, 5))
        self.assertEqual(tuple(scene["family"] for scene in case["scenarios"]),
                         ("grid", "hubs", "bottleneck"))
        bottleneck = case["scenarios"][2]
        bridge = (bottleneck["node_count"] // 2 - 1,
                  bottleneck["node_count"] // 2)
        self.assertTrue(any(op[:3] == ("delete", *bridge)
                            for op in bottleneck["operations"]))
        self.assertTrue(any(op[:3] == ("set", *bridge)
                            for op in bottleneck["operations"]))
        self.assertTrue(self.task.is_solution(case, self.task.solve(case)))

    def test_generated_optimum_uses_four_job_capacity(self):
        case = self.task.generate_problem(60, 0)
        capped = {"scenarios": tuple({
            **scene,
            "operations": tuple(
                op[:4] + (min(op[4], 3),) + op[5:] if op[0] == "telemetry" else op
                for op in scene["operations"]),
        } for scene in case["scenarios"])}
        full_costs = self.task.solve(case)
        capped_costs = self.task.solve(capped)
        self.assertTrue(all(full <= limited
                            for full_scene, limited_scene in zip(full_costs, capped_costs)
                            for full, limited in zip(full_scene, limited_scene)))
        self.assertNotEqual(full_costs, capped_costs,
                            "Generated tours must benefit from capacity four")


if __name__ == "__main__":
    unittest.main()
