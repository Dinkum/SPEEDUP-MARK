"""Exactness, tie-breaks and the float64 divergence claim for ray queries."""

import copy
import pathlib
import unittest
from fractions import Fraction

from speedupmark.harness import load_task


ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = ("static_coherent", "moving_bursts", "mixed_sizes", "incoherent")
COORD_LIMIT = 1 << 34


def _float_ranking(spec, triangles, ray):
    """Rank hits in float64 the way a numerically naive submission would."""
    import math

    origin = tuple(float(value) for value in ray[0])
    direction = tuple(float(value) for value in ray[1])
    best = None
    best_id = -1
    for identifier, triangle in enumerate(triangles):
        if triangle is None:
            continue
        vertices = [tuple(float(value) for value in vertex) for vertex in triangle]
        edge1 = tuple(vertices[1][index] - vertices[0][index] for index in range(3))
        edge2 = tuple(vertices[2][index] - vertices[0][index] for index in range(3))
        pvec = (direction[1] * edge2[2] - direction[2] * edge2[1],
                direction[2] * edge2[0] - direction[0] * edge2[2],
                direction[0] * edge2[1] - direction[1] * edge2[0])
        determinant = sum(edge1[index] * pvec[index] for index in range(3))
        if determinant == 0.0:
            continue
        tvec = tuple(origin[index] - vertices[0][index] for index in range(3))
        u = sum(tvec[index] * pvec[index] for index in range(3)) / determinant
        if u < 0.0 or u > 1.0:
            continue
        qvec = (tvec[1] * edge1[2] - tvec[2] * edge1[1],
                tvec[2] * edge1[0] - tvec[0] * edge1[2],
                tvec[0] * edge1[1] - tvec[1] * edge1[0])
        v = sum(direction[index] * qvec[index] for index in range(3)) / determinant
        if v < 0.0 or u + v > 1.0:
            continue
        t = sum(edge2[index] * qvec[index] for index in range(3)) / determinant
        if t <= 0.0 or math.isnan(t):
            continue
        if best is None or t < best:
            best, best_id = t, identifier
    return best_id


class RayQueryTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(ROOT / "tasks/dynamic_exact_ray_queries")

    def answers(self, scene):
        return self.task.solve({"scenes": (scene,)})

    def test_shared_edge_hit_returns_the_smallest_id(self):
        # Two triangles sharing edge (0,0,0)-(1,0,0); a ray straight through that
        # edge is at exactly equal distance from both.
        scene = {
            "family": "unit",
            "triangles": (
                ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
                ((1, 0, 0), (0, 0, 0), (1, -1, 0)),
            ),
            "phases": ({"rays": (((0, 0, -4), (1, 0, 4)),)},),
        }
        self.assertEqual(self.answers(scene), (((0,),),))
        reversed_scene = dict(scene, triangles=tuple(reversed(scene["triangles"])))
        self.assertEqual(self.answers(reversed_scene), (((0,),),))

    def test_parallel_and_backward_rays_miss(self):
        scene = {
            "family": "unit",
            "triangles": (((0, 0, 0), (4, 0, 0), (0, 4, 0)),),
            "phases": ({"rays": (((0, 1, 1), (1, 0, 0)),        # parallel to the plane
                                 ((0, 0, 4), (0, 0, 1)),        # pointing away
                                 ((9, 9, -1), (0, 0, 1)))},),   # outside the triangle
        }
        self.assertEqual(self.answers(scene), (((-1, -1, -1),),))

    def test_boundary_contact_is_inclusive_and_exact(self):
        scene = {
            "family": "unit",
            "triangles": (((0, 0, 0), (4, 0, 0), (0, 4, 0)),),
            # Aimed exactly at the vertex and along an edge.
            "phases": ({"rays": (((0, 0, -5), (0, 0, 5)),
                                 ((0, 0, -5), (4, 0, 5)))},),
        }
        self.assertEqual(self.answers(scene), (((0, 0),),))

    def test_updates_move_add_and_remove_live_geometry(self):
        scene = {
            "family": "unit",
            "triangles": (((0, 0, 0), (4, 0, 0), (0, 4, 0)),
                          ((10, 10, 0), (14, 10, 0), (10, 14, 0))),
            "phases": (
                {"rays": (((1, 1, -1), (0, 0, 1)),)},
                {"updates": (("remove", 0),)},
                {"rays": (((1, 1, -1), (0, 0, 1)),)},
                {"updates": (("move", 1, ((0, 0, 0), (4, 0, 0), (0, 4, 0))),)},
                {"rays": (((1, 1, -1), (0, 0, 1)),)},
                {"updates": (("add", ((8, 0, 0), (12, 0, 0), (8, 4, 0))),)},
                {"rays": (((9, 1, -1), (0, 0, 1)),)},
            ),
        }
        self.assertEqual(self.answers(scene), (((0,), (-1,), (1,), (2,)),))

    def test_float64_ranking_disagrees_with_the_exact_answer(self):
        """The exactness contract is load-bearing, not decoration."""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "ray_spec", ROOT / "tasks/dynamic_exact_ray_queries/task_spec.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ray_spec"] = module
        spec.loader.exec_module(module)

        problem = self.task.generate_problem(self.task.default_n, 0)
        exact = self.task.solve(problem)
        disagreements = 0
        ties = 0
        for scene, scene_answers in zip(problem["scenes"], exact):
            triangles = list(scene["triangles"])
            phase_index = -1
            for phase in scene["phases"]:
                if "rays" not in phase:
                    for update in phase["updates"]:
                        if update[0] == "remove":
                            triangles[update[1]] = None
                        elif update[0] == "move":
                            triangles[update[1]] = update[2]
                        else:
                            triangles.append(update[1])
                    continue
                phase_index += 1
                for ray, expected in zip(phase["rays"], scene_answers[phase_index]):
                    hits = {}
                    for identifier, triangle in enumerate(triangles):
                        if triangle is None:
                            continue
                        hit = module._hit_parameter(ray, triangle)
                        if hit is not None:
                            hits[identifier] = Fraction(hit[0], hit[1])
                    if hits:
                        closest = min(hits.values())
                        if sum(1 for value in hits.values() if value == closest) > 1:
                            ties += 1
                    if _float_ranking(module, triangles, ray) != expected:
                        disagreements += 1
        self.assertGreaterEqual(ties, 20, "exact ties must be reachable")
        self.assertGreaterEqual(disagreements, 5, "float64 ranking must be punished")

    def test_generated_scenes_respect_their_documented_bounds(self):
        for seed in range(2):
            problem = self.task.generate_problem(self.task.default_n, seed)
            self.assertEqual([scene["family"] for scene in problem["scenes"]], list(FAMILIES))
            for scene in problem["scenes"]:
                for triangle in scene["triangles"]:
                    for vertex in triangle:
                        self.assertTrue(all(type(value) is int and abs(value) <= COORD_LIMIT
                                            for value in vertex))
                for phase in scene["phases"]:
                    for ray in phase.get("rays", ()):
                        self.assertNotEqual(ray[1], (0, 0, 0))
            pristine = copy.deepcopy(problem)
            answers = self.task.solve(problem)
            self.assertEqual(problem, pristine)
            self.assertTrue(self.task.is_solution(problem, answers))

    def test_lazy_containers_are_rejected(self):
        """A sequence subclass could do its work during untimed verification."""
        class Lazy(tuple):
            def __new__(cls, source):
                return super().__new__(cls, source)

            def __iter__(self):
                raise AssertionError("verification must not iterate a lazy container")

        problem = self.task.generate_problem(32, 0)
        answers = self.task.solve(problem)
        self.assertTrue(self.task.is_solution(problem, answers))
        self.assertFalse(self.task.is_solution(problem, Lazy(answers)))

    def test_wrong_type_and_shape_are_rejected(self):
        problem = self.task.generate_problem(32, 0)
        answers = self.task.solve(problem)
        self.assertTrue(self.task.is_solution(problem, answers))
        self.assertFalse(self.task.is_solution(problem, "nope"))
        self.assertFalse(self.task.is_solution(problem, answers[:-1]))
        first = list(answers)
        first[0] = [list(phase) for phase in answers[0]]
        first[0][0][0] = True
        self.assertFalse(self.task.is_solution(problem, first))


if __name__ == "__main__":
    unittest.main()
