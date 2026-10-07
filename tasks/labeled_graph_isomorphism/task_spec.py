"""Find an exact isomorphism between small vertex-labeled graphs."""

from __future__ import annotations

import random
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_find_mapping = _reference._find_mapping


def _valid_mapping(problem, mapping):
    left, right = problem["left"], problem["right"]
    size = len(left["labels"])
    if len(right["labels"]) != size or len(mapping) != size:
        return False
    if any(type(value) is not int for value in mapping):
        return False
    if set(mapping) != set(range(size)):
        return False
    if any(left["labels"][i] != right["labels"][mapping[i]] for i in range(size)):
        return False
    mapped_edges = {
        tuple(sorted((mapping[left_node], mapping[right_node])))
        for left_node, right_node in left["edges"]
    }
    return mapped_edges == {tuple(sorted(edge)) for edge in right["edges"]}


class LabeledGraphIsomorphismTask:
    name = "labeled_graph_isomorphism"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 10
    grading_cases = (10, 20)

    def generate_problem(self, n=10, random_seed=0):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        rng = random.Random(random_seed)
        size = n

        def graph(edges, labels):
            permutation = list(range(size))
            rng.shuffle(permutation)
            renamed = [0] * size
            for vertex, image in enumerate(permutation):
                renamed[image] = labels[vertex]
            return {"labels": tuple(renamed),
                    "edges": tuple(sorted(tuple(sorted((permutation[a], permutation[b])))
                                          for a, b in edges))}

        def connected(edges):
            adjacency = [set() for _ in range(size)]
            for a, b in edges:
                adjacency[a].add(b)
                adjacency[b].add(a)
            seen, pending = {0}, [0]
            while pending:
                for neighbor in adjacency[pending.pop()] - seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
            return len(seen) == size

        family = random_seed % 3
        for attempt in range(32):
            edges = {tuple(sorted((i, (i + 1) % size))) for i in range(size)} if size > 2 else set()
            if size == 2:
                edges.add((0, 1))
            if family == 0:
                probability = rng.uniform(0.15, 0.35)
                for a in range(size):
                    for b in range(a + 1, size):
                        if rng.random() < probability:
                            edges.add((a, b))
                labels = tuple(rng.randrange(max(1, min(3, size // 3))) for _ in range(size))
            else:
                # A cycle plus a random matching has very similar local degree
                # signatures, yet many possible global graph structures.
                vertices = list(range(size))
                rng.shuffle(vertices)
                for a, b in zip(vertices[::2], vertices[1::2]):
                    edges.add(tuple(sorted((a, b))))
                labels = (0,) * size
            left = graph(edges, labels)
            if family != 2 or size < 6:
                return {"left": left, "right": graph(edges, labels)}
            changed = set(edges)
            for _ in range(32):
                (a, b), (c, d) = rng.sample(sorted(changed), 2)
                if len({a, b, c, d}) != 4:
                    continue
                added = {tuple(sorted((a, d))), tuple(sorted((c, b)))}
                if added & changed:
                    continue
                proposal = (changed - {(a, b), (c, d)}) | added
                if not connected(proposal):
                    continue
                changed = proposal
                problem = {"left": left, "right": graph(changed, labels)}
                # Equal degrees and connectedness do not certify a negative.
                # Exact search conditions the distribution on non-isomorphism.
                if _find_mapping(problem) is None:
                    return problem
        raise ValueError("could not generate a connected non-isomorphic pair")

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            if proposed is None:
                return _find_mapping(problem) is None
            if type(proposed) not in (tuple, list):
                return False
            return _valid_mapping(problem, proposed)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = LabeledGraphIsomorphismTask()
