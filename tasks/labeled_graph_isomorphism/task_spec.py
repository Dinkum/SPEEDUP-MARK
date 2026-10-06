"""Find an exact isomorphism between small vertex-labeled graphs."""

from __future__ import annotations

import random
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _adjacency(graph):
    size = len(graph["labels"])
    adjacency = [set() for _ in range(size)]
    for left, right in graph["edges"]:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _find_mapping(problem):
    left, right = problem["left"], problem["right"]
    if len(left["labels"]) != len(right["labels"]):
        return None
    size = len(left["labels"])
    left_adj, right_adj = _adjacency(left), _adjacency(right)
    candidates = []
    for node in range(size):
        matches = [
            other
            for other in range(size)
            if left["labels"][node] == right["labels"][other]
            and len(left_adj[node]) == len(right_adj[other])
        ]
        if not matches:
            return None
        candidates.append(matches)
    order = []
    remaining = set(range(size))
    while remaining:
        node = min(remaining, key=lambda v: (-len(left_adj[v].intersection(order)),
                                            len(candidates[v]), -len(left_adj[v]), v))
        order.append(node)
        remaining.remove(node)
    mapping = [-1] * size
    used = set()

    def visit(depth):
        if depth == size:
            return True
        node = order[depth]
        for other in candidates[node]:
            if other in used:
                continue
            if any(
                ((prior in left_adj[node]) != (mapping[prior] in right_adj[other]))
                for prior in range(size)
                if mapping[prior] >= 0
            ):
                continue
            mapping[node] = other
            used.add(other)
            if visit(depth + 1):
                return True
            used.remove(other)
            mapping[node] = -1
        return False

    return tuple(mapping) if visit(0) else None


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
    task_version = "1.2.1"
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

    def solve(self, problem):
        return _find_mapping(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

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
