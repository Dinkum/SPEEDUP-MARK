"""Cluster documents by exact shingle Jaccard similarity."""

from __future__ import annotations

import random
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)


def _checked_clusters(problem):
    """Independent integer-bitset similarity graph and component traversal."""
    identifiers = {}
    masks = []
    width = problem["shingle_width"]
    for document in problem["documents"]:
        words = document.split()
        mask = 0
        for start in range(max(1, len(words) - width + 1) if words else 0):
            shingle = tuple(words[start:start + width])
            if shingle not in identifiers:
                identifiers[shingle] = len(identifiers)
            mask |= 1 << identifiers[shingle]
        masks.append(mask)
    counts = [mask.bit_count() for mask in masks]
    numerator, denominator = problem["threshold"]
    neighbors = [[] for _ in masks]
    for left, mask in enumerate(masks):
        if not mask:
            continue
        for right in range(left + 1, len(masks)):
            if not masks[right]:
                continue
            overlap = (mask & masks[right]).bit_count()
            if overlap * (denominator + numerator) >= (
                    counts[left] + counts[right]) * numerator:
                neighbors[left].append(right)
                neighbors[right].append(left)
    labels = [-1] * len(masks)
    for root in range(len(masks)):
        if labels[root] != -1:
            continue
        labels[root] = root
        pending = [root]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if labels[neighbor] == -1:
                    labels[neighbor] = root
                    pending.append(neighbor)
    return tuple(labels)


class NearDuplicateDocumentClusteringTask:
    name = "near_duplicate_document_clustering"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 360
    grading_cases = (360, 720)

    def generate_problem(self, n=360, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        width, threshold = ((2, (1, 2)), (3, (2, 3)),
                            (4, (4, 5)))[random_seed % 3]
        numerator, denominator = threshold
        vocabulary = [f"w{i}" for i in range(max(1024, n * 3))]
        boilerplate = rng.sample(vocabulary, 24)
        documents = []
        while len(documents) < n:
            length = rng.choice((24, 48, 96, 192))
            count = min(rng.randrange(3, 9), n - len(documents))
            shingle_count = length - width + 1
            # Shifted windows put adjacent pairs on opposite sides of the
            # exact threshold. Longer chains require component closure, not
            # comparison with one representative. Tokens reveal no group id.
            boundary = shingle_count * (denominator - numerator) // (
                denominator + numerator)
            step = max(1, boundary + rng.choice((0, 1)))
            source = rng.sample(vocabulary, length + step * (count - 1))
            shared = boilerplate if rng.random() < 0.5 else []
            for offset in range(count):
                words = shared + source[offset * step:offset * step + length]
                if rng.random() < 0.25:
                    # Interior edits stress shingle overlap rather than only
                    # length filtering; repeated boilerplate creates skew.
                    for _ in range(max(1, length // 24)):
                        words[rng.randrange(len(words))] = rng.choice(vocabulary)
                documents.append(" ".join(words))
        rng.shuffle(documents)
        return {
            "documents": tuple(documents),
            "shingle_width": width,
            "threshold": threshold,
        }

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            if not plain_containers(proposed) or type(proposed) not in (tuple, list):
                return False
            if len(proposed) != len(problem["documents"]):
                return False
            if any(type(value) is not int for value in proposed):
                return False
            return tuple(proposed) == _checked_clusters(problem)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = NearDuplicateDocumentClusteringTask()
