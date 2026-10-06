"""Exact global sequence alignment with integer affine gap costs."""

import heapq
import random
import sys

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import forbidden_imports, load_candidate, watch_imports


_candidate = load_candidate(__file__)
ENGINE_ROOTS = ("Bio", "edlib", "parasail", "pywfa", "wavefront",
                "Levenshtein", "rapidfuzz", "skbio")


def _reference_cost(left, right, mismatch, gap_open, gap_extend):
    """Rolling-row three-state dynamic programming, with all cells scored."""
    width = len(right)
    infinity = (len(left) + width) * (mismatch + gap_open + gap_extend) + 1
    matched = [infinity] * (width + 1)
    deleted = [infinity] * (width + 1)
    inserted = [infinity] + [gap_open + (j - 1) * gap_extend for j in range(1, width + 1)]
    matched[0] = 0
    for i, symbol in enumerate(left, 1):
        next_match = [infinity] * (width + 1)
        next_delete = [gap_open + (i - 1) * gap_extend] + [infinity] * width
        next_insert = [infinity] * (width + 1)
        for j, other in enumerate(right, 1):
            next_match[j] = min(matched[j - 1], deleted[j - 1], inserted[j - 1]) + (
                0 if symbol == other else mismatch
            )
            next_delete[j] = min(matched[j] + gap_open, inserted[j] + gap_open,
                                 deleted[j] + gap_extend)
            next_insert[j] = min(next_match[j - 1] + gap_open,
                                 next_delete[j - 1] + gap_open,
                                 next_insert[j - 1] + gap_extend)
        matched, deleted, inserted = next_match, next_delete, next_insert
    return min(matched[-1], deleted[-1], inserted[-1])


def _checked_cost(left, right, mismatch, gap_open, gap_extend):
    """Shortest path on the edit graph; independent of the reference recurrence."""
    width, height = len(right) + 1, len(left) + 1
    infinity = (len(left) + len(right)) * (mismatch + gap_open + gap_extend) + 1
    distances = [infinity] * (3 * width * height)
    distances[0] = 0
    unit = min(gap_open, gap_extend)
    # The remaining length difference requires at least that many gap steps.
    # Each such step costs at least unit, so this is a consistent A* lower bound.
    queue = [(abs(len(left) - len(right)) * unit, 0, 0)]
    while queue:
        _bound, cost, state = heapq.heappop(queue)
        if cost != distances[state]:
            continue
        cell, previous = divmod(state, 3)
        i, j = divmod(cell, width)
        if i == len(left) and j == len(right):
            return cost
        transitions = []
        if i < len(left) and j < len(right):
            transitions.append((i + 1, j + 1, 0, 0 if left[i] == right[j] else mismatch))
        if i < len(left):
            transitions.append((i + 1, j, 1, gap_extend if previous == 1 else gap_open))
        if j < len(right):
            transitions.append((i, j + 1, 2, gap_extend if previous == 2 else gap_open))
        for next_i, next_j, kind, penalty in transitions:
            target = 3 * (next_i * width + next_j) + kind
            total = cost + penalty
            if total < distances[target]:
                distances[target] = total
                bound = total + abs((len(left) - next_i) - (len(right) - next_j)) * unit
                heapq.heappush(queue, (bound, total, target))
    raise AssertionError("alignment graph must have a terminal path")


def _dna(rng, length):
    return bytes(rng.choice(b"ACGT") for _ in range(length))


def _mutate(rng, data, count):
    result = bytearray(data)
    for index in rng.sample(range(len(result)), min(count, len(result))):
        result[index] = rng.choice(tuple(base for base in b"ACGT" if base != result[index]))
    return bytes(result)


class AffineGapSequenceAlignment:
    name = "affine_gap_sequence_alignment"
    display_name = TASK_CATALOG[name].display_name
    task_version = "1.0.0"
    default_n = 256
    grading_cases = (256, 384)

    def generate_problem(self, n=256, random_seed=0):
        if type(n) is not int or not 1 <= n <= 512:
            raise ValueError("n must be an integer from 1 through 512")
        rng = random.Random(random_seed)
        near = _dna(rng, n)
        near_other = _mutate(rng, near, max(1, n // rng.randrange(30, 70)))
        position = rng.randrange(n + 1)
        near_other = near_other[:position] + _dna(rng, max(1, n // 50)) + near_other[position:]
        gapped = _dna(rng, n)
        start = rng.randrange(max(1, n // 3))
        length = max(1, n // rng.randrange(4, 8))
        gapped_other = gapped[:start] + gapped[start + length:]
        position = rng.randrange(len(gapped_other) + 1)
        gapped_other = (gapped_other[:position] + _dna(rng, max(1, n // 5))
                        + gapped_other[position:])
        motif = _dna(rng, rng.randrange(3, 13))
        repeated = (motif * (n // len(motif) + 1))[:n]
        shift = rng.randrange(1, max(2, min(n, len(motif))))
        repeated_other = _mutate(rng, repeated[shift:] + repeated[:shift], max(1, n // 25))
        unrelated = _dna(rng, n)
        unrelated_other = _dna(rng, rng.randrange(max(1, n // 2), n + max(2, n // 3)))
        return {"alignments": (
            (near, near_other, rng.randrange(2, 6), rng.randrange(4, 9), 1),
            (gapped, gapped_other, rng.randrange(3, 8), rng.randrange(5, 12), rng.randrange(1, 3)),
            (repeated, repeated_other, rng.randrange(2, 7), rng.randrange(2, 7), rng.randrange(1, 4)),
            (unrelated, unrelated_other, rng.randrange(3, 10), rng.randrange(2, 9), rng.randrange(1, 4)),
        )}

    def solve(self, problem):
        return tuple(_reference_cost(*alignment) for alignment in problem["alignments"])

    def candidate_solve(self, problem):
        self.policy_violations = ()
        loaded = set(sys.modules)
        with watch_imports(ENGINE_ROOTS) as imported_during:
            result = _candidate.solve(problem, self.solve)
        violations = forbidden_imports(_candidate, ENGINE_ROOTS, loaded, imported_during)
        if violations:
            self.policy_violations = violations
            print(f"candidate uses forbidden engine imports: {', '.join(violations)}")
            return None
        return result

    def is_solution(self, problem, proposed):
        if type(proposed) not in (tuple, list) or len(proposed) != len(problem["alignments"]):
            return False
        if any(type(cost) is not int or cost < 0 for cost in proposed):
            return False
        return all(cost == _checked_cost(*alignment)
                   for cost, alignment in zip(proposed, problem["alignments"]))


TASK = AffineGapSequenceAlignment()
