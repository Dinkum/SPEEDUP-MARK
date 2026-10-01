"""Example task — copy this directory to add a new task. No registry edits needed."""

import gzip
import hashlib

from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


class GzipTask:
    name = "example_gzip"
    task_version = "1.1.0"
    default_n = 1000
    grading_cases = (1000, 2000)

    def generate_problem(self, n: int = 1000, random_seed: int = 0) -> bytes:
        # Deterministic pseudo-random bytes scaled by n.
        h = hashlib.sha256(str(random_seed).encode()).digest()
        return (h * (n // len(h) + 1))[:n]

    def solve(self, problem: bytes) -> bytes:
        return gzip.compress(problem)

    def is_solution(self, problem: bytes, proposed: bytes) -> bool:
        try:
            return gzip.decompress(proposed) == problem
        except Exception:
            return False

    def candidate_solve(self, problem: bytes) -> bytes:
        return _candidate.solve(problem, self.solve)


TASK = GzipTask()
