"""Example task — copy this directory to add a new task. No registry edits needed."""

import gzip
import hashlib


from speedupmark.task import load_reference


_reference = load_reference(__file__)


class GzipTask:
    name = "example_gzip"
    task_version = "2.0.0"
    default_n = 1000
    grading_cases = (1000, 2000)

    def generate_problem(self, n: int = 1000, random_seed: int = 0) -> bytes:
        # Deterministic pseudo-random bytes scaled by n.
        h = hashlib.sha256(str(random_seed).encode()).digest()
        return (h * (n // len(h) + 1))[:n]

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem: bytes, proposed: bytes) -> bool:
        try:
            return gzip.decompress(proposed) == problem
        except Exception:
            return False


TASK = GzipTask()
