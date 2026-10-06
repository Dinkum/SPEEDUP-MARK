"""Exact signed-integer signal convolution with full/same/valid outputs."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _convolve(problem):
    x, y = problem["signal_x"], problem["signal_y"]
    if not x or not y:
        return ()
    output = [0] * (len(x) + len(y) - 1)
    for i, left in enumerate(x):
        for j, right in enumerate(y):
            output[i + j] += left * right
    mode = problem["mode"]
    if mode == "full":
        return tuple(output)
    if mode == "same":
        start = (len(y) - 1) // 2
        return tuple(output[start : start + len(x)])
    if mode == "valid":
        return tuple(output[min(len(x), len(y)) - 1 : max(len(x), len(y))])
    raise ValueError("unsupported convolution mode")


class ConvolutionTask:
    name = "exact_integer_convolution"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 420
    grading_cases = (420, 840)

    def generate_problem(self, n=420, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        x = tuple(rng.randrange(-32, 33) for _ in range(n))
        y = tuple(rng.randrange(-32, 33) for _ in range(rng.randrange(max(1, n // 2), n + 2)))
        # Sparse cases reward algorithm selection as well as dense kernels.
        if random_seed % 5 == 4:
            y = tuple(value if rng.random() < 0.1 else 0 for value in y)
        return {"signal_x": x, "signal_y": y, "mode": ("full", "same", "valid")[random_seed % 3]}

    def solve(self, problem):
        return {"convolution": _convolve(problem)}

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"convolution"}:
            return False
        values = proposed["convolution"]
        return (
            type(values) in (list, tuple)
            and all(type(value) is int for value in values)
            and tuple(values) == _convolve(problem)
        )


TASK = ConvolutionTask()
