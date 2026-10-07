"""Exact signed-integer signal convolution with full/same/valid outputs."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_convolve = _reference._convolve


class ConvolutionTask:
    name = "exact_integer_convolution"
    task_version = "2.0.0"
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

    solve = staticmethod(_reference.solve)


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
