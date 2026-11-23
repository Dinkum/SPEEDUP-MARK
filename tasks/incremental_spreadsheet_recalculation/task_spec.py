"""Evaluate edits and queries over an integer formula DAG."""

from __future__ import annotations

import operator
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)
_BINARY = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
           "min": min, "max": max}


def _verify_sheet(problem):
    """Evaluate only queried ancestors with an explicit DFS stack per edit."""
    inputs = list(problem["initial_inputs"])
    answers = []
    for operation in problem["operations"]:
        if operation[0] == "set":
            inputs[operation[1]] = operation[2]
            continue
        values = dict(enumerate(inputs))
        pending = [(operation[1], False)]
        while pending:
            cell, ready = pending.pop()
            if cell in values:
                continue
            op, left, right = problem["formulas"][cell]
            if ready:
                if op == "scale":
                    values[cell] = values[left] * right
                else:
                    values[cell] = _BINARY[op](values[left], values[right])
            else:
                pending.append((cell, True))
                pending.append((left, False))
                if op != "scale":
                    pending.append((right, False))
        answers.append(values[operation[1]])
    return tuple(answers)

class IncrementalSpreadsheetRecalculationTask:
    name = "incremental_spreadsheet_recalculation"
    task_version = "2.2.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 2400
    grading_cases = (2400, 4800)
    uses_reference_output = True

    def correctness_problem(self, seed):
        value = random.Random(seed).randrange(5, 101)
        return {
            "input_count": 2, "initial_inputs": (value, -2),
            "formulas": (None, None, ("mul", 0, 1), ("min", 2, 1),
                         ("max", 3, 0), ("sub", 4, 3), ("scale", 5, -2)),
            "operations": (("query", 6), ("set", 0, -value), ("query", 6),
                           ("set", 1, 0), ("query", 6), ("query", 2)),
        }

    def generate_problem(self, n=2400, random_seed=0):
        if n < 2:
            raise ValueError("n must be at least 2")
        rng = random.Random(random_seed)
        input_count = max(1, n // 6)
        formulas = [None] * input_count
        for cell in range(input_count, n):
            low = max(0, cell - 24)
            if cell % 4 == 0:
                left = rng.randrange(input_count)
            elif cell % 4 == 1:
                left = rng.randrange(input_count, cell) if cell > input_count else rng.randrange(input_count)
            else:
                left = rng.randrange(low, cell)
            if cell % 7 == 0:
                formulas.append(("scale", left, rng.choice((-2, -1, 0, 1, 2))))
            elif cell % 4 == 0:
                formulas.append(("add", left, rng.randrange(input_count)))
            elif rng.random() < 0.25:
                formulas.append(("scale", left, rng.choice((-2, -1, 0, 1, 2))))
            elif cell % 11 == 0:
                # One bounded input factor keeps long formula chains practical.
                formulas.append(("mul", left, rng.randrange(input_count)))
            else:
                right = rng.randrange(low, cell)
                formulas.append((rng.choice(("add", "sub", "min", "max")), left, right))
        operations = []
        initial_inputs = [rng.randrange(-100, 101) for _ in range(input_count)]
        current_inputs = list(initial_inputs)
        hot_inputs = rng.sample(range(input_count), min(8, input_count))
        hot_queries = rng.sample(range(input_count, n), min(24, n - input_count))
        operation_count = max(40, n)
        for index in range(operation_count):
            phase = index * 4 // operation_count
            if rng.random() < (0.10, 0.88, 0.20, 0.65)[phase]:
                target = rng.choice(hot_inputs) if phase in (1, 3) else rng.randrange(input_count)
                value = current_inputs[target] if rng.random() < 0.20 else rng.randrange(-100, 101)
                operations.append(("set", target, value))
                current_inputs[target] = value
            else:
                target = rng.choice(hot_queries) if phase in (0, 3) else rng.randrange(n)
                operations.append(("query", target))
        return {
            "input_count": input_count,
            "initial_inputs": tuple(initial_inputs),
            "formulas": tuple(formulas),
            "operations": tuple(operations),
        }

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed, *, reference_output=None):
        try:
            if not plain_containers(proposed) or type(proposed) not in (tuple, list):
                return False
            if any(type(value) is not int for value in proposed):
                return False
            expected = _verify_sheet(problem) if reference_output is None else reference_output
            return tuple(proposed) == expected
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = IncrementalSpreadsheetRecalculationTask()
