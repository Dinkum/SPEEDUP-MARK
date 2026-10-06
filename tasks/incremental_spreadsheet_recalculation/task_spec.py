"""Evaluate edits and queries over an integer formula DAG."""

from __future__ import annotations

import random
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate, plain_containers


_candidate = load_candidate(__file__)


def _recalculate(problem, inputs):
    input_count = problem["input_count"]
    values = list(inputs) + [0] * (len(problem["formulas"]) - input_count)
    for cell in range(input_count, len(values)):
        operation, left, right = problem["formulas"][cell]
        if operation == "add":
            values[cell] = values[left] + values[right]
        elif operation == "sub":
            values[cell] = values[left] - values[right]
        else:
            values[cell] = values[left] * right
    return values


def _run(problem):
    inputs = list(problem["initial_inputs"])
    answers = []
    for operation in problem["operations"]:
        if operation[0] == "set":
            _, cell, value = operation
            inputs[cell] = value
        else:
            _, cell = operation
            answers.append(_recalculate(problem, inputs)[cell])
    return tuple(answers)


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
                elif op == "add":
                    values[cell] = values[left] + values[right]
                elif op == "sub":
                    values[cell] = values[left] - values[right]
                else:
                    raise ValueError(op)
            else:
                pending.append((cell, True))
                pending.append((left, False))
                if op != "scale":
                    pending.append((right, False))
        answers.append(values[operation[1]])
    return tuple(answers)

class IncrementalSpreadsheetRecalculationTask:
    name = "incremental_spreadsheet_recalculation"
    task_version = "1.1.1"
    display_name = TASK_CATALOG[name].display_name
    default_n = 650
    grading_cases = (650, 1300)

    def generate_problem(self, n=650, random_seed=0):
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
            else:
                right = rng.randrange(low, cell)
                formulas.append((rng.choice(("add", "sub")), left, right))
        operations = []
        initial_inputs = [rng.randrange(-100, 101) for _ in range(input_count)]
        current_inputs = list(initial_inputs)
        for index in range(max(40, n)):
            if index % 3 == 0:
                target = rng.randrange(input_count)
                value = current_inputs[target] if index % 9 == 0 else rng.randrange(-100, 101)
                operations.append(("set", target, value))
                current_inputs[target] = value
            else:
                operations.append(("query", rng.randrange(n)))
        return {
            "input_count": input_count,
            "initial_inputs": tuple(initial_inputs),
            "formulas": tuple(formulas),
            "operations": tuple(operations),
        }

    def solve(self, problem):
        return _run(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        try:
            if not plain_containers(proposed) or type(proposed) not in (tuple, list):
                return False
            if any(type(value) is not int for value in proposed):
                return False
            return tuple(proposed) == _verify_sheet(problem)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = IncrementalSpreadsheetRecalculationTask()
