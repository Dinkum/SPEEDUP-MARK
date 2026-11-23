"""Reference implementation for incremental_spreadsheet_recalculation; copied into fresh run candidates."""

from __future__ import annotations

def _recalculate(problem, inputs):
    input_count = problem["input_count"]
    values = list(inputs) + [0] * (len(problem["formulas"]) - input_count)
    for cell in range(input_count, len(values)):
        operation, left, right = problem["formulas"][cell]
        if operation == "add":
            values[cell] = values[left] + values[right]
        elif operation == "sub":
            values[cell] = values[left] - values[right]
        elif operation == "scale":
            values[cell] = values[left] * right
        elif operation == "min":
            values[cell] = min(values[left], values[right])
        elif operation == "max":
            values[cell] = max(values[left], values[right])
        elif operation == "mul":
            values[cell] = values[left] * values[right]
        else:
            raise ValueError(operation)
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


def solve(problem):
    return _run(problem)
