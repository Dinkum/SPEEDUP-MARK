"""Reference implementation for adaptive_query_engine; copied into fresh run candidates."""

from __future__ import annotations

def _comparison(op):
    if op == "eq":
        return lambda left, right: left == right
    if op == "ne":
        return lambda left, right: left != right
    if op == "lt":
        return lambda left, right: left < right
    if op == "le":
        return lambda left, right: left <= right
    if op == "gt":
        return lambda left, right: left > right
    if op == "ge":
        return lambda left, right: left >= right
    raise ValueError(f"unsupported comparison {op!r}")


def _evaluate_node(node, tables):
    """Return ``(schema, rows)`` for one plan node, evaluated left to right.

    The reference never reorders anything: it hashes the left input of a join,
    probes with the right input, and applies each filter exactly where the tree
    puts it.
    """
    kind = node[0]
    if kind == "scan":
        table = tables[node[1]]
        return tuple(table["columns"]), [tuple(row) for row in table["rows"]]
    if kind == "filter":
        _, source, column, op, value = node
        schema, rows = _evaluate_node(source, tables)
        index = schema.index(column)
        keep = _comparison(op)
        return schema, [row for row in rows if keep(row[index], value)]
    if kind == "join":
        _, left_node, right_node, left_column, right_column = node
        left_schema, left_rows = _evaluate_node(left_node, tables)
        right_schema, right_rows = _evaluate_node(right_node, tables)
        left_index = left_schema.index(left_column)
        right_index = right_schema.index(right_column)
        buckets = {}
        for row in left_rows:
            buckets.setdefault(row[left_index], []).append(row)
        joined = []
        for row in right_rows:
            for match in buckets.get(row[right_index], ()):
                joined.append(match + row)
        return left_schema + right_schema, joined
    if kind == "group":
        _, source, keys, aggregates = node
        schema, rows = _evaluate_node(source, tables)
        key_indices = [schema.index(key) for key in keys]
        aggregate_indices = [
            None if column is None else schema.index(column)
            for _, column in aggregates
        ]
        groups = {}
        for row in rows:
            key = tuple(row[index] for index in key_indices)
            accumulator = groups.get(key)
            if accumulator is None:
                accumulator = [None] * len(aggregates)
                groups[key] = accumulator
            for position, (function, column) in enumerate(aggregates):
                if function == "count":
                    accumulator[position] = (accumulator[position] or 0) + 1
                    continue
                value = row[aggregate_indices[position]]
                current = accumulator[position]
                if current is None:
                    # sum, min and max all start from the first value seen.
                    accumulator[position] = value
                elif function == "sum":
                    accumulator[position] = current + value
                elif function == "min":
                    accumulator[position] = min(current, value)
                else:
                    accumulator[position] = max(current, value)
        output_schema = tuple(keys) + tuple(
            "count" if column is None else f"{function}:{column}"
            for function, column in aggregates
        )
        return output_schema, [key + tuple(value) for key, value in sorted(groups.items())]
    if kind == "topk":
        _, source, column, limit = node
        schema, rows = _evaluate_node(source, tables)
        if limit < 1:
            raise ValueError("topk requires a positive limit")
        index = schema.index(column)
        # Total order: the named column descending, then the whole row ascending,
        # so the boundary between rank k and rank k+1 is decided exactly.
        rows.sort(key=lambda row: (-row[index], row))
        selected = rows[:limit]
        selected.sort()
        return schema, selected
    if kind == "project":
        _, source, columns = node
        schema, rows = _evaluate_node(source, tables)
        indices = [schema.index(column) for column in columns]
        return tuple(columns), [tuple(row[index] for index in indices) for row in rows]
    raise ValueError(f"unsupported plan node {kind!r}")


def _answer_query(node, tables):
    schema, rows = _evaluate_node(node, tables)
    for row in rows:
        if any(type(value) is not int or isinstance(value, bool) for value in row):
            raise ValueError("query results must be exact integers")
    return tuple(sorted(rows))


def _answer(problem):
    """Canonical answers: one tuple of rows per query, ascending, per family."""
    return tuple(
        tuple(_answer_query(query, family["tables"]) for query in family["queries"])
        for family in problem["families"]
    )


def solve(problem):
    return _answer(problem)
