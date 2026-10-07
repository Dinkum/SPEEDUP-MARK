"""Maintain a mutable corpus and answer exact integer-ranked searches."""

from __future__ import annotations

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)


ENGINE_ROOTS = ("sqlite3", "_sqlite3", "duckdb", "_duckdb", "sqlalchemy", "apsw",
                "polars", "pandas", "datafusion", "pyarrow", "whoosh", "tantivy")


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _verify_search(problem):
    """Raw token counts, without the reference's frequency maps."""
    documents, answers = {}, []
    for op in problem["operations"]:
        if op[0] == "add":
            documents[op[1]] = op[2].split()
        elif op[0] == "delete":
            documents.pop(op[1], None)
        else:
            hits = [(identifier, sum(words.count(term) for term in op[1]))
                    for identifier, words in documents.items()]
            answers.append(tuple(sorted((hit for hit in hits if hit[1]),
                                        key=lambda hit: (-hit[1], hit[0]))[:op[2]]))
    return tuple(answers)

class DynamicDocumentSearchTask:
    forbidden_import_roots = ENGINE_ROOTS
    name = "dynamic_document_search"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 800
    grading_cases = (800, 1600)

    def generate_problem(self, n=800, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        vocabulary = tuple(f"term-{index}" for index in range(max(160, min(768, n // 3))))
        hot = rng.sample(vocabulary, 8)
        tail = tuple(term for term in vocabulary if term not in hot)
        operations, live = [], set()
        next_id = 0
        warmup = min(n, max(1, n // 8))
        # Every normal-size stream crosses query-heavy and update-heavy phases,
        # and common/rare query terms put different pressure on posting lists.
        query_rates = (0.85, 0.25, 0.90, 0.45)
        common_rates = (0.15, 0.80, 0.85, 0.25)
        for index in range(n):
            phase = min(3, max(0, index - warmup) * 4 // max(1, n - warmup))
            if index >= warmup and live and rng.random() < query_rates[phase]:
                terms = tuple(rng.choice(hot if rng.random() < common_rates[phase] else tail)
                              for _ in range(rng.randrange(2, 7)))
                if rng.random() < 0.1:
                    terms += ("absent-term",)
                operations.append(("query", terms, rng.choice((3, 10, 30))))
            elif live and index >= warmup and rng.random() < 0.15:
                document_id = rng.choice(sorted(live))
                live.remove(document_id)
                operations.append(("delete", document_id))
            else:
                if live and index >= warmup and rng.random() < 0.65:
                    document_id = rng.choice(sorted(live))
                else:
                    document_id = next_id
                    next_id += 1
                long = rng.random() < (0.75 if phase in (1, 3) else 0.15)
                length = rng.randrange(80, 161) if long else rng.randrange(8, 25)
                text = " ".join(rng.choice(hot if rng.random() < 0.65 else tail)
                                for _ in range(length))
                operations.append(("add", document_id, text))
                live.add(document_id)
        return {"operations": tuple(operations)}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            return plain_containers(proposed) and _same_materialized(proposed, _verify_search(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = DynamicDocumentSearchTask()
