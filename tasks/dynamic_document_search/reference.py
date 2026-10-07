"""Reference implementation for dynamic_document_search; copied into fresh run candidates."""

from __future__ import annotations

def _search(problem):
    documents = {}
    answers = []
    for operation in problem["operations"]:
        if operation[0] == "add":
            _, document_id, text = operation
            frequencies = {}
            for term in text.split():
                frequencies[term] = frequencies.get(term, 0) + 1
            documents[document_id] = frequencies
        elif operation[0] == "delete":
            documents.pop(operation[1], None)
        else:
            _, query_terms, limit = operation
            query_frequencies = {}
            for term in query_terms:
                query_frequencies[term] = query_frequencies.get(term, 0) + 1
            ranked = []
            for document_id, frequencies in documents.items():
                score = sum(
                    frequencies.get(term, 0) * weight
                    for term, weight in query_frequencies.items()
                )
                if score > 0:
                    ranked.append((document_id, score))
            ranked.sort(key=lambda row: (-row[1], row[0]))
            answers.append(tuple(ranked[:limit]))
    return tuple(answers)


def solve(problem):
    return _search(problem)
