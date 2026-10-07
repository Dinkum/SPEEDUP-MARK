"""Reference implementation for near_duplicate_document_clustering; copied into fresh run candidates."""

from __future__ import annotations

def _clusters(problem):
    docs = problem["documents"]
    width = problem["shingle_width"]
    numerator, denominator = problem["threshold"]
    shingles = []
    for document in docs:
        words = document.split()
        if not words:
            shingles.append(None)
        elif len(words) < width:
            shingles.append({tuple(words)})
        else:
            shingles.append(
                {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}
            )

    parent = list(range(len(docs)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            if left > right:
                left, right = right, left
            parent[right] = left

    for left in range(len(docs)):
        for right in range(left + 1, len(docs)):
            if shingles[left] is None or shingles[right] is None:
                continue
            intersection = len(shingles[left] & shingles[right])
            union_size = len(shingles[left] | shingles[right])
            similar = intersection * denominator >= union_size * numerator
            if similar:
                union(left, right)

    groups = {}
    for index in range(len(docs)):
        groups.setdefault(find(index), []).append(index)
    labels = [0] * len(docs)
    for members in groups.values():
        label = min(members)
        for member in members:
            labels[member] = label
    return tuple(labels)


def solve(problem):
    return _clusters(problem)
