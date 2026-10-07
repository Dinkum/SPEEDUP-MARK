"""Reference implementation for ranked_bpe_tokenization; copied into fresh run candidates."""

from __future__ import annotations

def _encode(problem):
    tokens = list(problem["data"])
    ranks = {}
    for rank, pair in enumerate(problem["merges"]):
        # A repeated pair retains its lowest rank and that rank's token ID.
        ranks.setdefault(pair, rank)
    while len(tokens) > 1:
        best_rank = None
        for index in range(len(tokens) - 1):
            rank = ranks.get((tokens[index], tokens[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
        if best_rank is None:
            break
        pair = problem["merges"][best_rank]
        merged_id = 256 + best_rank
        output = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                output.append(merged_id)
                index += 2
            else:
                output.append(tokens[index])
                index += 1
        tokens = output
    return tuple(tokens)


def solve(problem):
    return _encode(problem)
