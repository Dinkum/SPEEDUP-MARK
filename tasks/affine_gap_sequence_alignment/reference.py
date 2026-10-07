"""Reference implementation for affine_gap_sequence_alignment; copied into fresh run candidates."""

def _reference_cost(left, right, mismatch, gap_open, gap_extend):
    """Rolling-row three-state dynamic programming, with all cells scored."""
    width = len(right)
    infinity = (len(left) + width) * (mismatch + gap_open + gap_extend) + 1
    matched = [infinity] * (width + 1)
    deleted = [infinity] * (width + 1)
    inserted = [infinity] + [gap_open + (j - 1) * gap_extend for j in range(1, width + 1)]
    matched[0] = 0
    for i, symbol in enumerate(left, 1):
        next_match = [infinity] * (width + 1)
        next_delete = [gap_open + (i - 1) * gap_extend] + [infinity] * width
        next_insert = [infinity] * (width + 1)
        for j, other in enumerate(right, 1):
            next_match[j] = min(matched[j - 1], deleted[j - 1], inserted[j - 1]) + (
                0 if symbol == other else mismatch
            )
            next_delete[j] = min(matched[j] + gap_open, inserted[j] + gap_open,
                                 deleted[j] + gap_extend)
            next_insert[j] = min(next_match[j - 1] + gap_open,
                                 next_delete[j - 1] + gap_open,
                                 next_insert[j - 1] + gap_extend)
        matched, deleted, inserted = next_match, next_delete, next_insert
    return min(matched[-1], deleted[-1], inserted[-1])


def solve(problem):
    return tuple(_reference_cost(*alignment) for alignment in problem["alignments"])
