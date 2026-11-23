"""Tokenize arbitrary bytes with an exact ranked BPE merge table."""

from __future__ import annotations

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)
_encode = _reference._encode


TOKENIZER_ROOTS = ("tokenizers", "tiktoken", "sentencepiece", "transformers",
                   "subword_nmt", "youtokentome")


def _verify_encoding(problem):
    """A linked-token priority queue, independent of whole-list merge passes."""
    import heapq
    tokens = list(problem["data"])
    count = len(tokens)
    previous, following = list(range(-1, count - 1)), list(range(1, count + 1))
    alive = [True] * count
    versions = [0] * count
    ranks = {}
    for rank, pair in enumerate(problem["merges"]):
        ranks.setdefault(pair, rank)
    pending = []
    def enqueue(i):
        if 0 <= i < count and alive[i] and following[i] < count:
            j = following[i]
            rank = ranks.get((tokens[i], tokens[j]))
            if rank is not None:
                heapq.heappush(pending, (rank, i, j, versions[i], versions[j]))
    for i in range(count):
        enqueue(i)
    while pending:
        rank, i, j, vi, vj = heapq.heappop(pending)
        if not alive[i] or not alive[j] or following[i] != j or versions[i] != vi or versions[j] != vj:
            continue
        tokens[i] = 256 + rank
        versions[i] += 1
        alive[j] = False
        following[i] = following[j]
        if following[j] < count:
            previous[following[j]] = i
        enqueue(previous[i])
        enqueue(i)
    return tuple(token for token, live in zip(tokens, alive) if live)

class RankedBPETokenizationTask:
    forbidden_import_roots = TOKENIZER_ROOTS
    name = "ranked_bpe_tokenization"
    task_version = "2.2.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 6000
    grading_cases = (6000, 12000)
    family_period = 3
    timing_batch = 2
    uses_reference_output = True

    def correctness_problem(self, seed):
        return self.generate_problem(96, seed)

    def generate_problem(self, n=6000, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        family = random_seed % 3  # shallow, chain, balanced
        alphabet = rng.sample(range(256), rng.randrange(32, 65))
        prefixes = [bytes(rng.choices(alphabet, k=rng.randrange(2, 7))) for _ in range(12)]
        motifs = [
            rng.choice(prefixes) + bytes(rng.choices(alphabet, k=rng.randrange(4, 30)))
            for _ in range(min(96, max(12, n // 80)))
        ]
        a, b, c = alphabet[:3]
        motifs += [bytes([a]) * rng.randrange(8, 25), bytes([a, b]) * rng.randrange(4, 12),
                   bytes([a, b, c, a, b, a]) * rng.randrange(2, 5)]
        rng.shuffle(motifs)
        merges, ranks = [], {}
        lengths = [1] * 256
        target = min(768, max(32, n // 12 + 16))
        for motif in motifs:
            if len(merges) == target:
                break
            tokens = _encode({"data": motif, "merges": tuple(merges)})
            while len(merges) < target:
                choices = [i for i in range(len(tokens) - 1)
                           if family != 0 or (tokens[i] < 256 and tokens[i + 1] < 256)]
                if not choices:
                    break
                if family == 1:
                    position = choices[0]
                elif family == 2:
                    position = min(choices, key=lambda i: lengths[tokens[i]] + lengths[tokens[i + 1]])
                else:
                    position = rng.choice(choices)
                pair = tuple(tokens[position:position + 2])
                rank = len(merges)
                ranks[pair] = rank
                merges.append(pair)
                lengths.append(lengths[pair[0]] + lengths[pair[1]])
                # Earlier rules cannot mention the new token. Advance this
                # motif with just the new rule, including overlapping pairs.
                updated = []
                position = 0
                while position < len(tokens):
                    if tokens[position:position + 2] == pair:
                        updated.append(256 + rank)
                        position += 2
                    else:
                        updated.append(tokens[position])
                        position += 1
                tokens = tuple(updated)
        while len(merges) < target:
            upper = 256 if family == 0 else 256 + len(merges)
            pair = (rng.randrange(upper), rng.randrange(upper))
            if pair not in ranks:
                ranks[pair] = len(merges)
                merges.append(pair)
                lengths.append(lengths[pair[0]] + lengths[pair[1]])
        data = bytearray()
        while len(data) < n:
            motif = bytearray(rng.choice(motifs))
            if rng.random() < (0.20 if family == 0 else 0.08):
                motif[rng.randrange(len(motif))] = rng.randrange(256)
            data.extend(motif)
        return {"data": bytes(data[:n]), "merges": tuple(merges)}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed, *, reference_output=None):
        try:
            if not plain_containers(proposed) or type(proposed) not in (tuple, list):
                return False
            if any(type(token) is not int or token < 0 for token in proposed):
                return False
            expected = _verify_encoding(problem) if reference_output is None else reference_output
            return tuple(proposed) == expected
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = RankedBPETokenizationTask()
