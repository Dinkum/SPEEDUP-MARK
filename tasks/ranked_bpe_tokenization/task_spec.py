"""Tokenize arbitrary bytes with an exact ranked BPE merge table."""

from __future__ import annotations

import random
import sys

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import forbidden_imports, load_candidate, plain_containers, watch_imports


_candidate = load_candidate(__file__)
TOKENIZER_ROOTS = ("tokenizers", "tiktoken", "sentencepiece", "transformers",
                   "subword_nmt", "youtokentome")


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
    name = "ranked_bpe_tokenization"
    task_version = "1.2.2"
    display_name = TASK_CATALOG[name].display_name
    default_n = 3000
    grading_cases = (3000, 6000)

    def generate_problem(self, n=3000, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        family = random_seed % 3  # shallow, chain, balanced
        alphabet = rng.sample(range(256), rng.randrange(8, 25))
        motifs = [bytes(rng.choices(alphabet, k=rng.randrange(8, 25))) for _ in range(6)]
        a, b, c = alphabet[:3]
        motifs += [bytes([a]) * rng.randrange(8, 25), bytes([a, b]) * rng.randrange(4, 12),
                   bytes([a, b, c, a, b, a]) * rng.randrange(2, 5)]
        rng.shuffle(motifs)
        merges, ranks = [], {}
        lengths = [1] * 256
        target = min(128, max(32, n // 20 + 16))
        for motif in motifs:
            # Encode after every addition: new rules are useful on the actual
            # ranked tokenization, not just hypothetical token concatenations.
            while len(merges) < target:
                tokens = _encode({"data": motif, "merges": tuple(merges)})
                choices = [i for i in range(len(tokens) - 1)
                           if family != 0 or max(tokens[i:i + 2]) < 256]
                if not choices:
                    break
                if family == 1:
                    position = choices[0]
                elif family == 2:
                    position = min(choices, key=lambda i: lengths[tokens[i]] + lengths[tokens[i + 1]])
                else:
                    position = rng.choice(choices)
                pair = tuple(tokens[position:position + 2])
                ranks[pair] = len(merges)
                merges.append(pair)
                lengths.append(lengths[pair[0]] + lengths[pair[1]])
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

    def solve(self, problem):
        return _encode(problem)

    def candidate_solve(self, problem):
        self.policy_violations = ()
        loaded = set(sys.modules)
        with watch_imports(TOKENIZER_ROOTS) as imported_during:
            result = _candidate.solve(problem, self.solve)
        violations = forbidden_imports(_candidate, TOKENIZER_ROOTS, loaded, imported_during)
        if violations:
            self.policy_violations = violations
            print(f"candidate uses forbidden tokenizer imports: {', '.join(violations)}")
            return None
        return result

    def is_solution(self, problem, proposed):
        try:
            if not plain_containers(proposed) or type(proposed) not in (tuple, list):
                return False
            if any(type(token) is not int or token < 0 for token in proposed):
                return False
            return tuple(proposed) == _verify_encoding(problem)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = RankedBPETokenizationTask()
