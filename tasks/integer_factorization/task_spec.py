"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

import math


def prime(value):
    """Deterministic Miller-Rabin for integers below 2**64."""
    if not 2 <= value < 2**64:
        return False
    for divisor in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % divisor == 0:
            return value == divisor
    odd, shifts = value - 1, 0
    while odd % 2 == 0:
        odd //= 2
        shifts += 1
    # This base set is exhaustive over the declared 64-bit domain, rather
    # than a randomized probable-prime check in an exact-answer grader.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        residue = pow(base, odd, value)
        if residue in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            residue = residue * residue % value
            if residue == value - 1:
                break
        else:
            return False
    return True


def _smooth_prime(bits, rng, large_bits=0):
    """Prime with a varied, bounded-factor group order; no answer metadata."""
    small = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    for _ in range(4096):
        odd = 1
        if large_bits:
            odd = rng.randrange(1 << (large_bits - 1), 1 << large_bits) | 1
            while not prime(odd):
                odd += 2
        target = rng.randrange(max(3, bits // 2), bits - 1)
        while odd.bit_length() < target:
            factor = rng.choice(small)
            if (odd * factor).bit_length() > bits - 1:
                break
            odd *= factor
        shift = bits - odd.bit_length()
        if shift < 1:
            continue
        value = (odd << shift) + 1
        if prime(value):
            return value
    raise ValueError("could not construct a prime of the requested bit length")


def _random_prime(bits, rng):
    while True:
        value = rng.randrange(1 << (bits - 1), 1 << bits) | 1
        while value < 1 << bits:
            if prime(value):
                return value
            value += 2


def _factor(problem):
    """Fermat for near-square inputs, then deterministic-replay Pollard rho."""
    value = problem["composite"]
    if value % 2 == 0:
        return {"p": 2, "q": value // 2}
    root = math.isqrt(value)
    root += root * root < value
    for _ in range(128):
        square = root * root - value
        difference = math.isqrt(square)
        if difference * difference == square and root > difference + 1:
            p, q = root - difference, root + difference
            if p != q:
                return {"p": p, "q": q}
        root += 1
    rng = random.Random(value)
    while True:
        constant, x = rng.randrange(1, value), rng.randrange(2, value)
        y = x
        for _ in range(200000):
            x = (x * x + constant) % value
            y = (y * y + constant) % value
            y = (y * y + constant) % value
            divisor = math.gcd(abs(x - y), value)
            if 1 < divisor < value:
                return {"p": min(divisor, value // divisor), "q": max(divisor, value // divisor)}
            if divisor == value:
                break


class Task:
    name = "integer_factorization"
    task_version = "1.3.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 64
    grading_cases = (64, 80)

    def workload_family(self, n, random_seed=0):
        return ("balanced", "close", "smooth_factor", "unbalanced")[random_seed % 4]

    def generate_problem(self, n=64, random_seed=0):
        if type(n) is not int or not 24 <= n <= 80:
            raise ValueError("n must be a composite bit budget from 24 through 80")
        rng = random.Random(random_seed)
        family = random_seed % 4
        small_bits = min(32, n // 2) if family != 3 else min(20, n // 3)
        if family == 1:
            p = _random_prime(n // 2, rng)
            q = p + 2 * rng.randrange(1, 513)
            while not prime(q):
                q += 2
        elif family == 2:
            p = _smooth_prime(small_bits, rng)
            q = _random_prime(n - small_bits, rng)
        else:
            p = _random_prime(small_bits, rng)
            q = _random_prime(n - small_bits, rng)
        while q == p:
            q = _random_prime(n - p.bit_length(), rng)
        return {"composite": p * q}

    def solve(self, problem):
        return _factor(problem)

    def is_solution(self, problem, proposed):
        return (type(proposed) is dict and set(proposed) == {"p", "q"}
                and type(proposed["p"]) is int and type(proposed["q"]) is int
                and proposed["p"] < proposed["q"]
                and proposed["p"] * proposed["q"] == problem["composite"]
                and prime(proposed["p"]) and prime(proposed["q"]))

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
