"""A dependency-free discrete logarithm challenge inspired by AlgoTune."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_factor_powers = _reference._factor_powers


def _prime(value):
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
            while not _prime(odd):
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
        if _prime(value):
            return value
    raise ValueError("could not construct a prime of the requested bit length")


class DiscreteLogTask:
    name = "discrete_log"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 50
    grading_cases = (50, 60)

    def workload_family(self, n, random_seed=0):
        return ("smooth", "medium_subgroup", "large_subgroup")[random_seed % 3]

    def generate_problem(self, n=50, random_seed=0):
        if type(n) is not int or not 12 <= n <= 62:
            raise ValueError("n must be a modulus bit length from 12 through 62")
        rng = random.Random(random_seed)
        family = random_seed % 3
        large_bits = (0, min(17, n - 6), min(29, n - 6))[family]
        p = _smooth_prime(n, rng, large_bits)
        divisors = [factor for factor, _ in _factor_powers(p - 1)]
        while True:
            g = rng.randrange(2, p)
            if all(pow(g, (p - 1) // factor, p) != 1 for factor in divisors):
                break
        return {"p": p, "g": g, "h": pow(g, rng.randrange(p - 1), p)}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        return (
            type(proposed) is dict and set(proposed) == {"x"}
            and type(proposed["x"]) is int
            and 0 <= proposed["x"] < problem["p"] - 1
            and pow(problem["g"], proposed["x"], problem["p"]) == problem["h"]
        )


TASK = DiscreteLogTask()
