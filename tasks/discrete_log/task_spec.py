"""A dependency-free discrete logarithm challenge inspired by AlgoTune."""

import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


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


def _factor_powers(value):
    factors = []
    divisor = 2
    while divisor * divisor <= value:
        power = 0
        while value % divisor == 0:
            value //= divisor
            power += 1
        if power:
            factors.append((divisor, power))
        divisor = 3 if divisor == 2 else divisor + 2
    if value > 1:
        factors.append((value, 1))
    return factors


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


def _baby_steps(base, order, modulus):
    width = math.isqrt(order) + 1
    table, value = {}, 1
    for exponent in range(width):
        table.setdefault(value, exponent)
        value = value * base % modulus
    return width, table, pow(base, -width, modulus)


def _subgroup_log(target, order, modulus, prepared):
    width, table, stride = prepared
    for giant in range((order + width - 1) // width):
        if target in table:
            answer = giant * width + table[target]
            if answer < order:
                return answer
        target = target * stride % modulus
    raise ValueError("target outside subgroup")


def _pohlig_hellman(problem):
    modulus, base, target = problem["p"], problem["g"], problem["h"]
    order = modulus - 1
    answer, combined = 0, 1
    for prime, power in _factor_powers(order):
        subgroup_base = pow(base, order // prime, modulus)
        prepared = _baby_steps(subgroup_base, prime, modulus)
        residue, place = 0, 1
        for _ in range(power):
            adjusted = target * pow(base, -residue, modulus) % modulus
            digit_target = pow(adjusted, order // (place * prime), modulus)
            digit = _subgroup_log(digit_target, prime, modulus, prepared)
            residue += digit * place
            place *= prime
        # Incremental CRT on pairwise-coprime prime powers.
        answer += combined * ((residue - answer) * pow(combined, -1, place) % place)
        combined *= place
    return {"x": answer % order}


class DiscreteLogTask:
    name = "discrete_log"
    task_version = "1.3.0"
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

    def solve(self, problem):
        return _pohlig_hellman(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        return (
            type(proposed) is dict and set(proposed) == {"x"}
            and type(proposed["x"]) is int
            and 0 <= proposed["x"] < problem["p"] - 1
            and pow(problem["g"], proposed["x"], problem["p"]) == problem["h"]
        )


TASK = DiscreteLogTask()
