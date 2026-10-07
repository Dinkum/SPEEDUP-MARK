"""Reference implementation for integer_factorization; copied into fresh run candidates."""

import random


import math


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


def solve(problem):
    return _factor(problem)
