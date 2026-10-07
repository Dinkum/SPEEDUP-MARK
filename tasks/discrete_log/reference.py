"""Reference implementation for discrete_log; copied into fresh run candidates."""

import math


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


def solve(problem):
    return _pohlig_hellman(problem)
