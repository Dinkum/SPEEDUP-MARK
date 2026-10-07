"""Reference implementation for group_lasso; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _need():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "group_lasso requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy


def _groups(labels):
    found = {}
    for index, label in enumerate(labels):
        found.setdefault(label, []).append(index)
    return [(indexes, len(indexes) ** 0.5) for indexes in found.values()]


def _logistic(features, labels, beta):
    numpy = _need()
    labels = numpy.asarray(labels, dtype=float)
    linear = features @ beta
    positive = linear >= 0
    softplus = numpy.where(positive, linear + numpy.log1p(numpy.exp(-linear)), numpy.log1p(numpy.exp(linear)))
    loss = float(numpy.sum(softplus) - labels @ linear)
    sigmoid = numpy.where(positive, 1.0 / (1.0 + numpy.exp(-linear)), numpy.exp(linear) / (1.0 + numpy.exp(linear)))
    return loss, features.T @ (sigmoid - labels)


def _lipschitz(features):
    numpy = _need()
    vector = numpy.ones(features.shape[1])
    vector /= numpy.linalg.norm(vector)
    value = 0.0
    for _ in range(40):
        vector = features.T @ (features @ vector)
        value = float(numpy.linalg.norm(vector))
        vector /= value
    return max(value / 4.0, 1e-8)


def _apply_prox(coefficients, labels, step, penalty):
    numpy = _need()
    updated = coefficients.copy()
    for indexes, weight in _groups(labels):
        slot = [index + 1 for index in indexes]
        vector = updated[slot]
        norm = float(numpy.linalg.norm(vector))
        limit = step * penalty * weight
        if norm <= limit:
            updated[slot] = 0.0
        else:
            updated[slot] *= 1.0 - limit / norm
    return updated


def _objective(features, labels, group_labels, penalty, beta):
    numpy = _need()
    loss, _grad = _logistic(features, labels, beta)
    extra = 0.0
    for indexes, weight in _groups(group_labels):
        extra += weight * float(numpy.linalg.norm(beta[[index + 1 for index in indexes]]))
    return loss + penalty * extra


def _stationary(features, labels, group_labels, penalty, beta):
    numpy = _need()
    _loss, gradient = _logistic(features, labels, beta)
    scale = 1.0 + float(numpy.linalg.norm(gradient))
    tolerance = 1e-4 * scale
    if abs(float(gradient[0])) > tolerance:
        return False
    slopes = gradient[1:]
    for indexes, weight in _groups(group_labels):
        block = beta[[index + 1 for index in indexes]]
        slope = slopes[indexes]
        norm = float(numpy.linalg.norm(block))
        if norm <= 1e-8:
            if float(numpy.linalg.norm(slope)) > penalty * weight + tolerance:
                return False
        elif float(numpy.linalg.norm(slope + penalty * weight * block / norm)) > tolerance:
            return False
    return True


def solve(problem):
    numpy = _need()
    features = numpy.asarray(problem["X"], dtype=float)
    labels = numpy.asarray(problem["y"], dtype=float)
    penalty = float(problem["lba"])
    step = 1.0 / _lipschitz(features)
    current = numpy.zeros(features.shape[1])
    extrapolated = current.copy()
    momentum = 1.0
    for _ in range(12000):
        _loss, gradient = _logistic(features, labels, extrapolated)
        nxt = _apply_prox(extrapolated - step * gradient, problem["gl"], step, penalty)
        momentum_next = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        extrapolated = nxt + ((momentum - 1.0) / momentum_next) * (nxt - current)
        current = nxt
        momentum = momentum_next
        if _stationary(features, labels, problem["gl"], penalty, current):
            break
    if not _stationary(features, labels, problem["gl"], penalty, current):
        raise ValueError("group lasso reference did not reach stationarity")
    value = _objective(features, labels, problem["gl"], penalty, current)
    return {
        "beta0": float(current[0]),
        "beta": [float(value) for value in current[1:]],
        "optimal_value": float(value),
    }
