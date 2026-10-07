"""Reference implementation for rbf_interpolation; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _need():
    try:
        import numpy
        import scipy.interpolate
    except ImportError as exc:
        raise ImportError(
            "rbf_interpolation requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.interpolate


def solve(problem):
    _numpy, interpolate = _need()
    model = interpolate.RBFInterpolator(
        problem["x_train"], problem["y_train"],
        kernel=problem["rbf_config"]["kernel"],
        epsilon=problem["rbf_config"]["epsilon"],
        smoothing=problem["rbf_config"]["smoothing"],
    )
    return {"y_pred": [float(value) for value in model(problem["x_test"])]}
