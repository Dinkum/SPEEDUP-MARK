"""Reference implementation for exact_integer_convolution; copied into fresh run candidates."""

def _convolve(problem):
    x, y = problem["signal_x"], problem["signal_y"]
    if not x or not y:
        return ()
    output = [0] * (len(x) + len(y) - 1)
    for i, left in enumerate(x):
        for j, right in enumerate(y):
            output[i + j] += left * right
    mode = problem["mode"]
    if mode == "full":
        return tuple(output)
    if mode == "same":
        start = (len(y) - 1) // 2
        return tuple(output[start : start + len(x)])
    if mode == "valid":
        return tuple(output[min(len(x), len(y)) - 1 : max(len(x), len(y))])
    raise ValueError("unsupported convolution mode")


def solve(problem):
    return {"convolution": _convolve(problem)}
