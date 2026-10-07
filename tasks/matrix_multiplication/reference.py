"""Reference implementation for matrix_multiplication; copied into fresh run candidates."""

def solve(problem):
    a, b = problem["A"], problem["B"]
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]
