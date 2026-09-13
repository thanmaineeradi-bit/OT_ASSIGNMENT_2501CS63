import numpy as np

# Big M Method
M = 1000

# Objective function:
# Max Z = 3x1 + 5x2 - M*a1 - M*a2

# Variables:
# x1, x2, s1, s2, a1, a2

A = np.array([
    [1, 1, -1, 0, 1, 0],
    [2, 1,  0, 1, 0, 0],
    [1, 3,  0, 0, 0, 1]
], dtype=float)

b = np.array([4, 6, 9], dtype=float)

c = np.array([3, 5, 0, 0, -M, -M], dtype=float)

# Initial basis: a1, s2, a2
basis = [4, 3, 5]

def print_tableau(A, b, c, basis):
    print("\nCurrent Tableau")
    print("-" * 70)

    headers = ["x1", "x2", "s1", "s2", "a1", "a2", "RHS"]
    print("Basic\t" + "\t".join(headers))

    for i in range(len(b)):
        print(
            headers[basis[i]],
            "\t",
            *[round(x, 3) for x in A[i]],
            round(b[i], 3)
        )

def big_m_simplex(A, b, c, basis):
    m, n = A.shape

    while True:

        # Calculate Cb
        cb = np.array([c[i] for i in basis])

        # Zj
        zj = cb @ A

        # Cj - Zj
        cj_zj = c - zj

        # Current objective value
        z = cb @ b

        print_tableau(A, b, c, basis)
        print("Cj-Zj:", np.round(cj_zj, 3))
        print("Z =", round(z, 3))

        # For maximization, positive Cj-Zj means improvement
        entering = np.argmax(cj_zj)

        if cj_zj[entering] <= 1e-9:
            break

        # Ratio test
        ratios = []

        for i in range(m):
            if A[i, entering] > 1e-9:
                ratios.append(b[i] / A[i, entering])
            else:
                ratios.append(np.inf)

        leaving = np.argmin(ratios)

        if ratios[leaving] == np.inf:
            print("Unbounded solution.")
            return

        print("Entering variable:", entering)
        print("Leaving variable:", basis[leaving])

        # Pivot
        pivot = A[leaving, entering]

        A[leaving] = A[leaving] / pivot
        b[leaving] = b[leaving] / pivot

        for i in range(m):
            if i != leaving:
                factor = A[i, entering]

                A[i] = A[i] - factor * A[leaving]
                b[i] = b[i] - factor * b[leaving]

        basis[leaving] = entering

    # Final solution
    solution = np.zeros(n)

    for i in range(m):
        solution[basis[i]] = b[i]

    print("\nOptimal Solution")
    print("-" * 40)

    names = ["x1", "x2", "s1", "s2", "a1", "a2"]

    for i in range(n):
        print(names[i], "=", round(solution[i], 3))

    print("Maximum Z =", round(c @ solution, 3))

    # Artificial variables should be zero
    if solution[4] > 1e-6 or solution[5] > 1e-6:
        print("No feasible solution.")
    else:
        print("Solution is feasible.")


big_m_simplex(A, b, c, basis)
