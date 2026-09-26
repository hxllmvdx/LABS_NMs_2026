import numpy as np


def _find_pivot(A: np.ndarray, start: int) -> int:
    mx = abs(A[start, start])
    pivot = start
    for i in range(start, A.shape[0]):
        if abs(A[i, start]) > mx:
            mx = abs(A[i, start])
            pivot = i

    return pivot


def create_LUP(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    swaps = 0

    for k in range(n - 1):
        pivot = _find_pivot(U, k)
        S = np.eye(n)

        if pivot != k:
            S[[k, pivot]] = S[[pivot, k]]
            U[[k, pivot]] = U[[pivot, k]]
            swaps += 1

        P = S @ P
        M_rev = np.eye(n)

        for i in range(k + 1, A.shape[0]):
            mu = U[i, k] / U[k, k]
            U[i, :] -= mu * U[k, :]
            M_rev[i, k] = mu
        L = S @ L @ S @ M_rev

    return L, U, P, swaps


def _solve_lz_eq_b(L: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.zeros_like(L[:, 0])
    new_b = P @ b
    for i in range(L.shape[0]):
        s = new_b[i] - sum(L[i, j] * z[j] for j in range(i))
        z[i] = s / L[i, i]
    return z


def solve_system(
    L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray
) -> np.ndarray:
    z = _solve_lz_eq_b(L, P, b)
    x = np.zeros_like(U[:, 0])
    for i in range(U.shape[0] - 1, -1, -1):
        s = z[i] - sum(U[i, j] * x[j] for j in range(i + 1, U.shape[0]))
        x[i] = s / U[i, i]
    return x


def inverse_from_lu(L: np.ndarray, U: np.ndarray, P: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    A_inv = np.zeros_like(L)

    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        x = solve_system(L, U, P, e)

        for i in range(n):
            A_inv[i, j] = x[i]

    return A_inv


def det_from_lu(U: np.ndarray, swaps: int) -> float:
    d = np.prod(np.diag(U), dtype=float)
    return -d if swaps % 2 else d


def main():
    M = np.loadtxt("input.txt")
    A, b = M[:, :-1], M[:, -1]

    L, U, P, swaps = create_LUP(A)
    assert np.allclose(L @ U, P @ A)

    x = solve_system(L, U, P, b)
    assert np.allclose(A @ x, b)

    x_ref = np.linalg.solve(A, b)

    A_inv = inverse_from_lu(L, U, P)
    assert np.allclose(A @ A_inv, np.eye(A.shape[0]))

    det_A = det_from_lu(U, swaps)

    output = f"""
1)    L-matrix =\n{L}

      U-matrix =\n{U}    

2)    L*U =\n{L @ U} 

3)    x = {x}

4)    A^-1 =\n{A_inv}

5)    det(A) = {det_A}

6)    A @ A^-1 == E: {np.allclose(A @ A_inv, np.eye(A.shape[0]))}

7)    L @ U == P @ A: {np.allclose(L @ U, P @ A)}

8)    Совпадение с эталоном (numpy.linalg.solve): {np.allclose(x, x_ref)}
"""

    print(output)


if __name__ == "__main__":
    main()
