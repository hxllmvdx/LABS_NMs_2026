import numpy as np


def read_input(path: str) -> tuple[float, np.ndarray]:
    with open(path) as f:
        eps = float(f.readline().strip())
        A = np.loadtxt(f)
    return eps, A


def max_offdiag(A: np.ndarray) -> tuple[int, int, float]:
    n = A.shape[0]
    i, j, mx = 0, 1, 0.0
    for p in range(n):
        for q in range(p + 1, n):
            if abs(A[p, q]) > mx:
                mx = abs(A[p, q])
                i, j = p, q
    return i, j, mx


def rotation_matrix(n: int, i: int, j: int, phi: float) -> np.ndarray:
    U = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    U[i, i] = U[j, j] = c
    U[i, j] = -s
    U[j, i] = s
    return U


def offdiag_norm(A: np.ndarray) -> float:
    n = A.shape[0]
    return np.sqrt(sum(A[p, q] ** 2 for p in range(n) for q in range(p + 1, n)))


def jacobi(
    A: np.ndarray, eps: float, max_iter: int = 10000
) -> tuple[np.ndarray, np.ndarray, int]:
    assert A.shape[0] == A.shape[1], "матрица должна быть квадратной"
    assert np.allclose(A, A.T), "метод применим только к симметрическим матрицам"

    n = A.shape[0]
    Ak = A.copy()
    V = np.eye(n)
    it = 0

    while offdiag_norm(Ak) > eps and it < max_iter:
        it += 1
        i, j, _ = max_offdiag(Ak)

        if np.isclose(Ak[i, i], Ak[j, j]):
            phi = np.pi / 4 if Ak[i, j] > 0 else -np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * Ak[i, j] / (Ak[i, i] - Ak[j, j]))

        U = rotation_matrix(n, i, j, phi)
        Ak = U.T @ Ak @ U
        V = V @ U

    return np.diag(Ak).copy(), V, it


def main():
    eps, A = read_input("input.txt")

    eigenvalues, V, iters = jacobi(A, eps)

    Lambda = np.diag(eigenvalues)
    AV = A @ V
    VL = V @ Lambda

    true_vals, _ = np.linalg.eigh(A)

    vec_lines = "\n".join(f"    x{k + 1} = {V[:, k]}" for k in range(V.shape[1]))

    output = f"""
Заданная точность eps = {eps}

Число итераций метода: {iters}

1)  Собственные значения (метод Якоби):
    {eigenvalues}

    Матрица V (столбцы — собственные векторы):
{V}

2)  Собственные векторы:
{vec_lines}
3)  Проверка A @ V = V @ Lambda:
    ||A @ V - V @ Lambda|| = {np.linalg.norm(AV - VL):.3e}
    Выполняется: {np.allclose(AV, VL, atol=max(eps, 1e-6))}

4)  Проверка ортонормированности V (V^T @ V == E):
    Выполняется: {np.allclose(V.T @ V, np.eye(A.shape[0]), atol=1e-6)}

5)  Эталон (np.linalg.eigh):
    {true_vals}
    Совпадение с эталоном: {np.allclose(np.sort(eigenvalues), true_vals, atol=max(eps, 1e-8))}
"""
    print(output)


if __name__ == "__main__":
    main()
