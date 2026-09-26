import time

import numpy as np
from iterations import (  # pyright: ignore[reportMissingImports]
    norm_inf_matrix,
    prepare,
    simple_iteration,
)
from seidel import seidel  # pyright: ignore[reportMissingImports]


def read_input(path: str) -> tuple[float, np.ndarray, np.ndarray]:
    with open(path) as f:
        eps = float(f.readline().strip())
        M = np.loadtxt(f)

    A, b = M[:, :-1], M[:, -1]
    return eps, A, b


def main():
    eps, A, b = read_input("input.txt")
    n = A.shape[0]

    alpha, beta = prepare(A, b)
    q = norm_inf_matrix(alpha)

    x_ref = np.linalg.solve(A, b)

    B = np.tril(alpha, -1)
    C = np.triu(alpha, 1)
    rho_jacobi = max(abs(np.linalg.eigvals(alpha)))
    rho_seidel = max(abs(np.linalg.eigvals(np.linalg.solve(np.eye(n) - B, C))))
    conv_note = "выполнено" if q < 1 else "НЕ выполнено, проверяем rho < 1"

    t0 = time.perf_counter()
    x_jacobi, k_jacobi, err_jacobi = simple_iteration(alpha, beta, eps)
    t_jacobi = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_seidel, k_seidel, err_seidel = seidel(alpha, beta, eps)
    t_seidel = time.perf_counter() - t0

    output = f"""
1)    Размерность системы: n = {n}
      Заданная точность: eps = {eps}
      ||alpha||_inf = {q:.6f} (достаточное условие сходимости {conv_note})
      rho(alpha) = {rho_jacobi:.6f},  rho((E-B)^-1 C) = {rho_seidel:.6f}

2)    Метод простых итераций:
      x = {x_jacobi}
      Число итераций: {k_jacobi}

3)    Метод Зейделя:
      x = {x_seidel}
      Число итераций: {k_seidel}

4)    Эталонное решение (np.linalg.solve): x = {x_ref}

5)    Невязки ||A x - b||_inf:
      простые итерации: {np.max(np.abs(A @ x_jacobi - b)):.3e}
      Зейдель:          {np.max(np.abs(A @ x_seidel - b)):.3e}

6)    Сравнение по скорости сходимости:
      итерации: {k_jacobi} (простые) против {k_seidel} (Зейдель)
      время:    {t_jacobi:.6f} c против {t_seidel:.6f} c
      оценки погрешности по итерациям (простые): {[f"{e:.3e}" for e in err_jacobi]}
      оценки погрешности по итерациям (Зейдель): {[f"{e:.3e}" for e in err_seidel]}
"""
    print(output)


if __name__ == "__main__":
    main()
