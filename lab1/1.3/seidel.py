import numpy as np
from iterations import (  # pyright: ignore[reportMissingImports]
    norm_inf_matrix,
    norm_inf_vector,
)


def _split_alpha(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    B = np.tril(alpha, k=-1)
    C = np.triu(alpha, k=1)
    return B, C


def _forward_substitution(L: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros(n, dtype=float)

    for i in range(n):
        y[i] = rhs[i] - sum(L[i, j] * y[j] for j in range(i))

    return y


def _iteration_matrix(alpha: np.ndarray) -> np.ndarray:
    n = alpha.shape[0]
    B, C = _split_alpha(alpha)
    E_minus_B = np.eye(n) - B

    M = np.zeros_like(alpha)
    for j in range(n):
        M[:, j] = _forward_substitution(E_minus_B, C[:, j])

    return M


def seidel(
    alpha: np.ndarray,
    beta: np.ndarray,
    eps: float,
    x0: np.ndarray | None = None,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int, list[float]]:
    n = alpha.shape[0]
    B, C = _split_alpha(alpha)

    M = _iteration_matrix(alpha)
    q = norm_inf_matrix(M)
    if q < 1:
        coef = q / (1.0 - q)
    else:
        coef = None

    if x0 is None:
        x = beta.copy()
    else:
        x = x0.copy()

    errors = []

    for k in range(1, max_iter + 1):
        x_new = np.empty(n, dtype=float)
        for i in range(n):
            x_new[i] = C[i, i:] @ x[i:] + B[i, :i] @ x_new[:i] + beta[i]

        diff = norm_inf_vector(x_new - x)
        error = coef * diff if coef is not None else diff
        errors.append(error)

        x = x_new
        if error <= eps:
            return x, k, errors

    raise RuntimeError(f"Метод не сошелся за {max_iter} итераций")
