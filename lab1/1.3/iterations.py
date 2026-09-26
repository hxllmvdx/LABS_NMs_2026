import numpy as np


def prepare(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diag = np.diag(A)

    if np.any(diag == 0):
        raise ValueError(
            "На главной диагонали есть нулевые элементы: "
            "требуется перестановка уравнений."
        )

    alpha = -A / diag[:, None]
    np.fill_diagonal(alpha, 0.0)
    beta = b / diag

    return alpha, beta


def norm_inf_vector(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def norm_inf_matrix(A: np.ndarray) -> float:
    return float(np.max(np.sum(np.abs(A), axis=1)))


def simple_iteration(
    alpha: np.ndarray,
    beta: np.ndarray,
    eps: float,
    x0: np.ndarray | None = None,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int, list[float]]:
    q = norm_inf_matrix(alpha)
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
        x_new = alpha @ x + beta

        diff = norm_inf_vector(x_new - x)
        error = coef * diff if coef is not None else diff
        errors.append(error)

        x = x_new
        if error <= eps:
            return x, k, errors

    raise RuntimeError(f"Метод не сошелся за {max_iter} итераций")
