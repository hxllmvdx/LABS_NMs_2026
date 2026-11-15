import matplotlib.pyplot as plt
import numpy as np


def rhs(x: float, state: np.ndarray) -> np.ndarray:
    y, v = state
    return np.array([v, (2.0 * v + np.exp(x) * y) / (np.exp(x) + 1.0)])


def exact_solution(x: float | np.ndarray) -> float | np.ndarray:
    return np.exp(x) - 1.0 + 1.0 / (np.exp(x) + 1.0)


def make_grid(a: float, b: float, h: float) -> np.ndarray:
    n = round((b - a) / h)
    if not np.isclose(a + n * h, b):
        raise ValueError("The interval length must be divisible by h")
    return np.linspace(a, b, n + 1)


def rk4_step(f, x: float, state: np.ndarray, h: float) -> np.ndarray:
    k1 = f(x, state)
    k2 = f(x + h / 2.0, state + h * k1 / 2.0)
    k3 = f(x + h / 2.0, state + h * k2 / 2.0)
    k4 = f(x + h, state + h * k3)
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def runge_kutta4(f, x: np.ndarray, initial: np.ndarray) -> np.ndarray:
    h = x[1] - x[0]
    states = np.zeros((len(x), len(initial)))
    states[0] = initial
    for i in range(len(x) - 1):
        states[i + 1] = rk4_step(f, x[i], states[i], h)
    return states


def shooting(
    x: np.ndarray, left_derivative: float, right_derivative: float
) -> tuple[np.ndarray, float]:
    def terminal_derivative(initial_y: float) -> float:
        states = runge_kutta4(rhs, x, np.array([initial_y, left_derivative]))
        return states[-1, 1]

    first_y = 0.0
    second_y = 1.0
    first_residual = terminal_derivative(first_y) - right_derivative
    second_residual = terminal_derivative(second_y) - right_derivative
    denominator = second_residual - first_residual
    if abs(denominator) < 1e-14:
        raise ValueError("The shooting parameter cannot be determined")
    initial_y = first_y - first_residual * (second_y - first_y) / denominator
    states = runge_kutta4(rhs, x, np.array([initial_y, left_derivative]))
    return states, float(initial_y)


def solve_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs_vector: np.ndarray
) -> np.ndarray:
    n = len(diag)
    modified_upper = np.zeros(n - 1)
    modified_rhs = np.zeros(n)
    solution = np.zeros(n)

    modified_upper[0] = upper[0] / diag[0]
    modified_rhs[0] = rhs_vector[0] / diag[0]
    for i in range(1, n - 1):
        denominator = diag[i] - lower[i - 1] * modified_upper[i - 1]
        modified_upper[i] = upper[i] / denominator
        modified_rhs[i] = (
            rhs_vector[i] - lower[i - 1] * modified_rhs[i - 1]
        ) / denominator
    modified_rhs[-1] = (rhs_vector[-1] - lower[-1] * modified_rhs[-2]) / (
        diag[-1] - lower[-1] * modified_upper[-1]
    )

    solution[-1] = modified_rhs[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]
    return solution


def finite_difference(
    x: np.ndarray, left_derivative: float, right_derivative: float
) -> np.ndarray:
    h = x[1] - x[0]
    n = len(x) - 1
    a = np.exp(x) + 1.0
    b = -2.0 * np.ones_like(x)
    c = -np.exp(x)

    lower = np.zeros(n)
    diag = np.zeros(n + 1)
    upper = np.zeros(n)
    right_hand_side = np.zeros(n + 1)

    interior_lower = a[1:-1] / h**2 - b[1:-1] / (2.0 * h)
    diag[1:-1] = -2.0 * a[1:-1] / h**2 + c[1:-1]
    interior_upper = a[1:-1] / h**2 + b[1:-1] / (2.0 * h)
    lower[:-1] = interior_lower
    upper[1:] = interior_upper

    left_lower = a[0] / h**2 - b[0] / (2.0 * h)
    left_upper = a[0] / h**2 + b[0] / (2.0 * h)
    diag[0] = -2.0 * a[0] / h**2 + c[0]
    upper[0] = left_lower + left_upper
    right_hand_side[0] = 2.0 * h * left_lower * left_derivative

    right_lower = a[-1] / h**2 - b[-1] / (2.0 * h)
    right_upper = a[-1] / h**2 + b[-1] / (2.0 * h)
    lower[-1] = right_lower + right_upper
    diag[-1] = -2.0 * a[-1] / h**2 + c[-1]
    right_hand_side[-1] = -2.0 * h * right_upper * right_derivative

    return solve_tridiagonal(lower, diag, upper, right_hand_side)


def runge_romberg_error(coarse: np.ndarray, fine: np.ndarray, order: int) -> float:
    return float(np.max(np.abs(coarse - fine[::2])) / (2.0**order - 1.0))


def boundary_derivatives(y: np.ndarray, h: float) -> tuple[float, float]:
    left = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h)
    right = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * h)
    return float(left), float(right)


def plot_solutions(
    a: float,
    b: float,
    results: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    dense_x = np.linspace(a, b, 1000)
    for axis, (name, x, y) in zip(axes, results):
        axis.plot(dense_x, exact_solution(dense_x), "k", label="точное решение")
        axis.plot(x, y, "--", label=name)
        axis.scatter(x, y, s=18)
        axis.set_ylabel("y(x)")
        axis.set_title(name)
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("x")
    figure.suptitle("Краевая задача")
    figure.tight_layout()
    figure.savefig("plot_boundary.png", dpi=150)
    plt.close(figure)


def main() -> None:
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]
    a, b, h = (float(value) for value in lines[0].split())
    left_derivative = 3.0 / 4.0
    right_derivative = np.e**2 * (np.e + 2.0) / (np.e + 1.0) ** 2

    x_coarse = make_grid(a, b, h)
    x_fine = make_grid(a, b, h / 2.0)
    shooting_coarse, initial_y_coarse = shooting(
        x_coarse, left_derivative, right_derivative
    )
    shooting_fine, initial_y_fine = shooting(x_fine, left_derivative, right_derivative)
    difference_coarse = finite_difference(x_coarse, left_derivative, right_derivative)
    difference_fine = finite_difference(x_fine, left_derivative, right_derivative)

    results = [
        ("Метод стрельбы", 4, shooting_coarse[:, 0], shooting_fine[:, 0]),
        ("Конечно-разностный метод", 2, difference_coarse, difference_fine),
    ]
    output = (
        "Краевая задача: (e^x + 1)*y'' - 2*y' - e^x*y = 0\n"
        f"y'(0) = {left_derivative:.12f}, y'(1) = {right_derivative:.12f}\n"
        "Точное решение: y = e^x - 1 + 1/(e^x + 1)\n"
        f"Шаги: h = {h}, h/2 = {h / 2.0}\n"
        f"Начальное значение y(0) в методе стрельбы: {initial_y_fine:.12f}\n"
        f"Точное значение y(0): {exact_solution(0.0):.12f}\n"
    )
    for name, order, coarse, fine in results:
        exact_coarse = exact_solution(x_coarse)
        exact_fine = exact_solution(x_fine)
        error_coarse = np.abs(coarse - exact_coarse)
        error_fine = np.abs(fine - exact_fine)
        rr_error = runge_romberg_error(coarse, fine, order)
        output += f"""
{name}:
  max |y_h - y_exact|   = {np.max(error_coarse):.10e}
  max |y_h/2 - y_exact| = {np.max(error_fine):.10e}
  Оценка Рунге–Ромберга для h/2 = {rr_error:.10e}
  Ошибка в конце отрезка (h/2) = {error_fine[-1]:.10e}
  y_numeric(1) = {fine[-1]:.12f}
  y_exact(1)   = {exact_fine[-1]:.12f}
"""

    fd_left, fd_right = boundary_derivatives(difference_fine, h / 2.0)
    output += f"""
Проверка граничных условий:
  Метод стрельбы: y'(0) = {shooting_fine[0, 1]:.12f}, y'(1) = {shooting_fine[-1, 1]:.12f}
  Разностный метод: y'(0) ≈ {fd_left:.12f}, y'(1) ≈ {fd_right:.12f}
"""
    print(output)
    plot_solutions(
        a,
        b,
        [
            ("Метод стрельбы", x_fine, shooting_fine[:, 0]),
            ("Конечно-разностный метод", x_fine, difference_fine),
        ],
    )


if __name__ == "__main__":
    main()
