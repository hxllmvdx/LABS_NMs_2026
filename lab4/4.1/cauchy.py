import matplotlib.pyplot as plt
import numpy as np


def rhs(x: float, state: np.ndarray) -> np.ndarray:
    y, v = state
    return np.array([v, -2.0 * np.tan(x) * v - 3.0 * y])


def exact_solution(x: float | np.ndarray) -> float | np.ndarray:
    return np.cos(x) ** 3 + np.sin(x) * (1.0 + 2.0 * np.cos(x) ** 2)


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


def euler(f, x: np.ndarray, initial: np.ndarray) -> np.ndarray:
    h = x[1] - x[0]
    states = np.zeros((len(x), len(initial)))
    states[0] = initial
    for i in range(len(x) - 1):
        states[i + 1] = states[i] + h * f(x[i], states[i])
    return states


def runge_kutta4(f, x: np.ndarray, initial: np.ndarray) -> np.ndarray:
    h = x[1] - x[0]
    states = np.zeros((len(x), len(initial)))
    states[0] = initial
    for i in range(len(x) - 1):
        states[i + 1] = rk4_step(f, x[i], states[i], h)
    return states


def adams4(f, x: np.ndarray, initial: np.ndarray) -> np.ndarray:
    if len(x) < 4:
        raise ValueError("Adams method requires at least four grid nodes")
    h = x[1] - x[0]
    states = np.zeros((len(x), len(initial)))
    states[0] = initial
    for i in range(3):
        states[i + 1] = rk4_step(f, x[i], states[i], h)

    derivatives = [f(x[i], states[i]) for i in range(4)]
    for i in range(3, len(x) - 1):
        states[i + 1] = states[i] + h / 24.0 * (
            55.0 * derivatives[i]
            - 59.0 * derivatives[i - 1]
            + 37.0 * derivatives[i - 2]
            - 9.0 * derivatives[i - 3]
        )
        derivatives.append(f(x[i + 1], states[i + 1]))
    return states


def runge_romberg_error(coarse: np.ndarray, fine: np.ndarray, order: int) -> float:
    return float(np.max(np.abs(coarse - fine[::2])) / (2.0**order - 1.0))


def plot_solutions(
    a: float,
    b: float,
    results: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    dense_x = np.linspace(a, b, 1000)
    for axis, (name, x, states) in zip(axes, results):
        axis.plot(dense_x, exact_solution(dense_x), "k", label="точное решение")
        axis.plot(x, states[:, 0], "--", label=name)
        axis.scatter(x, states[:, 0], s=15)
        axis.set_ylabel("y(x)")
        axis.set_title(name)
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("x")
    figure.suptitle("Задача Коши")
    figure.tight_layout()
    figure.savefig("plot_cauchy.png", dpi=150)
    plt.close(figure)


def main() -> None:
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]
    a, b, h = (float(value) for value in lines[0].split())
    initial = np.array([1.0, 3.0])
    methods = [
        ("Метод Эйлера", euler, 1),
        ("Метод Рунге–Кутты 4", runge_kutta4, 4),
        ("Метод Адамса 4", adams4, 4),
    ]

    output = (
        "Задача Коши: y'' + 2*tg(x)*y' + 3*y = 0\n"
        "y(0) = 1, y'(0) = 3, x in [0, 1]\n"
        "Точное решение: y = cos^3(x) + sin(x)*(1 + 2*cos^2(x))\n"
        f"Шаги: h = {h}, h/2 = {h / 2.0}\n"
    )
    plot_results = []
    for name, method, order in methods:
        x_coarse = make_grid(a, b, h)
        x_fine = make_grid(a, b, h / 2.0)
        coarse = method(rhs, x_coarse, initial)
        fine = method(rhs, x_fine, initial)
        exact_coarse = exact_solution(x_coarse)
        exact_fine = exact_solution(x_fine)
        error_coarse = np.abs(coarse[:, 0] - exact_coarse)
        error_fine = np.abs(fine[:, 0] - exact_fine)
        rr_error = runge_romberg_error(coarse[:, 0], fine[:, 0], order)

        output += f"""
{name}:
  max |y_h - y_exact|   = {np.max(error_coarse):.10e}
  max |y_h/2 - y_exact| = {np.max(error_fine):.10e}
  Оценка Рунге–Ромберга для h/2 = {rr_error:.10e}
  Ошибка в конце отрезка (h/2) = {error_fine[-1]:.10e}
  y_numeric(1) = {fine[-1, 0]:.12f}
  y_exact(1)   = {exact_fine[-1]:.12f}
"""
        plot_results.append((name, x_fine, fine))

    print(output)
    plot_solutions(a, b, plot_results)


if __name__ == "__main__":
    main()
