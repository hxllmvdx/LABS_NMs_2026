import matplotlib.pyplot as plt
import numpy as np


def solve_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    n = len(diag)
    c = np.zeros(n - 1)
    d = np.zeros(n)
    x = np.zeros(n)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    d[-1] = (rhs[-1] - lower[-1] * d[-2]) / (diag[-1] - lower[-1] * c[-2])

    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def build_natural_spline(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    n = len(x) - 1
    h = np.diff(x)

    mu = h[:-1] / (h[:-1] + h[1:])
    lam = 1.0 - mu
    d = (
        6.0
        * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        / (h[:-1] + h[1:])
    )

    lower = mu[1:]
    diag = 2.0 * np.ones(n - 1)
    upper = lam[:-1]
    rhs = d

    M_inner = solve_tridiagonal(lower, diag, upper, rhs)
    M = np.zeros(n + 1)
    M[1:-1] = M_inner

    a = y[:-1]
    b = (y[1:] - y[:-1]) / h - h * (2.0 * M[:-1] + M[1:]) / 6.0
    c = M[:-1] / 2.0
    d = (M[1:] - M[:-1]) / (6.0 * h)

    return a, b, c, d, M


def spline_value(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, x0: float
) -> tuple[float, int]:
    for i in range(len(x) - 1):
        if x[i] <= x0 <= x[i + 1]:
            dx = x0 - x[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3, i
    raise ValueError("x0 is outside spline interval")


def format_poly(a: float, b: float, c: float, d: float) -> str:
    parts = []
    for coeff, label in [
        (a, ""),
        (b, "*(x-x_i)"),
        (c, "*(x-x_i)^2"),
        (d, "*(x-x_i)^3"),
    ]:
        if abs(coeff) < 1e-14:
            continue
        parts.append(f"{coeff:.6f}{label}")
    return " + ".join(parts) if parts else "0"


def plot_spline(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    x0: float,
    s0: float,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="red", zorder=5, label="исходные узлы")

    for i in range(len(x) - 1):
        xs = np.linspace(x[i], x[i + 1], 100)
        dx = xs - x[i]
        ys = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        plt.plot(xs, ys, label=f"S_{i}(x)")

    plt.scatter(
        [x0], [s0], color="green", marker="s", zorder=5, label=r"$(x^*, S(x^*))$"
    )
    plt.axvline(x0, color="gray", linewidth=0.8, linestyle="-.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Кубический сплайн с нулевой кривизной на концах")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_spline.png", dpi=150)
    plt.close()


def main():
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]

    x_star = float(lines[0])
    x = np.array([float(v) for v in lines[1].split()])
    y = np.array([float(v) for v in lines[2].split()])

    a, b, c, d, M = build_natural_spline(x, y)
    s_star, interval = spline_value(x, a, b, c, d, x_star)

    check_nodes = [spline_value(x, a, b, c, d, xi)[0] for xi in x]

    output = f"""
Исходные узлы и значения функции:
x = {x}
y = {y}
x* = {x_star}

Коэффициенты сплайна S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3:
"""
    for i in range(len(x) - 1):
        output += (
            f"  i={i}: a={a[i]:.6f}, b={b[i]:.6f}, c={c[i]:.6f}, d={d[i]:.6f}, "
            f"интервал [{x[i]}, {x[i + 1]}]\n"
            f"         S_{i}(x) = {format_poly(a[i], b[i], c[i], d[i])}\n"
        )

    output += f"""
Значение сплайна в точке x*: {s_star:.10f}
Номер интервала, содержащего x*: {interval}

Проверка прохождения сплайна через узлы:
x_i       y_i        S(x_i)
"""
    for xi, yi, si in zip(x, y, check_nodes):
        output += f"{xi:<10.6f}{yi:<10.6f}{si:<12.10f}\n"

    output += f"""
Значения второй производной на концах интервала:
S''(x_0) = {M[0]:.3e} (должно быть 0)
S''(x_n) = {M[-1]:.3e} (должно быть 0)
"""

    print(output)
    plot_spline(x, y, a, b, c, d, x_star, s_star)


if __name__ == "__main__":
    main()
