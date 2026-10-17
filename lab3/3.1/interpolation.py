import matplotlib.pyplot as plt
import numpy as np


def f(x: float | np.ndarray) -> float | np.ndarray:
    return np.cos(x) + x


def lagrange_basis(x_nodes: np.ndarray, i: int, x: float) -> float:
    result = 1.0
    for j in range(len(x_nodes)):
        if j != i:
            result *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return result


def lagrange_poly(x_nodes: np.ndarray, y_nodes: np.ndarray, x: float) -> float:
    return sum(y_nodes[i] * lagrange_basis(x_nodes, i, x) for i in range(len(x_nodes)))


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros(len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] += a[i] * b[j]
    return result


def divided_differences(x_nodes: np.ndarray, y_nodes: np.ndarray) -> list[list[float]]:
    n = len(x_nodes)
    table: list[list[float]] = [y_nodes.tolist()]
    for k in range(1, n):
        row = []
        for i in range(n - k):
            row.append(
                (table[k - 1][i + 1] - table[k - 1][i]) / (x_nodes[i + k] - x_nodes[i])
            )
        table.append(row)
    return table


def newton_poly(x_nodes: np.ndarray, table: list[list[float]], x: float) -> float:
    result = table[0][0]
    term = 1.0
    for k in range(1, len(x_nodes)):
        term *= x - x_nodes[k - 1]
        result += table[k][0] * term
    return result


def _to_standard_lagrange(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    n = len(x_nodes)
    coeffs = np.zeros(n)
    for i in range(n):
        denom = 1.0
        for j in range(n):
            if j != i:
                denom *= x_nodes[i] - x_nodes[j]
        c = np.array([1.0])
        for j in range(n):
            if j != i:
                c = _convolve(c, [-x_nodes[j], 1.0])  # pyright: ignore[reportArgumentType]
        coeffs += y_nodes[i] * c / denom
    return coeffs


def _to_standard_newton(x_nodes: np.ndarray, table: list[list[float]]) -> np.ndarray:
    n = len(x_nodes)
    coeffs = np.zeros(n)
    coeffs[0] = table[0][0]
    for k in range(1, n):
        term = np.array([table[k][0]])
        for j in range(k):
            term = _convolve(term, [-x_nodes[j], 1.0])  # pyright: ignore[reportArgumentType]
        for i in range(k + 1):
            coeffs[i] += term[i]
    return coeffs


def format_poly(coeffs: np.ndarray) -> str:
    parts = []
    for i in range(len(coeffs) - 1, -1, -1):
        a = coeffs[i]
        if abs(a) < 1e-14:
            continue
        if i == 0:
            term = f"{a:.6f}"
        elif i == 1:
            term = f"{a:.6f}*x"
        else:
            term = f"{a:.6f}*x^{i}"
        if parts and a > 0:
            parts.append(" + " + term)
        elif parts and a < 0:
            parts.append(" - " + term.lstrip("-"))
        else:
            parts.append(term)
    return "".join(parts) if parts else "0"


def print_divided_diff_table(x_nodes: np.ndarray, table: list[list[float]]) -> str:
    lines = ["i    x_i        f(x_i)     разности"]
    n = len(x_nodes)
    for i in range(n):
        row_parts = [f"{i:<4} {x_nodes[i]:<10.6f}"]
        for k in range(n - i):
            row_parts.append(f"{table[k][i]:<12.6f}")
        lines.append("  ".join(row_parts))
    return "\n".join(lines)


def plot_variant(
    variant: int,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    table: list[list[float]],
    x_star: float,
    y_lagrange: float,
    y_newton: float,
) -> None:
    x_min, x_max = x_nodes.min() - 0.5, x_nodes.max() + 0.5
    xs = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, f(xs), label=r"$f(x) = \cos(x) + x$")
    plt.plot(
        xs,
        [lagrange_poly(x_nodes, y_nodes, x) for x in xs],
        "--",
        label="многочлен Лагранжа",
    )
    plt.plot(
        xs,
        [newton_poly(x_nodes, table, x) for x in xs],
        ":",
        label="многочлен Ньютона",
    )
    plt.scatter(x_nodes, y_nodes, color="red", zorder=5, label="узлы")
    plt.scatter(
        [x_star],
        [y_lagrange],
        color="green",
        marker="s",
        zorder=5,
        label=r"$x^*$ (Лагранж)",
    )
    plt.scatter(
        [x_star],
        [y_newton],
        color="purple",
        marker="^",
        zorder=5,
        label=r"$x^*$ (Ньютон)",
    )
    plt.axvline(x_star, color="gray", linewidth=0.8, linestyle="-.")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Интерполяция, вариант {'а' if variant == 0 else 'б'}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_variant_{'a' if variant == 0 else 'b'}.png", dpi=150)
    plt.close()


def process_variant(variant: int, x_star: float, x_nodes: np.ndarray) -> str:
    y_nodes = f(x_nodes)
    table = divided_differences(x_nodes, y_nodes)  # pyright: ignore[reportArgumentType]

    y_lagrange = lagrange_poly(x_nodes, y_nodes, x_star)  # pyright: ignore[reportArgumentType]
    y_newton = newton_poly(x_nodes, table, x_star)
    y_exact = float(f(x_star))
    abs_err_lagrange = abs(y_lagrange - y_exact)
    abs_err_newton = abs(y_newton - y_exact)
    diff_methods = abs(y_lagrange - y_newton)

    coeffs_lagrange = _to_standard_lagrange(x_nodes, y_nodes)  # pyright: ignore[reportArgumentType]
    coeffs_newton = _to_standard_newton(x_nodes, table)

    plot_variant(variant, x_nodes, y_nodes, table, x_star, y_lagrange, y_newton)  # pyright: ignore[reportArgumentType]

    label = "а" if variant == 0 else "б"
    return f"""
--- Вариант {label} ---
Исходные узлы интерполяции:
x_i = {x_nodes}
y_i = {y_nodes}
x*  = {x_star}

Интерполяционный многочлен Лагранжа (в стандартном базисе):
L(x) = {format_poly(coeffs_lagrange)}

Таблица разделённых разностей:
{print_divided_diff_table(x_nodes, table)}

Интерполяционный многочлен Ньютона (в стандартном базисе):
N(x) = {format_poly(coeffs_newton)}

Значение многочлена Лагранжа в точке x*: {y_lagrange:.10f}
Значение многочлена Ньютона в точке x*:  {y_newton:.10f}
Точное значение f(x*):                    {y_exact:.10f}
Абсолютная погрешность Лагранжа:          {abs_err_lagrange:.3e}
Абсолютная погрешность Ньютона:           {abs_err_newton:.3e}
Разность между результатами методов:      {diff_methods:.3e}
"""


def main():
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]

    output_parts = []
    for variant, line in enumerate(lines):
        values = [float(v) for v in line.split()]
        x_star = values[0]
        x_nodes = np.array(values[1:])
        output_parts.append(process_variant(variant, x_star, x_nodes))

    print("\n".join(output_parts))


if __name__ == "__main__":
    main()
