import matplotlib.pyplot as plt
import numpy as np


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(M[i, k]))
        if pivot != k:
            M[[k, pivot]] = M[[pivot, k]]

        diag = M[k, k]
        if abs(diag) < 1e-14:
            raise ValueError("Singular matrix")
        M[k, :] /= diag

        for i in range(n):
            if i != k:
                factor = M[i, k]
                M[i, :] -= factor * M[k, :]

    return M[:, -1]


def build_normal_system(
    x: np.ndarray, y: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    n = degree + 1
    A = np.zeros((n, n))
    b = np.zeros(n)
    for k in range(n):
        for j in range(n):
            A[k, j] = np.sum(x ** (k + j))
        b[k] = np.sum(x**k * y)
    return A, b


def poly_value(coeffs: np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result


def mse(coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((poly_value(coeffs, x) - y) ** 2))


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


def plot_approximations(
    x: np.ndarray,
    y: np.ndarray,
    coeffs1: np.ndarray,
    coeffs2: np.ndarray,
    coeffs3: np.ndarray,
) -> None:
    xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 500)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="red", zorder=5, label="исходные данные")
    plt.plot(xs, poly_value(coeffs1, xs), label="P_1(x)")
    plt.plot(xs, poly_value(coeffs2, xs), "--", label="P_2(x)")
    plt.plot(xs, poly_value(coeffs3, xs), ":", label="P_3(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Метод наименьших квадратов")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_least_squares.png", dpi=150)
    plt.close()


def main():
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]

    x = np.array([float(v) for v in lines[0].split()])
    y = np.array([float(v) for v in lines[1].split()])

    results = {}
    for degree in (1, 2, 3):
        A, b_vec = build_normal_system(x, y, degree)
        coeffs = solve_linear_system(A, b_vec)
        results[degree] = coeffs

    mse1 = mse(results[1], x, y)
    mse2 = mse(results[2], x, y)
    mse3 = mse(results[3], x, y)
    best = min(results, key=lambda d: mse(results[d], x, y))

    ref_coeffs = {d: np.polyfit(x, y, d)[::-1] for d in (1, 2, 3)}

    output = f"""
Исходные данные:
x = {x}
y = {y}

Коэффициенты многочлена первой степени:
{results[1]}
P_1(x) = {format_poly(results[1])}
MSE1 = {mse1:.6e}

Коэффициенты многочлена второй степени:
{results[2]}
P_2(x) = {format_poly(results[2])}
MSE2 = {mse2:.6e}

Коэффициенты многочлена третьей степени:
{results[3]}
P_3(x) = {format_poly(results[3])}
MSE3 = {mse3:.6e}

Сравнение ошибок:
MSE1 = {mse1:.6e}
MSE2 = {mse2:.6e}
MSE3 = {mse3:.6e}
Лучший многочлен: P_{best}(x) (наименьшая MSE)

Совпадение с numpy.polyfit:
P_1: {np.allclose(results[1], ref_coeffs[1])}
P_2: {np.allclose(results[2], ref_coeffs[2])}
P_3: {np.allclose(results[3], ref_coeffs[3])}
"""

    print(output)
    plot_approximations(x, y, results[1], results[2], results[3])


if __name__ == "__main__":
    main()
