import numpy as np


def f(x: float | np.ndarray) -> float | np.ndarray:
    return x**2 / (x**3 - 27.0)


def left_rectangles(a: float, b: float, h: float) -> float:
    n = round((b - a) / h)
    return h * sum(f(a + i * h) for i in range(n))  # pyright: ignore[reportReturnType]


def right_rectangles(a: float, b: float, h: float) -> float:
    n = round((b - a) / h)
    return h * sum(f(a + (i + 1) * h) for i in range(n))  # pyright: ignore[reportReturnType]


def middle_rectangles(a: float, b: float, h: float) -> float:
    n = round((b - a) / h)
    return h * sum(f(a + (i + 0.5) * h) for i in range(n))  # pyright: ignore[reportReturnType]


def trapezoid(a: float, b: float, h: float) -> float:
    n = round((b - a) / h)
    inner = sum(f(a + i * h) for i in range(1, n))
    return h * (0.5 * f(a) + inner + 0.5 * f(b))  # pyright: ignore[reportReturnType]


def simpson(a: float, b: float, h: float) -> float:
    n = round((b - a) / h)
    if n % 2 != 0:
        raise ValueError("Simpson method requires even number of intervals")
    odd = sum(f(a + i * h) for i in range(1, n, 2))
    even = sum(f(a + i * h) for i in range(2, n, 2))
    return h / 3.0 * (f(a) + f(b) + 4.0 * odd + 2.0 * even)  # pyright: ignore[reportReturnType]


def runge_romberg(i_h1: float, i_h2: float, p: int) -> float:
    return i_h2 + (i_h2 - i_h1) / (2.0**p - 1.0)


def error_estimate(i_h1: float, i_h2: float, p: int) -> float:
    return abs(i_h2 - i_h1) / (2.0**p - 1.0)


def exact_integral(a: float, b: float) -> float:
    F = lambda x: np.log(np.abs(x**3 - 27.0)) / 3.0
    return F(b) - F(a)


def main():
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]

    a, b, h1, h2 = (float(v) for v in lines[0].split())

    methods = [
        ("Левые прямоугольники", left_rectangles, 1),
        ("Правые прямоугольники", right_rectangles, 1),
        ("Средние прямоугольники", middle_rectangles, 2),
        ("Трапеции", trapezoid, 2),
        ("Симпсон", simpson, 4),
    ]

    exact = exact_integral(a, b)
    results = []
    for name, method, p in methods:
        i_h1 = method(a, b, h1)
        i_h2 = method(a, b, h2)
        refined = runge_romberg(i_h1, i_h2, p)
        err_est = error_estimate(i_h1, i_h2, p)
        abs_err_h1 = abs(i_h1 - exact)
        abs_err_h2 = abs(i_h2 - exact)
        results.append((name, p, i_h1, i_h2, refined, err_est, abs_err_h1, abs_err_h2))

    most_accurate = min(results, key=lambda r: r[7])

    output = f"""
Интеграл: F = integral_{{{a}}}^{{{b}}} x^2 / (x^3 - 27) dx
Точное значение: {exact:.10f}

{"Метод":<26}{"p":<4} {"I(h1)":<14} {"I(h2)":<14} {"I_уточн":<14}{"err_est":<12}  {"|err| h1":<12}  {"|err| h2":<12}
"""
    for name, p, i_h1, i_h2, refined, err_est, abs_err_h1, abs_err_h2 in results:
        output += (
            f"{name:<26}{p:<4} {i_h1:<14.8f} {i_h2:<14.8f} {refined:<14.8f}"
            f"{err_est:<12.6e}  {abs_err_h1:<12.6e}  {abs_err_h2:<12.6e}\n"
        )

    output += f"""
Наиболее точный метод (по |err| h2): {most_accurate[0]}
"""

    print(output)


if __name__ == "__main__":
    main()
