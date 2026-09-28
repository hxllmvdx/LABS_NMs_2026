import matplotlib.pyplot as plt
import numpy as np


def f(x: float) -> float:
    return np.log(x + 1) - 2 * x + 0.5


def df(x: float) -> float:
    return 1 / (x + 1) - 2


def phi(x: float) -> float:
    return (np.log(x + 1) + 0.5) / 2


def simple_iteration(x0: float, eps: float, max_iter: int = 1000) -> tuple[float, list]:
    xs = [x0]
    for _ in range(max_iter):
        x_next = phi(xs[-1])
        xs.append(x_next)
        if abs(x_next - xs[-2]) < eps:
            break
    return xs[-1], xs


def newton(x0: float, eps: float, max_iter: int = 1000) -> tuple[float, list]:
    xs = [x0]
    for _ in range(max_iter):
        x_next = xs[-1] - f(xs[-1]) / df(xs[-1])
        xs.append(x_next)
        if abs(x_next - xs[-2]) < eps:
            break
    return xs[-1], xs


def plot_function() -> None:
    x = np.linspace(-0.9, 3, 500)
    plt.figure(figsize=(8, 5))
    plt.plot(x, f(x), label=r"$f(x) = \ln(x+1) - 2x + 0.5$")  # pyright: ignore[reportArgumentType]
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График функции")
    plt.savefig("plot_f.png", dpi=150)
    plt.show()


def plot_errors(si_xs: list, nw_xs: list, x_ref: float) -> None:
    plt.figure(figsize=(8, 5))
    for xs, name in [(si_xs, "простая итерация"), (nw_xs, "Ньютон")]:
        iters = np.arange(len(xs))
        err = [abs(x - x_ref) for x in xs]
        plt.semilogy(iters, err, "o-", label=name)
    plt.xlabel("Номер итерации k")
    plt.ylabel(r"$|x^{(k)} - x^*|$")
    plt.title("Зависимость погрешности от количества итераций")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("plot_errors.png", dpi=150)
    plt.show()


def main():
    plot_function()

    eps = float(input("Точность вычислений eps: "))

    x0 = float(input("Начальное приближение x0: "))

    x_si, si_xs = simple_iteration(x0, eps)
    x_nw, nw_xs = newton(x0, eps)

    x_ref = 0.0
    for _ in range(100):
        x_ref = phi(x_ref)
    x_ref = x_ref - f(x_ref) / df(x_ref)

    output = f"""
1)    Начальное приближение (графически): x0 = {x0}

2)    Метод простой итерации:
      корень x = {x_si}
      итераций: {len(si_xs) - 1}
      последние приближения: {si_xs[-4:]}

3)    Метод Ньютона:
      корень x = {x_nw}
      итераций: {len(nw_xs) - 1}
      последние приближения: {nw_xs[-4:]}

4)    Совпадение с эталоном (уточненный корень {x_ref}):
      простая итерация: {abs(x_si - x_ref) < eps}
      Ньютон:           {abs(x_nw - x_ref) < eps}

5)    Погрешность |x - x*|:
      простая итерация: {abs(x_si - x_ref):.3e}
      Ньютон:           {abs(x_nw - x_ref):.3e}
"""

    print(output)

    plot_errors(si_xs, nw_xs, x_ref)


if __name__ == "__main__":
    main()
