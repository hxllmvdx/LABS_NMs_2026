import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def make_f(a: float):
    def f(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                x[0] ** 2 / a**2 + x[1] ** 2 / (a / 2) ** 2 - 1,
                a * x[1] - np.exp(x[0]) - x[0],
            ]
        )

    return f


def make_jacobian(a: float):
    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [2 * x[0] / a**2, 8 * x[1] / a**2],
                [-np.exp(x[0]) - 1, a],
            ]
        )

    return jacobian


def make_phi(a: float):
    def phi(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.log(a * x[1] - x[0]),
                (a / 2) * np.sqrt(1 - (x[0] / a) ** 2),
            ]
        )

    return phi


def simple_iteration(
    phi, x0: np.ndarray, eps: float, max_iter: int = 1000
) -> tuple[np.ndarray, list]:
    xs = [x0.copy()]
    for _ in range(max_iter):
        x_next = phi(xs[-1])
        xs.append(x_next)
        if np.linalg.norm(x_next - xs[-2], ord=np.inf) < eps:
            break
    return xs[-1], xs


def newton(
    f, jacobian, x0: np.ndarray, eps: float, max_iter: int = 1000
) -> tuple[np.ndarray, list]:
    xs = [x0.copy()]
    for _ in range(max_iter):
        dx = np.linalg.solve(jacobian(xs[-1]), -f(xs[-1]))
        x_next = xs[-1] + dx
        xs.append(x_next)
        if np.linalg.norm(x_next - xs[-2], ord=np.inf) < eps:
            break
    return xs[-1], xs


def plot_system(a: float) -> None:
    X1, X2 = np.meshgrid(np.linspace(-a - 0.5, a + 0.5, 500), np.linspace(-a, a, 500))
    F1 = X1**2 / a**2 + X2**2 / (a / 2) ** 2 - 1
    F2 = a * X2 - np.exp(X1) - X1

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, F1, levels=[0], colors="tab:blue")
    plt.contour(X1, X2, F2, levels=[0], colors="tab:orange")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(True)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Графики кривых $f_1(x_1, x_2) = 0$ и $f_2(x_1, x_2) = 0$")
    plt.legend(["f1 = 0", "f2 = 0"])
    plt.savefig("plot_system.png", dpi=150)
    plt.show()


def plot_errors(si_xs: list, nw_xs: list, x_ref: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    for xs, name in [(si_xs, "простая итерация"), (nw_xs, "Ньютон")]:
        iters = np.arange(len(xs))
        err = [np.linalg.norm(x - x_ref, ord=np.inf) for x in xs]
        plt.semilogy(iters, err, "o-", label=name)
    plt.xlabel("Номер итерации k")
    plt.ylabel(r"$\|x^{(k)} - x^*\|_\infty$")
    plt.title("Зависимость погрешности от количества итераций")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("plot_errors_system.png", dpi=150)
    plt.show()


def main():
    a = float(input("Параметр a: "))

    plot_system(a)

    eps = float(input("Точность eps: "))
    x0 = np.array(
        [float(v) for v in input("Начальное приближение x1 x2 (графически): ").split()]
    )

    f = make_f(a)
    jacobian = make_jacobian(a)
    phi = make_phi(a)

    x_si, si_xs = simple_iteration(phi, x0, eps)
    x_nw, nw_xs = newton(f, jacobian, x0, eps)

    x_ref = fsolve(f, x0, xtol=1e-14)

    output = f"""
1)    Параметр a = {a}, начальное приближение (графически): x0 = {x0}

2)    Метод простой итерации:
      решение x = {x_si}
      итераций: {len(si_xs) - 1}
      невязка ||f(x)|| = {np.linalg.norm(f(x_si)):.3e}

3)    Метод Ньютона:
      решение x = {x_nw}
      итераций: {len(nw_xs) - 1}
      невязка ||f(x)|| = {np.linalg.norm(f(x_nw)):.3e}

4)    Эталон (scipy.optimize.fsolve): x* = {x_ref}

5)    Совпадение с эталоном:
      простая итерация: {np.linalg.norm(x_si - x_ref, ord=np.inf) < eps}
      Ньютон:           {np.linalg.norm(x_nw - x_ref, ord=np.inf) < eps}

6)    Погрешность ||x - x*||:
      простая итерация: {np.linalg.norm(x_si - x_ref, ord=np.inf):.3e}
      Ньютон:           {np.linalg.norm(x_nw - x_ref, ord=np.inf):.3e}
"""

    print(output)

    plot_errors(si_xs, nw_xs, x_ref)  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    main()
