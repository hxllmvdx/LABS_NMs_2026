import numpy as np


def solve_tridiagonal(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    P[n - 1] = 0.0
    Q[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / (b[n - 1] + a[n - 1] * P[n - 2])

    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return P, Q, x


def main():
    M = np.loadtxt("input.txt")
    A, d = M[:, :-1], M[:, -1]
    n = A.shape[0]

    a = np.diag(A, -1)
    b = np.diag(A)
    c = np.diag(A, 1)
    a = np.concatenate(([0.0], a))
    c = np.concatenate((c, [0.0]))

    if np.any(np.abs(b) < np.abs(a) + np.abs(c)):
        i = np.argmax(np.abs(a) + np.abs(c) - np.abs(b))
        raise ValueError(f"Условие нарушено в строке {i + 1}")
    if not np.any(np.abs(b) > np.abs(a) + np.abs(c)):
        print("Предупреждение: строгое неравенство в не выполнено ни для одного i.\n")

    P, Q, x = solve_tridiagonal(a, b, c, d)

    forward_lines = [f"     P_1 = -c_1/b_1 = {P[0]:.6g},   Q_1 = d_1/b_1 = {Q[0]:.6g}"]
    for i in range(1, n - 1):
        forward_lines.append(f"     P_{i + 1} = {P[i]:.6g},   Q_{i + 1} = {Q[i]:.6g}")
    forward_lines.append(f"     P_{n} = 0,   Q_{n} = {Q[n - 1]:.6g}")

    backward_lines = [f"     x_{n} = Q_{n} = {x[n - 1]:.6g}"]
    for i in range(n - 2, -1, -1):
        backward_lines.append(
            f"     x_{i + 1} = P_{i + 1}*x_{i + 2} + Q_{i + 1} = {x[i]:.6g}"
        )

    forward_text = "\n".join(forward_lines)
    backward_text = "\n".join(backward_lines)
    residual = np.max(np.abs(A @ x - d))

    output = f"""Порядок системы: n = {n}

1) Прямой ход — прогоночные коэффициенты P_i, Q_i:
{forward_text}

2) Обратный ход:
{backward_text}

3) Ответ:
     x = {np.array2string(x, precision=6)}

4) Проверка: max|A*x - d| = {residual:.3e}
   Невязка меньше 1e-9: {residual < 1e-9}
"""
    print(output)


if __name__ == "__main__":
    main()
