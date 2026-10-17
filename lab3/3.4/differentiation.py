import numpy as np


def first_derivative_central2(y: np.ndarray, h: float, i: int) -> float:
    return (y[i + 1] - y[i - 1]) / (2.0 * h)


def first_derivative_central4(y: np.ndarray, h: float, i: int) -> float:
    return (y[i - 2] - 8.0 * y[i - 1] + 8.0 * y[i + 1] - y[i + 2]) / (12.0 * h)


def second_derivative_central2(y: np.ndarray, h: float, i: int) -> float:
    return (y[i - 1] - 2.0 * y[i] + y[i + 1]) / h**2


def second_derivative_central4(y: np.ndarray, h: float, i: int) -> float:
    return (-y[i - 2] + 16.0 * y[i - 1] - 30.0 * y[i] + 16.0 * y[i + 1] - y[i + 2]) / (
        12.0 * h**2
    )


def main():
    with open("input.txt", encoding="utf-8") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]

    x_star = float(lines[0])
    x = np.array([float(v) for v in lines[1].split()])
    y = np.array([float(v) for v in lines[2].split()])

    h = x[1] - x[0]
    i = round((x_star - x[0]) / h)

    used_nodes = x[i - 2 : i + 3]
    used_values = y[i - 2 : i + 3]

    d1_o2 = first_derivative_central2(y, h, i)
    d1_o4 = first_derivative_central4(y, h, i)
    d2_o2 = second_derivative_central2(y, h, i)
    d2_o4 = second_derivative_central4(y, h, i)

    output = f"""
Исходная таблица значений функции:
{"i":<4}{"x_i":<10}{"y_i":<12}
"""
    for idx, (xi, yi) in enumerate(zip(x, y)):
        output += f"{idx:<4}{xi:<10.4f}{yi:<12.6f}\n"

    output += f"""
Точка вычисления: x* = {x_star}
Шаг таблицы: h = {h}
Индекс точки в таблице: i = {i}

Узлы, использованные при вычислении производных:
x = {used_nodes}
y = {used_values}

Первая производная:
  формула 2-го порядка: f'(x*) ≈ {d1_o2:.6f}
  формула 4-го порядка: f'(x*) ≈ {d1_o4:.6f}

Вторая производная:
  формула 2-го порядка: f''(x*) ≈ {d2_o2:.6f}
  формула 4-го порядка: f''(x*) ≈ {d2_o4:.6f}
"""

    print(output)


if __name__ == "__main__":
    main()
