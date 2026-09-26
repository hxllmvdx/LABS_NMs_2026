import numpy as np


def read_input(path: str) -> tuple[float, np.ndarray]:
    with open(path) as f:
        eps = float(f.readline().strip())
        A = np.loadtxt(f, dtype=float)
    A = np.atleast_2d(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица в файле должна быть квадратной")
    return eps, A


def sign(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    R = A.astype(float).copy()
    Q = np.eye(n)

    for k in range(n - 1):
        b = R[k:, k].copy()

        norm_b = np.sqrt(np.sum(b**2))
        e1 = np.zeros_like(b)
        e1[0] = 1.0
        v = b + sign(b[0]) * norm_b * e1

        vTv = float(v @ v)
        if vTv == 0.0:
            continue

        H_k = np.eye(n - k) - 2.0 * np.outer(v, v) / vTv

        R[k:, k:] = H_k @ R[k:, k:]
        Q[:, k:] = Q[:, k:] @ H_k.T

    return Q, R


def eigenvalues_of_2x2(block: np.ndarray) -> tuple[complex, complex]:
    a, b = block[0, 0], block[0, 1]
    c, d = block[1, 0], block[1, 1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    root = np.sqrt(complex(disc))
    l1 = (tr + root) / 2.0
    l2 = (tr - root) / 2.0
    return l1, l2


def qr_eigenvalues(
    A: np.ndarray, eps: float = 1e-6, max_iter: int = 5000
) -> list[complex]:
    Ak = A.astype(float).copy()
    n = A.shape[0]
    eig: list[complex] = []
    k = 0
    prev_block: tuple[complex, complex] | None = None
    iterations = 0

    while n - k > 2 and iterations < max_iter:
        iterations += 1

        subdiag_sq = np.sum(Ak[k + 1 : n, k] ** 2)
        if np.sqrt(subdiag_sq) <= eps:
            eig.append(complex(Ak[k, k]))
            k += 1
            prev_block = None
            continue

        if abs(Ak[n - 1, n - 2]) <= eps:
            eig.append(complex(Ak[n - 1, n - 1]))
            n -= 1
            prev_block = None
            continue

        block = Ak[n - 2 : n, n - 2 : n]
        l1, l2 = eigenvalues_of_2x2(block)

        if prev_block is not None:
            dl1 = min(abs(l1 - prev_block[0]), abs(l1 - prev_block[1]))
            dl2 = min(abs(l2 - prev_block[0]), abs(l2 - prev_block[1]))
            if max(dl1, dl2) <= eps:
                eig.append(l1)
                eig.append(l2)
                n -= 2
                prev_block = None
                continue

        prev_block = (l1, l2)

        Q, R = qr_decomposition(Ak[k:n, k:n])
        Ak[k:n, k:n] = R @ Q

    if n - k == 2:
        l1, l2 = eigenvalues_of_2x2(Ak[k:n, k:n])
        eig.extend([l1, l2])
    elif n - k == 1:
        eig.append(complex(Ak[k, k]))

    real_part = sorted([z for z in eig if abs(z.imag) < 1e-8], key=lambda z: -z.real)
    complex_part = sorted(
        [z for z in eig if abs(z.imag) >= 1e-8], key=lambda z: -z.real
    )
    return [z.real if abs(z.imag) < 1e-8 else z for z in real_part + complex_part]


def format_number(x: float) -> str:
    return f"{x:10.6f}"


def print_matrix(M: np.ndarray) -> str:
    return "\n".join("  ".join(format_number(v) for v in row) for row in M)


def format_eigenvalue(z: complex) -> str:
    if abs(z.imag) < 1e-8:
        return f"{z.real:10.6f}"
    return f"{z.real:10.6f} {z.imag:+10.6f}i"


def main():
    eps, A = read_input("input_complex.txt")

    Q, R = qr_decomposition(A)
    QR = Q @ R

    lam = qr_eigenvalues(A, eps)
    lam_lines = "\n".join(
        f"   lambda_{i} = {format_eigenvalue(complex(z))}" for i, z in enumerate(lam, 1)
    )

    lam_np = np.linalg.eigvals(A)
    lam_np_sorted = sorted(lam_np, key=lambda z: -z.real)
    lam_np_lines = "\n".join(
        f"   lambda_{i} = {format_eigenvalue(z)}"
        for i, z in enumerate(lam_np_sorted, 1)
    )
    match = all(any(abs(a - b) < 1e-4 for b in lam_np) for a in lam)

    output = f"""
============================================================
1) Исходная матрица A:
{print_matrix(A)}

2) Матрица Q (ортогональная):
{print_matrix(Q)}

   Матрица R (верхняя треугольная):
{print_matrix(R)}

3) Произведение Q · R:
{print_matrix(QR)}

   Проверка Q·R == A: {np.allclose(QR, A)}
   Проверка Q^T·Q == E: {np.allclose(Q.T @ Q, np.eye(A.shape[0]))}

4) Собственные значения (QR-алгоритм):
{lam_lines}

   Проверка (numpy.linalg.eigvals):
{lam_np_lines}

   Совпадение с эталоном: {match}
"""
    print(output)


if __name__ == "__main__":
    main()
