import numpy as np


def get_f_matrix(
    c1: float,
    c2: float,
    x1: float,
    x2: float,
) -> np.ndarray[float]:
    return np.array([
        c1 + c2 - 2,
        c1 * x1 + c2 * x2,
        c1 * x1**2 + c2 * x2**2 - 2 / 3,
        c1 * x1**3 + c2 * x2**3,
    ])


def get_jacobian_matrix(
    c1: float,
    c2: float,
    x1: float,
    x2: float,
) -> np.ndarray[np.ndarray[float]]:
    return np.array([
        [1, 1, 0, 0],
        [x1, x2, c1, c2],
        [x1**2, x2**2, 2 * c1 * x1, 2 * c2 * x2],
        [x1**3, x2**3, 3 * c1 * x1**2, 3 * c2 * x2**2],
    ])


def get_nr_step(
    c1: float,
    c2: float,
    x1: float,
    x2: float,
) -> np.ndarray[float]:
    f = get_f_matrix(c1, c2, x1, x2)
    j = get_jacobian_matrix(c1, c2, x1, x2)
    j_inv = np.linalg.inv(j)

    return np.matmul(j_inv, f)


if __name__ == "__main__":
    c1 = 1
    c2 = -1
    x1 = 1
    x2 = -1

    counter = 0
    while True:
        step = get_nr_step(c1, c2, x1, x2)
        c1 -= step[0]
        c2 -= step[1]
        x1 -= step[2]
        x2 -= step[3]
        print(f"Counter: {counter}:")
        print(f"Step: {step}")
        print(f"c1={c1}, c2={c2}, x1={x1}, x2={x2}")
        print("=====")

        counter += 1
        input()
