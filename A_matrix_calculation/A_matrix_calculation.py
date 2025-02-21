import numpy as np
from numpy import cos, sin


def A_matrix_column(theta_i: float, phi_i: float, w_i: float, d: float) -> np.ndarray:
    ratio_k2_k1 = 0.01

    print(f"{round(cos(theta_i) * d, 4)} {round(sin(theta_i) * d, 4)}")
    return np.array(
        [
            sin(theta_i) * sin(phi_i),
            -cos(theta_i) * sin(phi_i),
            cos(phi_i),
            (d * cos(phi_i) - w_i * ratio_k2_k1 * sin(phi_i)) * sin(theta_i),
            -(d * cos(phi_i) - w_i * ratio_k2_k1 * sin(phi_i)) * cos(theta_i),
            -d * sin(phi_i) - w_i * ratio_k2_k1 * cos(phi_i),
        ]
    )


def A_matrix(
    theta: np.ndarray, phi: np.ndarray, w: np.ndarray, d: np.ndarray
) -> np.ndarray:
    A = np.zeros((6, theta.size))
    for i in range(theta.size):
        A[:, i] = A_matrix_column(theta[i], phi[i], w[i], d[i])
    return A


def main():
    theta = np.array([0, 60, 120, 180, 240, 300])
    phi = np.array([55, -55, 55, -55, 55, -55])
    w = np.array([-1, 1, -1, 1, -1, 1])
    d = np.array([0.16] * 6)



    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    A = A_matrix(theta, phi, w, d)
    A = np.round(A, 3)
    np.save("A.npy", A)

    print(A)


if __name__ == "__main__":
    main()
