import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Simulator:
    def __init__(self, A, J, mass):
        self.A = A
        self.J = J
        self.mass = mass

    def unflat(self, X):
        v = X[:3]
        p = X[3:6]
        omega = X[6:9]
        q = X[9:13]
        return v, p, omega, q

    def flat(self, v, p, omega, q):
        return np.concatenate([v, p, omega, q])

    def R(self, q):
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ])
    def w_dot(self, M, J, omega):
        return np.linalg.inv(J) @ (M - np.cross(omega, J @ omega))

    def Q(self, q):
        qw, qx, qy, qz = q
        return np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw],
        ])

    def q_dot(self, q, omega):
        return 0.5 * self.Q(q) @ omega

    def v_dot(self, F, q):
        return (1 / self.mass) * self.R(q) @ F

    def dynamics(self, t, X, u):
        F = self.A[:3, :] @ u
        M = self.A[3:, :] @ u
        v, p, omega, q = self.unflat(X)

        dv = self.v_dot(F, q)
        dp = v
        domega = self.w_dot(M, self.J, omega)
        dq = self.q_dot(q, omega)

        return self.flat(dv, dp, domega, dq)

    def simulate(self, X0, U, dt):
        time_span = (0, dt * len(U))
        t_eval = np.arange(0, dt * len(U), dt)
        state_trajectory = [X0]

        for i, u in enumerate(U.T):  # Iterate through each control input
            sol = solve_ivp(
                lambda t, X: self.dynamics(t, X, u),
                (i * dt, (i + 1) * dt),
                state_trajectory[-1],
                t_eval=[(i + 1) * dt],
                method='RK45'
            )
            state_trajectory.append(sol.y.flatten())

        return np.array(state_trajectory)


def plot_trajectory(state_trajectory):
    # Extract variables
    time = np.linspace(0, state_trajectory.shape[0] * 0.1, state_trajectory.shape[0])  # Assuming dt=0.1
    velocity = state_trajectory[:, :3]
    position = state_trajectory[:, 3:6]
    quaternion = state_trajectory[:, 9:13]
    angular_velocity = state_trajectory[:, 6:9]

    # Plot velocity components
    plt.figure(figsize=(10, 6))
    plt.plot(time, velocity[:, 0], label="v_x")
    plt.plot(time, velocity[:, 1], label="v_y")
    plt.plot(time, velocity[:, 2], label="v_z")
    plt.title("Velocity Components Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    # Plot position components
    plt.figure(figsize=(10, 6))
    plt.plot(time, position[:, 0], label="p_x")
    plt.plot(time, position[:, 1], label="p_y")
    plt.plot(time, position[:, 2], label="p_z")
    plt.title("Position Components Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()

    # Plot quaternion components
    plt.figure(figsize=(10, 6))
    plt.plot(time, quaternion[:, 0], label="q_w")
    plt.plot(time, quaternion[:, 1], label="q_x")
    plt.plot(time, quaternion[:, 2], label="q_y")
    plt.plot(time, quaternion[:, 3], label="q_z")
    plt.title("Quaternion Components Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Quaternion")
    plt.legend()
    plt.grid()

    # Plot angular velocity components
    plt.figure(figsize=(10, 6))
    plt.plot(time, angular_velocity[:, 0], label="omega_x")
    plt.plot(time, angular_velocity[:, 1], label="omega_y")
    plt.plot(time, angular_velocity[:, 2], label="omega_z")
    plt.title("Angular Velocity Components Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    J = np.diag([0.03, 0.04, 0.06])
    mass = 3.0
    A = np.array([
       [-0.   , -0.709,  0.709, -0.   , -0.709,  0.709],
       [-0.819,  0.41 ,  0.41 , -0.819,  0.41 ,  0.41 ],
       [ 0.574,  0.574,  0.574,  0.574,  0.574,  0.574],
       [-0.   ,  0.087,  0.087, -0.   , -0.087, -0.087],
       [-0.1  , -0.05 ,  0.05 ,  0.1  ,  0.05 , -0.05 ],
       [-0.125,  0.125, -0.125,  0.125, -0.125,  0.125]
    ])

    simulator = Simulator(A, J, mass)

    # Initial state
    X0 = np.zeros(13)

    ## Set initial pos
    X0[3:6] = np.array([0, 0, 0.5])  # Set initial position to [0, 0, 0.5]

    ## set initial q 
    X0[9:13] = np.array([.707, 0, 0, 0.707])  # Set initial quaternion to [1, 0, 0, 0]

    # Simulated control inputs (e.g., sinusoidal inputs)
    T = 10  # Total simulation time
    dt = 0.1
    N = int(T / dt)
    U = np.zeros((6, N))
    U[0, :] = 1  # Set motor 0 to apply 1N of thrust at all times



    state_trajectory = simulator.simulate(X0, U, dt)
    print("Simulation complete. Plotting results...")

    # Plot the state trajectory
    plot_trajectory(state_trajectory)


if __name__ == "__main__":
    main()
