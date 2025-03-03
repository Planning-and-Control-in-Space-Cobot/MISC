import numpy as np
import scipy.spatial.transform as trf
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp


class Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def f(self, t, state, u):
        pass


class Model3D(Model):
    def __init__(self, mass=1, inertia=np.eye(3)):
        """
        Initialize the 3D motion model.

        Parameters:
            mass (float): Mass of the rigid body.
            inertia (np.array): 3x3 inertia matrix.
        """
        super().__init__()
        self.m = mass
        self.J = inertia
        self.Jinv = np.linalg.inv(self.J)

    @staticmethod
    def q_matrix(q):
        """
        Compute the quaternion kinematic matrix.

        Parameters:
            q (np.array): Quaternion [x, y, z, w] (SciPy convention).

        Returns:
            np.array: 4x3 matrix for quaternion kinematics.
        """
        x, y, z, w = q
        return 0.5 * np.array(
            [
                [w, -z, y],
                [z, w, -x],
                [-y, x, w],
                [
                    -x,
                    -y,
                    -z,
                ],
            ]
        )

    def f(self, t, state, u):
        """
        Compute the time derivative of the state given the current state and input.

        Parameters:
            t (float): Time (required for ODE solvers but unused).
            state (np.array): Flat 13D state vector: [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz].
            u (np.array): Control input [Fx, Fy, Fz, Mx, My, Mz].

        Returns:
            np.array: Flat 13D derivative vector.
        """
        # Unpack state
        p = state[0:3]  # Position
        v = state[3:6]  # Velocity
        q = state[6:10]  # Quaternion (x, y, z, w)
        w = state[10:13]  # Angular velocity

        # Unpack control
        F = u[0:3]  # Force
        M = u[3:6]  # Torque

        # Convert quaternion to rotation matrix
        R = trf.Rotation.from_quat(q).as_matrix()

        # Compute derivatives
        p_dot = v
        v_dot = (1 / self.m) * R @ F
        q_dot = self.q_matrix(q) @ w
        w_dot = self.Jinv @ (M - np.cross(w, self.J @ w))

        # Flatten and return
        return np.concatenate([p_dot, v_dot, q_dot, w_dot])


class Simulator:
    """
    A simulator that integrates the Model3D dynamics over time.
    """

    def __init__(self, model, initial_state, dt=0.001):
        """
        Initialize the simulator.

        Parameters:
            model (Model3D): The dynamics model.
            initial_state (np.array): Initial 13D state vector.
            dt (float): Simulation time step.
        """
        self.model = model
        self.state = np.array(initial_state, dtype=float)
        self.dt = dt

    def step(self, u):
        """
        Simulate one time step forward.

        Parameters:
            u (np.array): Control input [Fx, Fy, Fz, Mx, My, Mz].

        Returns:
            np.array: Updated state after one step.
        """
        t_span = (0, self.dt)  # Integrate over one time step
        sol = solve_ivp(
            fun=lambda t, s: self.model.f(t, s, u),
            t_span=t_span,
            y0=self.state,
            method="RK45",
            t_eval=[self.dt],
        )

        # Store new state
        self.state = sol.y[:, -1]

        # Normalize quaternion
        self.state[6:10] /= np.linalg.norm(self.state[6:10])

        return self.state


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Define mass and inertia
    mass = 10.0
    inertia = np.diag([5.0, 5.0, 5.0])  # Simplified diagonal inertia

    # Create the model
    model = Model3D(mass, inertia)

    # Define initial state: [position, velocity, quaternion, angular velocity]
    initial_state = np.array(
        [
            0,
            0,
            0,  # Position
            0,
            0,
            0,  # Velocity
            0,
            0,
            0,
            1,  # Quaternion (Identity)
            0,
            0,
            0,
        ]
    )  # Angular velocity

    # Create simulator
    simulator = Simulator(model, initial_state, dt=0.001)

    # Define force (N) and torque (Nm) input
    F_body = np.array([0, 0, 10])  # Thrust in +Z direction
    M_body = np.array([0, 0.1, 0])  # Small torque around Y-axis
    u = np.concatenate([F_body, M_body])

    # Simulate for 20 ms (20 steps)
    state = initial_state
    for _ in range(20):
        state = simulator.step(u)

    print("Final state:", state)
