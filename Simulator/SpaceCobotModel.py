import numpy as np
import scipy.spatial.transform as trf

from AbstractModel import Model

class Model3D(Model):
    def __init__(self, mass=1, inertia=np.eye(3), A=np.ones((6, 6))):
        """Initialize the 3D motion model.

        Parameters:
            mass (float): Mass of the rigid body.
            inertia (np.array): 3x3 inertia matrix.
        """
        super().__init__()
        self.m = mass
        self.J = inertia
        self.Jinv = np.linalg.inv(self.J)
        self.A = A

    @staticmethod
    def q_matrix(q):
        """Compute the quaternion kinematic matrix.

        Parameters:
            q (np.array): Quaternion [x, y, z, w] (SciPy convention).

        Returns:
            np.array: 4x3 matrix for quaternion kinematics.
        """
        x, y, z, w = q
        return np.array(
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
        """Compute the time derivative of the state given the current state and
        input.

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
        F = self.A[0:3, :] @ u
        M = self.A[3:6, :] @ u

        print(F)

        # Convert quaternion to rotation matrix
        R = trf.Rotation.from_quat(q).as_matrix()

        # Compute derivatives
        p_dot = v
        v_dot = (1 / self.m) * R.T @ F
        q_dot = 1 / 2 * self.q_matrix(q) @ w
        w_dot = self.Jinv @ (M - np.cross(w, (self.J @ w)))

        # Flatten and return
        return np.concatenate([p_dot, v_dot, q_dot, w_dot])