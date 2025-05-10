import casadi as ca
import numpy as np
import spatial_casadi as sc


class AABBObstacle:
    """Axis-Aligned Bounding Box (AABB) Obstacle"""

    def __init__(self, x_min, y_min, z_min, x_max, y_max, z_max, safety_margin=0.1):
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max
        self.safety_margin = safety_margin  # Minimum clearance distance

    def closest_point(self, p):
        """Compute the closest point on the AABB to a given point p"""
        px = ca.fmax(self.x_min, ca.fmin(p[0], self.x_max))
        py = ca.fmax(self.y_min, ca.fmin(p[1], self.y_max))
        pz = ca.fmax(self.z_min, ca.fmin(p[2], self.z_max))
        return ca.vertcat(px, py, pz)


class EllipsoidOptimizer:
    def __init__(self, m, J, A):
        if not self.check_mechanical_parameters(m, J, A):
            raise ValueError("Invalid mechanical parameters")
        self.m, self.J, self.A = m, J, A

    @staticmethod
    def check_mechanical_parameters(m, J, A):
        return (
            A.shape == (6, 6)
            and J.shape == (3, 3)
            and np.all(J >= 0)
            and m.shape == (1,)
            and m > 0
        )

    @staticmethod
    def quaternion_multiplication(q1, q2):
        """Quaternion multiplication"""
        q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
        q_ = ca.vertcat(
            ca.horzcat(q1w, q1z, -q1y, q1x),
            ca.horzcat(-q1z, q1w, q1x, q1y),
            ca.horzcat(q1y, -q1x, q1w, q1z),
            ca.horzcat(-q1x, -q1y, -q1z, q1w),
        )
        return ca.mtimes(q_, q2)

    @staticmethod
    def quaternion_integration(q, w, dt):
        """Quaternion integration using Rodrigues formula"""
        w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)
        q_ = ca.vertcat(w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2))
        return EllipsoidOptimizer.quaternion_multiplication(q_, q)

    def setup_problem(self, N=60, dt=0.1, aabb_obstacles_list=None):
        self.opti = ca.Opti()
        self.N, self.dt = N, dt

        # State Variables
        self.p = self.opti.variable(3, N)  # Position
        self.v = self.opti.variable(3, N)  # Velocity
        self.q = self.opti.variable(4, N)  # Quaternion (xyzw)
        self.w = self.opti.variable(3, N)  # Angular velocity
        self.u = self.opti.variable(6, N)  # Control actuation

        # Parameters
        J_ = self.opti.parameter(3, 3)
        A_ = self.opti.parameter(6, 6)
        m = self.opti.parameter(1)

        self.opti.set_value(J_, self.J)
        self.opti.set_value(A_, self.A)
        self.opti.set_value(m, self.m)

        # Initial & Final State Constraints
        self.p0 = self.opti.parameter(3)
        self.v0 = self.opti.parameter(3)
        self.q0 = self.opti.parameter(4)
        self.w0 = self.opti.parameter(3)
        self.p_n = self.opti.parameter(3)
        self.q_n = self.opti.parameter(4)

        # Apply initial constraints
        self.opti.subject_to(self.p[:, 0] == self.p0)
        self.opti.subject_to(self.v[:, 0] == self.v0)
        self.opti.subject_to(self.q[:, 0] == self.q0)
        self.opti.subject_to(self.w[:, 0] == self.w0)

        # Robot dimensions (ellipsoid approximation)
        a_robot, b_robot, c_robot = 0.24, 0.24, 0.10

        # Dynamics Constraints
        for i in range(N - 1):
            R = sc.Rotation.from_quat(self.q[:, i], "xyzw")
            F = ca.mtimes(A_[:3, :], self.u[:, i])
            M = ca.mtimes(A_[3:, :], self.u[:, i])

            self.opti.subject_to(self.p[:, i + 1] == self.p[:, i] + self.v[:, i] * dt)
            self.opti.subject_to(
                self.v[:, i + 1] == self.v[:, i] + (R.as_matrix() @ F) / m * dt
            )
            self.opti.subject_to(
                self.q[:, i + 1]
                == self.quaternion_integration(self.q[:, i], self.w[:, i], dt)
            )
            self.opti.subject_to(
                self.w[:, i + 1]
                == self.w[:, i]
                + ca.inv(J_) @ (M - ca.cross(self.w[:, i], J_ @ self.w[:, i])) * dt
            )

        # Quaternion normalization
        for i in range(N):
            self.opti.set_initial(self.q[:, i], np.array([0, 0, 0, 1]))

        # Control Constraints
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(-2, self.u[:, i], 2))

        # AABB Obstacle Avoidance
        if aabb_obstacles_list:
            for i in range(N):
                for obstacle in aabb_obstacles_list:
                    closest_point = obstacle.closest_point(self.p[:, i])
                    min_distance = ca.norm_2(self.p[:, i] - closest_point)
                    self.opti.subject_to(min_distance > obstacle.safety_margin)

        # Cost Function
        cost = 0
        for i in range(N):
            R = sc.Rotation.from_quat(self.q[:, i], "xyzw")
            R_des = sc.Rotation.from_quat(self.q_n, "xyzw")

            cost += self.u[:, i].T @ self.u[:, i]  # Minimize actuation
            cost += (
                (self.p[:, i] - self.p_n).T @ 100 @ (self.p[:, i] - self.p_n)
            )  # Minimize position error
            cost += (
                100 * 0.5 * ca.trace(ca.MX.eye(3) - R.as_matrix() @ R_des.as_matrix().T)
            )  # Minimize orientation error

        self.opti.minimize(cost)
        self.opti.solver("ipopt")

    def solve(self, p0, v0, q0, w0, p_n, q_n):
        self.opti.set_value(self.p0, p0)
        self.opti.set_value(self.v0, v0)
        self.opti.set_value(self.q0, q0)
        self.opti.set_value(self.w0, w0)
        self.opti.set_value(self.p_n, p_n)
        self.opti.set_value(self.q_n, q_n)

        sol = self.opti.solve()
        return sol
