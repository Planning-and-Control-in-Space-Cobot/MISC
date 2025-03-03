import casadi as ca
import spatial_casadi as sc
import numpy as np


class LongHorizonMPC:
    def __init__(self, sdf, map):
        self.sdf = sdf
        self.map = map

    def check_mechanical_parameters(self, m, J, A):
        if A.shape != (6, 6):
            return False

        if J.shape != (3, 3):
            return False

        if np.all(J != np.diag(np.diagonal(J))):
            return False

        if np.all(J <= 0):
            return False

        if m.shape != (1,):
            return False

        if m <= 0:
            return False

        return True

    @staticmethod
    def quaternion_multiplication(q1, q2):
        """
        q1 = [x y z w]
        q2 = [x y z w]

        We are going to mulitply q1 @ q2 according to the formula present in
        'Indirect Kalman filter for 3D attitude Estimation'
        """

        q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
        q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]

        q_ = ca.vertcat(
            ca.horzcat(q1w, q1z, -q1y, q1x),
            ca.horzcat(-q1z, q1w, q1x, q1y),
            ca.horzcat(q1y, -q1x, q1w, q1z),
            ca.horzcat(-q1x, -q1y, -q1z, q1w),
        )

        return ca.vertcat(q2x, q2y, q2z, q2w)

    def quaternion_integration(self, q, w, dt):
        """
        q = [x y z w]
        w = [wx wy wz]
        dt = scalar

        We are going to use a formula (123) from 'Indirect Kalman filter for 3D attitude Estimation'
        """

        w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)

        q_ = ca.vertcat(
            w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2)
        )

        return self.quaternion_multiplication(q_, q)

    def setup_problem(
        self,
        N,
        dt,
        A_,
        J_,
        m_,
        sdf_interpoland,
        actuation_cost=10,
        goal_cost=50,
        sdf_cost=20,
    ):
        if not self.check_mechanical_parameters(m_, J_, A_):
            raise ValueError("Invalid mechanical parameters")

        if N <= 0:
            raise ValueError("Invalid horizon length")

        if dt <= 0:
            raise ValueError("Invalid time step")
        self.N = N
        self.dt = dt

        self.opti = ca.Opti()

        self.sdf = sdf_interpoland

        self.p = self.opti.variable(3, N)  # x, y, z
        self.v = self.opti.variable(3, N)  # vx, vy, vz
        self.q = self.opti.variable(4, N)  # x, y, z, w
        self.w = self.opti.variable(3, N)  # wx, wy, wz

        self.u = self.opti.variable(6, N)  # u1, u2, u3, u4, u5, u6

        self.J = self.opti.parameter(3, 3)  # Inertia matrix
        self.A = self.opti.parameter(6, 6)  # Mixer Matrix
        self.m = self.opti.parameter(1)  # Mass

        self.opti.set_value(self.J, J_)
        self.opti.set_value(self.A, A_)
        self.opti.set_value(self.m, m_)

        self.p0 = self.opti.parameter(3)
        self.v0 = self.opti.parameter(3)
        self.q0 = self.opti.parameter(4)
        self.w0 = self.opti.parameter(3)

        self.p_n = self.opti.parameter(3)
        self.v_n = self.opti.parameter(3)
        self.q_n = self.opti.parameter(4)
        self.w_n = self.opti.parameter(3)

        # Initial State Constraints
        self.opti.subject_to(self.p[:, 0] == self.p0)
        self.opti.subject_to(self.v[:, 0] == self.v0)
        self.opti.subject_to(self.q[:, 0] == self.q0)
        self.opti.subject_to(self.w[:, 0] == self.w0)

        # Dynamics Constraints
        for i in range(N - 1):
            R = sc.Rotation.from_quat(self.q[:, i], "xyzw")

            F = ca.mtimes(self.A[:3, :], self.u[:, i])
            M = ca.mtimes(self.A[3:, :], self.u[:, i])

            self.opti.subject_to(
                self.p[:, i + 1] == self.p[:, i] + self.v[:, i] * self.dt
            )
            self.opti.subject_to(
                self.v[:, i + 1]
                == self.v[:, i] + (R.as_matrix() @ F) / self.m * self.dt
            )

            self.opti.subject_to(
                self.q[:, i + 1]
                == self.quaternion_integration(
                    self.q[:, i], self.w[:, i], self.dt
                )
            )
            self.opti.subject_to(
                self.w[:, i + 1]
                == self.w[:, i]
                + ca.inv(self.J)
                @ (M - ca.cross(self.w[:, i], self.J @ self.w[:, i]))
                * self.dt
            )

        for i in range(N):
            self.opti.subject_to(self.opti.bounded(-2, self.u[:, i], 2))
            self.opti.subject_to(self.sdf(self.p[:, i]) > 0)
            self.opti.subject_to(ca.norm_2(self.q[:, i]) == 1)

        cost = 0

        for i in range(N):
            R = sc.Rotation.from_quat(self.q[:, i], "xyzw")
            R_des = sc.Rotation.from_quat(self.q_n, "xyzw")

            cost += (
                (self.p[:, i] - self.p_n).T
                @ goal_cost
                @ (self.p[:, i] - self.p_n)
            )
            cost += (
                1
                / 2
                * ca.trace(ca.MX.eye(3) - R.as_matrix() @ R_des.as_matrix().T)
                * goal_cost
            )
            cost += self.u[:, i].T @ actuation_cost @ self.u[:, i]
            # cost += 1 / (self.sdf(self.p[:, i])) * sdf_cost

        self.opti.minimize(cost)

        p_opts = {
            "expand": False,
        }
        s_opts = {
            "max_cpu_time": 40,
            "max_iter": 10000,
        }

        self.opti.solver("ipopt", p_opts, s_opts)

    def solve_problem(
        self,
        start_pos,
        start_vel,
        start_quat,
        start_ang_vel,
        goal_pos,
        goal_vel,
        goal_quat,
        goal_ang_vel,
    ):
        self.opti.set_value(self.p0, start_pos)
        self.opti.set_value(self.v0, start_vel)
        self.opti.set_value(self.q0, start_quat)
        self.opti.set_value(self.w0, start_ang_vel)

        self.opti.set_value(self.p_n, goal_pos)
        self.opti.set_value(self.v_n, goal_vel)
        self.opti.set_value(self.q_n, goal_quat)
        self.opti.set_value(self.w_n, goal_ang_vel)

        for i in range(self.N):
            self.opti.set_initial(self.p[:, i], start_pos)
            self.opti.set_initial(self.v[:, i], start_vel)
            self.opti.set_initial(self.q[:, i], start_quat)
            self.opti.set_initial(self.w[:, i], start_ang_vel)

        try:
            self.sol = self.opti.solve()
            return self.sol
        except:
            return None
