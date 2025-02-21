import casadi as ca
import numpy as np


class MPC:
    def __init__(self, A, J, mass):
        self.A = A
        self.J = J
        self.mass = mass

    def setup_problem(self, N, dt):
        # Define the optimization problem
        opti = ca.Opti()

        # State vector
        X = opti.variable(13, N)  # State vector
        X0 = opti.parameter(13)  # Initial state
        Xref = opti.parameter(13, N)  # Reference state

        u = opti.variable(6, N)  # Control input (e.g., motor commands)

        A = opti.parameter(6, 6)
        opti.set_value(A, self.A)

        J = opti.parameter(3, 3)
        opti.set_value(J, self.J)

        m = opti.parameter()
        opti.set_value(m, self.mass)

        def unflat(X):
            v = X[:3]
            p = X[3:6]
            omega = X[6:9]
            q = X[9:13]
            return v, p, omega, q

        def w_dot(M, J, omega):
            return ca.inv(J) @ (M - ca.cross(omega, J @ omega))

        def q_dot(q, omega):
            return (1 / 2) * self.Q(q) @ omega

        def v_dot(F, q):
            return (1 / m) * self.R(q) @ F

        def X_dot(F, M, X, J):
            v, p, omega, q = unflat(X)
            return ca.vertcat(v_dot(F, q), v, w_dot(M, J, omega), q_dot(q, omega))

        # Dynamics equations
        for i in range(N - 1):
            F = A[:3, :] @ u[:, i]
            M = A[3:, :] @ u[:, i]

            x_next = X[:, i] + dt * X_dot(F, M, X[:, i], J)
            x_next[9:13] = x_next[9:13] / ca.norm_2(x_next[9:13])  # Normalize quaternion
            opti.subject_to(X[:, i + 1] == x_next)


        # Cost function
        cost = 0
        for i in range(N):
            v_ref, p_ref, omega_ref, q_ref = unflat(Xref[:, i])
            v, p, omega, q = unflat(X[:, i])
            cost += (
                self.attitude_error(q, q_ref)
                + self.position_error(p, p_ref)
                +0.0001 * self.actuation_error(u[:, i])
            )
        opti.minimize(cost)

        # Constraints
        opti.subject_to(X[:, 0] == X0)  # Initial condition
        opti.subject_to(opti.bounded(-2, u[:, :], 2))  # Control bounds

        #for i in range(N):
        #    _, _, _, q = unflat(X[:, i])
        #    opti.subject_to(opti.bounded(0.95, ca.norm_2(q), 1.05))  # Quaternion normalization

        for i in range(N): 
            opti.set_initial(
                X[9:13, i], [1, 0, 0, 0]
            )

        opts = {
            "ipopt": {
                "print_level": 0,
                "acceptable_tol": 1e-2,
                "acceptable_obj_change_tol": 1e-2,
                "linear_solver": "mumps",
                "hessian_approximation": "limited-memory",
            },
            "jit": True,
            "jit_cleanup": True,
        }

        opti.solver("ipopt", opts)

        self.opti = opti
        self.X = X
        self.u = u
        self.X0 = X0
        self.Xref = Xref

    def solve_problem(self, X0, Xref):
        self.opti.set_value(self.X0, X0)
        self.opti.set_value(self.Xref, Xref)

        for i in range(10):
            print(f"X0   - {X0}")
            print(f"Xref - {Xref[:, i]}")

        try:
            sol = self.opti.solve()
            return sol.value(self.u), sol.value(self.X)
        except Exception as e:
            print(f"Solver encountered an error: {e}")
            return None, None

    def attitude_error(self, q1, q2):
        return 1 - ca.dot(q1, q2)**2

    def position_error(self, p1, p2):
        return (p1 - p2).T @ (p1 - p2)

    def actuation_error(self, u):
        return u.T @ u

    @staticmethod
    def R(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return ca.vertcat(
            ca.horzcat(
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ), 
            ca.horzcat(
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ), 
            ca.horzcat(
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            )
        )

    @staticmethod
    def Q(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return ca.vertcat(
            ca.horzcat(qw, -qz, qy),
            ca.horzcat(qz, qw, -qx),
            ca.horzcat(-qy, qx, qw),
            ca.horzcat(-qx, -qy, -qz),
        )


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

    mpc = MPC(A, J, mass)

    N = 10  # Horizon length
    dt = 0.1  # Time step
    mpc.setup_problem(N, dt)

    # Initial state: Zero velocity, zero position, identity quaternion
    X0 = np.zeros(13)
    X0[9] = 1.0  # Identity quaternion (qw = 1, qx = qy = qz = 0)
    X0[3:6] = 0.0  # Zero position

    # Reference state: Move to position (1, 1, 1)
    Xref = np.zeros((13, N))
    Xref[3:6, :] = 1.0  # Reference position (x, y, z)
    Xref[9, :] = 1.0  # Maintain identity quaternion

    u, X = mpc.solve_problem(X0, Xref)
    if u is not None and X is not None:
        np.save("control_inputs.npy", u)
        np.save("state_trajectory.npy", X)
        print("Optimal control inputs:", u)
        print("Optimal state trajectory:", X)
    else:
        print("Failed to solve the optimization problem.")
        sol = mpc.opti.debug.value(mpc.u)  # After solving, check the intermediate u values
        print (sol)
        sol = mpc.opti.debug.value(mpc.X)  # Check the state trajectory
        print (sol)



if __name__ == "__main__":
    main()