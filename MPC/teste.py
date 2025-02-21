import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Time parameters
N = 50  # Prediction horizon
dt = 0.1  # Time step

# Robot parameters
m = 1.0  # Mass
J = np.diag([0.01, 0.01, 0.02])  # Inertia matrix

# Desired attitude matrix (identity for simplicity)
R_des = np.eye(3)

# Dynamics matrices
A_matrix = np.eye(6)  # Placeholder for input to thrust/torque mapping

# CasADi optimization object
opti = ca.Opti()

# Decision variables
X = opti.variable(13, N+1)  # [p; v; q; omega], q is quaternion
U = opti.variable(6, N)  # Control inputs (thrust and torques)

# Initial state
X0 = opti.parameter(13)
opti.set_value(X0, np.zeros(13))

# Dynamics equations
def dynamics(x, u):
    p = x[:3]
    v = x[3:6]
    q = x[6:10]
    omega = x[10:13]

    qw, qx, qy, qz = q

    # Rotation matrix from quaternion
    R = ca.vertcat(
        ca.horzcat(1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw),
        ca.horzcat(2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw),
        ca.horzcat(2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2),
    )

    F = A_matrix[:3] @ u[:3]
    M = A_matrix[3:] @ u[3:]

    pdot = v
    vdot = (1 / m) * R @ F

    # Quaternion derivative
    q_vec = ca.vertcat(qx, qy, qz)
    omega_mat = ca.vertcat(
        ca.horzcat(0, -omega[2], omega[1]),
        ca.horzcat(omega[2], 0, -omega[0]),
        ca.horzcat(-omega[1], omega[0], 0),
    )
    qdot = ca.vertcat(-0.5 * ca.mtimes(q_vec.T, omega), 0.5 * (ca.mtimes(q_vec, omega_mat) + qw * omega))

    omegadot = ca.mtimes(ca.inv(J), (M - ca.cross(omega, J @ omega)))

    return ca.vertcat(pdot, vdot, qdot, omegadot)

# Objective
cost = 0
for k in range(N):
    # Position tracking cost (quadratic)
    cost += ca.sumsqr(X[:3, k])  # Tracking desired position assumed to be origin

    # Control effort cost (quadratic)
    cost += ca.sumsqr(U[:, k])

    # Attitude cost (1/2 * trace(I - R_des.T @ R_k))
    qw, qx, qy, qz = X[6:10, k]
    R_k = ca.vertcat(
        ca.horzcat(1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw),
        ca.horzcat(2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw),
        ca.horzcat(2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2),
    )
    attitude_cost = 0.5 * ca.trace(np.eye(3) - ca.mtimes(R_des.T, R_k))
    cost += attitude_cost

# Dynamics constraints
for k in range(N):
    opti.subject_to(X[:, k+1] == X[:, k] + dt * dynamics(X[:, k], U[:, k]))

# Initial condition constraint
opti.subject_to(X[:, 0] == X0)

# Solve the optimization
opti.minimize(cost)
opts = {"ipopt.print_level": 0, "print_time": 0}
opti.solver("ipopt", opts)
sol = opti.solve()

# Extract and plot results
X_sol = sol.value(X)
U_sol = sol.value(U)

time = np.arange(N+1) * dt
plt.figure(figsize=(10, 6))
plt.plot(time, X_sol[:3, :].T)
plt.title("Evolution of Position")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend(["x", "y", "z"])
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time[:-1], U_sol.T)
plt.title("Control Inputs")
plt.xlabel("Time [s]")
plt.ylabel("Input Values")
plt.legend(["u1", "u2", "u3", "u4", "u5", "u6"])
plt.grid()
plt.show()
