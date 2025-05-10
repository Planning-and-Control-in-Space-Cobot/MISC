# 2D Polytope - Ellipsoid Obstacle avoidance considering attitude on both polytope and ellipsoid
# This method is based on the paper "Efficient Collision Modelling for Numerical Optimal Control" - Section 3.B
# 3D Polytope - Ellipsoid Obstacle avoidance considering full SO(3) attitude
# 3D Ellipsoidal Robot - Polytope Obstacle avoidance using dual constraints (Proposition 3)
# 3D Ellipsoidal Robot with Rotation - Polytope Obstacle avoidance using dual constraints (Proposition 3)
# 3D Ellipsoidal Robot with Rotation - Polytope Obstacle avoidance using dual constraints (Proposition 3)

import casadi as ca
import spatial_casadi as sc
import scipy.spatial.transform as trf
import numpy as np
import pyvista as pv
from time import sleep


# Obstacle class: a polytope defined by its vertices (in its own frame),
# a position, a rotation (for orientation), and a safety distance.
class Obstacle:
    def __init__(
        self, vertices_local, position, rotation=np.eye(3), safety_distance=0.01
    ):
        self.vertices_local = vertices_local
        self.position = position
        self.rotation = rotation
        self.safety_distance = safety_distance

    def global_vertices(self):
        # Returns a 3 x num_vertices matrix in world coordinates.
        return self.rotation @ self.vertices_local + self.position[:, None]


# Map holds a list of obstacles.
class Map:
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)


class Robot:
    def __init__(self, mass, inertia, actuator_matrix, raddius):
        self.mass = mass
        self.inertia = inertia
        self.actuator_matrix = actuator_matrix
        self.raddius = raddius


# Quaternion multiplication and integration routines
def quaternion_multiplication(q1, q2):
    q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
    q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]
    q_ = ca.vertcat(
        ca.horzcat(q1w, q1z, -q1y, q1x),
        ca.horzcat(-q1z, q1w, q1x, q1y),
        ca.horzcat(q1y, -q1x, q1w, q1z),
        ca.horzcat(-q1x, -q1y, -q1z, q1w),
    )
    return q_ @ ca.vertcat(q2x, q2y, q2z, q2w)


def quaternion_integration(q, w, dt):
    w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)
    q_ = ca.vertcat(w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2))
    return quaternion_multiplication(q_, q)


# Setup
N = 40
# Define the ellipsoidal robot shape via its base matrix.
# Here, the ellipsoid is defined by: (x - pos)^T (R*P_R_base*R^T) (x - pos) <= 1.
P_R_base = np.diag([1 / 0.24**2, 1 / 0.24**2, 1 / 0.10**2])

opti = ca.Opti()

# State: [pos(3); vel(3); quat(4); ang_vel(3)] and control (6 motors)
x = opti.variable(13, N)
u = opti.variable(6, N)
dt = 0.1

x0 = opti.parameter(13)
xf = opti.parameter(13)
# Set initial and final state values
opti.set_value(x0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
opti.set_value(xf, np.array([8, 3, 0, 0, 0, 0, 0.707, 0, 0, 0.707, 0, 0, 0]))

# Parameters (mass, inertia, actuator mapping) loaded from files
J = opti.parameter(3, 3)
A = opti.parameter(6, 6)
m = opti.parameter(1)
import os

executable_dir = os.path.dirname(os.path.abspath(__file__))
J_ = np.load(os.path.join(executable_dir, "J_matrix.npy"))
A_ = np.load(os.path.join(executable_dir, "A_matrix.npy"))
m_ = np.load(os.path.join(executable_dir, "mass.npy"))


opti.set_value(J, J_)
opti.set_value(A, A_)
opti.set_value(m, m_)

# Define obstacles in the environment (polytopic obstacles)
map_env = Map()
vertices_obstacle = np.array(
    [
        [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
    ]
)
position_obstacle = np.array([2.0, 0, 0])
rotation_obstacle = np.eye(3)
safety_dist_obstacle = 0.05
obstacle1 = Obstacle(
    vertices_obstacle, position_obstacle, rotation_obstacle, safety_dist_obstacle
)
map_env.add_obstacle(obstacle1)

position_obstacle = np.array([0.0, 2.3, 0.0]) + position_obstacle
obstacle2 = Obstacle(
    vertices_obstacle, position_obstacle, rotation_obstacle, safety_dist_obstacle
)
map_env.add_obstacle(obstacle2)

position_obstacle = np.array([-2.0, 0.0, 0.0]) + position_obstacle
obstacle3 = Obstacle(
    vertices_obstacle, position_obstacle, rotation_obstacle, safety_dist_obstacle
)
map_env.add_obstacle(obstacle3)


# Set the initial state constraint
opti.subject_to(x[:, 0] == x0)

# Extract state components
pos, vel, quat, ang_vel = x[:3, :], x[3:6, :], x[6:10, :], x[10:, :]

# Dynamics constraints
for i in range(N - 1):
    F = ca.mtimes(A[:3, :], u[:, i])
    M = ca.mtimes(A[3:, :], u[:, i])
    R_current = sc.Rotation.from_quat(quat[:, i])
    # Update position and velocity using the rotated force
    opti.subject_to(pos[:, i + 1] == pos[:, i] + dt * vel[:, i])
    opti.subject_to(
        vel[:, i + 1] == vel[:, i] + dt * (1 / m) * ca.mtimes(R_current.as_matrix(), F)
    )
    # Quaternion update
    opti.subject_to(
        quat[:, i + 1] == quaternion_integration(quat[:, i], ang_vel[:, i], dt)
    )
    # Angular velocity update (using the standard rigid-body dynamics)
    opti.subject_to(
        ang_vel[:, i + 1]
        == ang_vel[:, i]
        + dt
        * ca.mtimes(ca.inv(J), M - ca.cross(ang_vel[:, i], ca.mtimes(J, ang_vel[:, i])))
    )

# Control bounds (for each time step)
for i in range(N):
    opti.subject_to(opti.bounded(-2, u[:, i], 2))
    opti.set_initial(quat[:, i], np.array([0, 0, 0, 1]))
    # (Example: restrict z-coordinate between 0 and 5)
    opti.subject_to(opti.bounded(0, pos[2, i], 5))
    opti.subject_to(pos[1, i] >= 0)

# --- Collision Constraints ---
# We enforce collision avoidance between the ellipsoidal robot (with center pos and shape P_R_base)
# and each polytope obstacle. Based on Proposition 3 from the paper, the following dual constraints
# are added (with Δ_min = safety distance).

for obs in map_env.obstacles:
    # For each obstacle, introduce dual (collision multiplier) variables per time step:
    # ξ ∈ ℝ³, μ_r ∈ ℝ, and ν ∈ ℝ.
    xi = opti.variable(3, N)
    mu_r = opti.variable(1, N)
    nu = opti.variable(1, N)

    V_O = obs.global_vertices()  # vertices of the polytope obstacle (3 x num_vertices)
    delta_min = obs.safety_distance

    # --- Initialize the dual variables to avoid singularities ---
    for i in range(N):
        # Following the suggestion in the paper for V-Sqrt/V-GI constraints,
        # initialize ξ away from zero (e.g. along the first coordinate)
        opti.set_initial(xi[:, i], np.array([2 * delta_min, 0, 0]))
        opti.set_initial(mu_r[0, i], 0)
        opti.set_initial(nu[0, i], 0.1)

    for i in range(N):
        # In our setting the ellipsoidal robot is defined by its center pos[:,i] and
        # rotated shape matrix R * P_R_base * Rᵀ. For the dual constraint we need the inverse:
        c_R = pos[:, i]
        R_current = sc.Rotation.from_quat(quat[:, i]).as_matrix()
        P_R_inv = R_current @ ca.DM(np.linalg.inv(P_R_base)) @ R_current.T

        # Dual constraint corresponding to -1/4 ||ξ||² - ξᵀ c_R - μ_r - ν ≥ Δ_min².
        opti.subject_to(
            -(1 / 4) * ca.dot(xi[:, i], xi[:, i])
            - ca.dot(xi[:, i], c_R)
            - mu_r[0, i]
            - nu[0, i]
            - delta_min**2
            >= 0
        )
        # Constraint from the vertex representation of the polytope obstacle:
        opti.subject_to(V_O.T @ xi[:, i] + mu_r[0, i] >= 0)
        # Enforce the norm condition: ||ξ||² ≥ 4 Δ_min².
        opti.subject_to(ca.dot(xi[:, i], xi[:, i]) >= 4 * delta_min**2)
        # Dual constraint linking ν and the ellipsoid shape:
        opti.subject_to(nu[0, i] ** 2 >= ca.dot(xi[:, i], ca.mtimes(P_R_inv, xi[:, i])))
        opti.subject_to(nu[0, i] >= 0)

# --- Cost Function ---
# Here, we simply penalize the squared distance between the current and desired positions.
cost = 0
for i in range(N):
    cost += u[:, i].T @ u[:, i]  # penalize control effort
    p, p_f = x[:3, i], xf[:3]
    cost += ca.mtimes((p - p_f).T, (p - p_f)) * 10
opti.minimize(cost)

# initial position
p_i = np.array([0, 0, 0])
p_f = np.array([8, 3, 0])
# interpolation of N points in straight line between p_i and p_f
p = np.linspace(p_i, p_f, N)
# opti.set_initial(x[:3, :], p.T)


# Set up and solve with IPOPT
opti.solver(
    "ipopt",
    {},
    {
        # "constr_viol_tol": 1e-3,
        # "acceptable_tol": 1e-2,
        "linear_solver": "ma27",
        "nlp_scaling_method": "none",
        "mu_strategy": "adaptive",
    },
)
sol = opti.solve()

# --- Visualization ---
plotter = pv.Plotter()
# Plot each obstacle (as a box for visualization) using its vertices.
for obs in map_env.obstacles:
    vertices = obs.global_vertices().T
    bounds = [
        vertices[:, 0].min(),
        vertices[:, 0].max(),
        vertices[:, 1].min(),
        vertices[:, 1].max(),
        vertices[:, 2].min(),
        vertices[:, 2].max(),
    ]
    cube = pv.Box(bounds=bounds)
    plotter.add_mesh(cube, color="salmon", opacity=1.0)

# Plot the ellipsoidal robot along the trajectory.
x_val = sol.value(x)[0, :]
y_val = sol.value(x)[1, :]
z_val = sol.value(x)[2, :]
R_list = [trf.Rotation.from_quat(sol.value(x[6:10, i])) for i in range(N)]

# Compute ellipsoid radii from the shape matrix (SVD of P_R_base)
U, s, _ = np.linalg.svd(P_R_base)
radii = 1.0 / np.sqrt(s)
radii = np.array([0.24, 0.24, 0.10])
for i in range(N):
    center = np.array([x_val[i], y_val[i], z_val[i]])
    R_i = R_list[i]
    ellipsoid = pv.ParametricEllipsoid(*radii)
    transform = np.eye(4)
    transform[:3, :3] = R_i.as_matrix()
    transform[:3, 3] = center
    ellipsoid.transform(transform)
    plotter.add_mesh(ellipsoid, color="skyblue", opacity=0.3)

trajectory = np.vstack((x_val, y_val, z_val)).T
plotter.show_grid()
plotter.add_axes_at_origin()
plotter.add_axes()

print("Final pos:", sol.value(x[:3, -1]))
print("Desired pos:", sol.value(xf[:3]))
print("Box 1 vertices:", obstacle1.global_vertices())
print("Box 2 vertices:", obstacle2.global_vertices())
print("Dt : ", sol.value(dt))

plotter.show()

print("Final quaternions (each row corresponds to a time step):")
print(sol.value(x[6:10, :].T))
