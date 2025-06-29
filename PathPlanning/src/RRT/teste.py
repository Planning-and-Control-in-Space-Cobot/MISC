import casadi as ca
import spatial_casadi as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform as trf

from Robot import Robot

import pyvista as pv

import os

# === Load Robot Parameters ===
script_dir = os.path.dirname(os.path.abspath(__file__))
A = np.load(os.path.join(script_dir, "A_matrix.npy"))
J = np.load(os.path.join(script_dir, "J_matrix.npy"))
m = np.load(os.path.join(script_dir, "mass.npy"))

# === Define Robot Instance (dummy for dynamics only) ===
robot = Robot(J=J, A=A, m=m, fcl_obj=None, mesh=None)

# === Trajectory Optimization Setup ===
N = 20
opti = ca.Opti()

# State: x, v, q (quat), w
x = opti.variable(13, N)

# Control: u
u = opti.variable(6, N - 1)

# Time step
dt = opti.variable()

# Initial and goal states (set arbitrarily)
x0 = np.array([0.0, 0.0, 0.0])
xT = np.array([1.5, 1.0, 0.8])
v0 = np.zeros(3)
w0 = np.zeros(3)
q0 = np.array([0.707, 0, 0, 0.707])
qT = np.array([0, 0, 0, 1.0])

opti.subject_to(x[0:3, 0] == x0)
opti.subject_to(x[3:6, 0] == v0)
opti.subject_to(x[10:13, 0] == w0)

opti.subject_to(x[0:3, -1] == xT)
opti.subject_to(x[6:10, -1] == qT)

# Dynamics constraints
for k in range(N - 1):
    opti.subject_to(x[:, k + 1] == robot.f(x[:, k], u[:, k], dt))

# === Bounds ===
v_max = 2.0
w_max = 1.0

for i in range(N):
    opti.subject_to(ca.sumsqr(x[3:6, i]) <= v_max**2)
    opti.subject_to(ca.sumsqr(x[10:13, i]) <= w_max**2)
    opti.subject_to(ca.sumsqr(x[6:10, i]) == 1)  # Quaternion normalization

# Obstacle avoidance constraints
# Define vertices for the polytope robot 
# Box (0.45, 0.45, 0.12) around the origin
polytope_vertices = np.array([
    [-0.225, -0.225, -0.06],
    [0.225, -0.225, -0.06],
    [0.225, 0.225, -0.06],
    [-0.225, 0.225, -0.06],
    [-0.225, -0.225, 0.06],
    [0.225, -0.225, 0.06],
    [0.225, 0.225, 0.06],
    [-0.225, 0.225, 0.06]
])

normal_left = ca.DM([0, 1, 0])
normal_right = ca.DM([1, 0, 0])

a_left =  -0.200 
a_right = 1.230
cost = 0
for i in range(N):
    # Position of the robot at time step i 
    pos = x[0:3, i]
    quat = x[6:10, i]

    R = sc.Rotation.from_quat(quat).as_matrix()
    for vertex in polytope_vertices:
        opti.subject_to(
            normal_left.T @ (R @ vertex + pos) >= a_left
        )
        opti.subject_to(
            normal_right.T @ (R @ vertex + pos) <= a_right
        )

# Time step bounds
opti.subject_to(dt >= 0.05)
opti.set_initial(dt, 0.1)
opti.subject_to(dt <= 0.3)

# === Cost Function: Minimize total time ===
#cost += dt * 1000
for i in range(N - 1):
    cost += u[:, i].T @ 0.1 @ u[:, i]

opti.minimize(cost)

# === Solver ===
opti.solver("ipopt", {}, {
    "max_iter": 10000,
    "warm_start_init_point": "yes",        # Use initial guess
    "linear_solver": "ma97",
    "mu_strategy" : "adaptive",
    "hessian_approximation" : "limited-memory",
})


for i in range(N):
    opti.set_initial(x[6:10, i], qT)

try:
    sol = opti.solve_limited()
except Exception as e:
    x_debug = opti.debug.value(x)
    u_debug = opti.debug.value(u)
    dt_debug = opti.debug.value(dt)

    print("Optimization failed:", e)
    print("Debug values:")
    print("x:", x_debug)
    print("u:", u_debug)
    print("dt:", dt_debug)
    opti.debug.show_infeasibilities()
    exit(1)


pv_ = pv.Plotter() 
for i in range(N):
    pos = sol.value(x[0:3, i])
    quat = sol.value(x[6:10, i])
    R = trf.Rotation.from_quat(quat).as_matrix()
    print(f"i {i}, pos: {pos}, quat: {quat}")

    # Create a box mesh for the robot
    box = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    box.transform(T)

    pv_.add_mesh(box, color='blue', show_edges=True)
plane = pv.Plane(center=(0, a_left, 0), direction=(0, 1, 0))
pv_.add_mesh(plane, color='green', show_edges=True)
plane = pv.Plane(center=(0, a_right, 0), direction=(0, -1, 0), i_size=10, j_size= 10)
pv_.add_mesh(plane, color='green', show_edges=True)
pv_.add_text("Robot Path", position='upper_edge', font_size=20, color='black')
pv_.show_grid()
pv_.add_axes()
pv_.show()



