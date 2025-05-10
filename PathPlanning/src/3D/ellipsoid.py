#!/usr/bin/env python
# coding: utf-8

# In[8]:


import casadi as ca
import spatial_casadi as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[9]:


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


# In[10]:


def quaternion_integration(q, w, dt):
    """Quaternion integration using Rodrigues formula"""
    w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)
    q_ = ca.vertcat(w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2))
    return quaternion_multiplication(q_, q)


# In[16]:


opti = ca.Opti()

N = 40
dt = 0.1

# Variables
x = opti.variable(3, N)
v = opti.variable(3, N)
q = opti.variable(4, N)
w = opti.variable(3, N)

# lambda_ = opti.variable(1, N)

# Actuation
u = opti.variable(6, N)

A = opti.parameter(6, 6)
J = opti.parameter(3, 3)
m = opti.parameter()

G = opti.parameter(3, 3)
G_ = np.diag([0.24**2, 0.24**2, 0.10**2])
opti.set_value(G, G_)

# obstacle


# Mecanical parameters
opti.set_value(A, np.load("A_matrix.npy"))
opti.set_value(J, np.load("J_matrix.npy"))
opti.set_value(m, 1.0)

for i in range(N - 1):
    F = ca.mtimes(A[:3, :], u[:, i])
    M = ca.mtimes(A[3:, :], u[:, i])

    R = sc.Rotation.from_quat(q[:, i])

    ## Dynamic model
    opti.subject_to(x[:, i + 1] == x[:, i] + v[:, i] * dt)
    opti.subject_to(v[:, i + 1] == v[:, i] + R.as_matrix() @ F / m * dt)

    q_ = quaternion_integration(q[:, i], w[:, i], dt)
    opti.subject_to(q[:, i + 1] == q_)
    opti.subject_to(
        w[:, i + 1] == w[:, i] + ca.inv(J) @ (M - ca.cross(w[:, i], J @ w[:, i])) * dt
    )


for i in range(N):
    opti.subject_to(opti.bounded(-2, u[:, i], 2))

# Final State constraints
opti.subject_to(v[:, N - 1] == np.zeros((3, 1)))
opti.subject_to(w[:, N - 1] == np.zeros((3, 1)))

# Initial State constraints
# Lets assume we are resting at the origin for simplicity
opti.subject_to(x[:, 0] == np.zeros((3, 1)))
opti.subject_to(v[:, 0] == np.zeros((3, 1)))
opti.subject_to(q[:, 0] == np.array([0, 0, 0, 1]))
opti.subject_to(w[:, 0] == np.zeros((3, 1)))

# Obstacle definition
# Ellipoid centered at (2.5, 2.5, 2.5) with semi-axes 0.5, 0.5, 0.5
M_m_ = np.diag([3.5**2, 3.5**2, 1.5**2])
t_m_ = np.array([2.5, 2.5, 2.5])

M_m = opti.parameter(3, 3)
t_m = opti.parameter(3, 1)

opti.set_value(M_m, M_m_)
opti.set_value(t_m, t_m_)


# Obstacle avoidance constraints
# First we will use the obstacles avoidance constraints with a single obstacle and a fixed gamma value - this will leed to suboptimality,
#  but will reduce the non linearity of the colision avoidance constraints
for i in range(N):
    R = sc.Rotation.from_quat(q[:, i])

    ## Auxiliar variables for the constraints
    eta = (
        x[:, i] - t_m
    )  # differences between the center of the obstacle and the drone at time i
    G_tilde = R.as_matrix() @ G @ R.as_matrix().T  # Ellipsoid matrix in the drone frame
    gamma_m_k_t = (
        1 / 2 * ca.log((eta.T @ M_m @ eta) / (eta.T @ G_tilde @ eta))
    )  # Multiplication factor

    # Colision avoidance constraint - the center of the difference between the drone and the obstacle ellipsoid should be outside one of the Minowski sum ellipsoids
    opti.subject_to(
        1
        <= eta.T
        @ ca.inv((1 + ca.exp(gamma_m_k_t)) @ G_tilde + (1 + ca.exp(-gamma_m_k_t)) @ M_m)
        @ eta
    )

# desired position
x_d = opti.parameter(3, 1)
opti.set_value(x_d, np.array([5, 5, 5]))

# Desired attitude
q_d = opti.parameter(4, 1)
opti.set_value(q_d, np.array([0, 0, 0, 1.0]))


# cost function
J = 0
for i in range(N):
    # J += u[:, i].T @ u[:, i] # Actuation cost
    J += (x[:, i] - x_d).T @ (x[:, i] - x_d)
    J += 1 - q[:, i].T @ q_d

opti.minimize(J)

opti.solver("ipopt")


for i in range(N):
    opti.set_initial(q[:, i], np.array([0, 0, 0, 1]))

sol = opti.solve()

# In[]:

from scipy.spatial.transform import Rotation

np.printoptions(precision=2, suppress=True)
for i in range(N):
    eta = sol.value(x[:, i]) - t_m_
    G_tilde = (
        Rotation.from_quat(sol.value(q[:, i])).as_matrix()
        @ G_
        @ Rotation.from_quat(sol.value(q[:, i])).as_matrix().T
    )
    gamma_m_k_t = 1 / 2 * np.log((eta.T @ M_m_ @ eta) / (eta.T @ G_tilde @ eta))
    print(np.round(gamma_m_k_t, 3), end=" ")
    print(
        np.round(
            eta.T
            @ np.linalg.inv(
                (1 + np.exp(gamma_m_k_t)) * G_tilde + (1 + np.exp(-gamma_m_k_t)) * M_m_
            )
            @ eta
        ),
        3,
        end=" ",
    )
    print(np.round(sol.value(x[:, i]), 3), end=" ")
    print(np.linalg.norm(sol.value(x[:, i] - t_m)))

print("\n\n\n\n\n")


import pyvista as pv

drone_mesh = pv.ParametricEllipsoid(0.24, 0.24, 0.10)
position = sol.value(x).T.tolist()
attitude = [Rotation.from_quat(q) for q in sol.value(q).T.tolist()]


p = pv.Plotter()
for p_, r in zip(position, attitude):
    print(p_, " ", r.as_euler("xyz"))
    drone_mesh_copy = drone_mesh.copy()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = p_
    transformation_matrix[:3, :3] = r.as_matrix()

    drone_mesh_copy.transform(transformation_matrix)
    p.add_mesh(drone_mesh_copy, color="b", opacity=0.5)

# Create ellipsoids for obstacle
ellipsoid = pv.ParametricEllipsoid(
    np.sqrt(M_m_[0, 0]), np.sqrt(M_m_[1, 1]), np.sqrt(M_m_[2, 2])
)
ellipsoid = ellipsoid.translate([2.5, 2.5, 2.5])

# Add mesh to the plotter
p.add_mesh(ellipsoid, color="r", opacity=1.0)
p.add_axes_at_origin()
p.add_floor()

p.show()

# %%
