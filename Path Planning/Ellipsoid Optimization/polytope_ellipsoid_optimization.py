# 2D Polytope - Ellipsoid Obstacle avoidance considering attitude on both polytope and ellipsoid 
# This method is based on the paper "Efficient Collision Modelling for Numerical Optimal Control" - Section 3.B
# 3D Polytope - Ellipsoid Obstacle avoidance considering full SO(3) attitude
# 3D Ellipsoidal Robot - Polytope Obstacle avoidance using dual constraints (Proposition 3)
# 3D Ellipsoidal Robot with Rotation - Polytope Obstacle avoidance using dual constraints (Proposition 3)
# 3D Ellipsoidal Robot with Rotation - Polytope Obstacle avoidance using dual constraints (Proposition 3)
import casadi as ca
import numpy as np
import pyvista as pv

import spatial_casadi as sc

# Problem parameters
N = 20
n_w = 3
n_ov = 8

V_local_obstacle = np.array([
    [-0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5],
    [-0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5]
])

obstacle_pos = np.array([2.5, 2.0, 1.5])
V_O = V_local_obstacle + obstacle_pos[:, None]

P_R_base = np.diag([0.24, 0.24, 0.10])
delta_min = 0.01

opti = ca.Opti()

x = opti.variable(N)
y = opti.variable(N)
z = opti.variable(N)
roll = opti.variable(N)
pitch = opti.variable(N)
yaw = opti.variable(N)

xi = opti.variable(n_w, N)
mu_o = opti.variable(1, N)
nu = opti.variable(1, N)

J = 0
for i in range(N - 1):
    J += (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 + (z[i+1] - z[i])**2
opti.minimize(J)

opti.subject_to(x[0] == -3)
opti.subject_to(y[0] == -3)
opti.subject_to(z[0] == 0)
opti.subject_to(x[-1] == 6)
opti.subject_to(y[-1] == 6)
opti.subject_to(z[-1] == 3)

for i in range(N):
    c_R = ca.vertcat(x[i], y[i], z[i])

    cr = ca.cos(roll[i]); sr = ca.sin(roll[i])
    cp = ca.cos(pitch[i]); sp = ca.sin(pitch[i])
    cy = ca.cos(yaw[i]); sy = ca.sin(yaw[i])

    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp,   cp*sr,            cp*cr)
    )
    P_R_inv = R @ ca.DM(np.linalg.inv(P_R_base)) @ R.T

    con1 = (1/4) * ca.dot(xi[:, i], xi[:, i]) + ca.dot(xi[:, i], c_R) + mu_o[0, i] + nu[0, i] + delta_min**2
    con2 = - V_O.T @ xi[:, i] - mu_o[0, i]
    con3 = ca.dot(xi[:, i], xi[:, i]) - 4 * delta_min**2
    xi_Pinv_xi = xi[:, i].T @ P_R_inv @ xi[:, i]
    con4 = nu[0, i]**2 - xi_Pinv_xi

    opti.subject_to(con1 <= 0)
    opti.subject_to(con2 <= 0)
    opti.subject_to(con3 >= 0)
    opti.subject_to(con4 >= 0)
    opti.subject_to(nu[0, i] >= 0)

opti.solver("ipopt")
sol = opti.solve()

x_val = sol.value(x)
y_val = sol.value(y)
z_val = sol.value(z)
roll_val = sol.value(roll)
pitch_val = sol.value(pitch)
yaw_val = sol.value(yaw)

plotter = pv.Plotter()

# Draw obstacle (box)
faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],
         [2,3,7,6],[1,2,6,5],[4,7,3,0]]
verts = [V_O[:, f].T for f in faces]

for face in verts:
    plotter.add_mesh(pv.PolyData(face), color="salmon", opacity=0.4)

# Draw ellipsoidal robot at each step
for i in range(N):
    c = np.array([x_val[i], y_val[i], z_val[i]])

    cr, sr = np.cos(roll_val[i]), np.sin(roll_val[i])
    cp, sp = np.cos(pitch_val[i]), np.sin(pitch_val[i])
    cy, sy = np.cos(yaw_val[i]), np.sin(yaw_val[i])

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    U, s, _ = np.linalg.svd(P_R_base)
    radii = 1. / np.sqrt(s)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_e = radii[0] * np.cos(u) * np.sin(v)
    y_e = radii[1] * np.sin(u) * np.sin(v)
    z_e = radii[2] * np.cos(v)
    ellipsoid = np.array([x_e, y_e, z_e]).reshape(3, -1)
    rotated = R @ ellipsoid
    X = rotated[0].reshape(u.shape) + c[0]
    Y = rotated[1].reshape(u.shape) + c[1]
    Z = rotated[2].reshape(u.shape) + c[2]
    surf = pv.StructuredGrid(X, Y, Z)
    plotter.add_mesh(surf, color="skyblue", opacity=0.3)

# Plot trajectory
trajectory = np.vstack((x_val, y_val, z_val)).T
plotter.add_lines(trajectory, color="black")
plotter.show_grid()
plotter.show()
