import casadi as ca
import spatial_casadi as sc
import scipy.spatial.transform as trf
import numpy as np
import pyvista as pv
import json
import os
from datetime import datetime
from time import time
import itertools
import multiprocessing as mp


class Obstacle:
    def __init__(self, vertices_local, position, rotation=np.eye(3), safety_distance=0.01):
        self.vertices_local = vertices_local
        self.position = position
        self.rotation = rotation
        self.safety_distance = safety_distance

    def global_vertices(self):
        return self.rotation @ self.vertices_local + self.position[:, None]


class Map:
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

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

class TrajectoryGeneration():
    def __init__(self, map = None):
        self.opti = ca.Opti()
        self.N = 40
        self.dt = 0.1

        self.map = map if map else Map()

        A = np.load("A_matrix.npy")
        J = np.load("J_matrix.npy")
        m = np.load("mass.npy")

        self.x = self.opti.variable(13, self.N)
        self.u = self.opti.variable(6, self.N)
        self.x0 = self.opti.parameter(13)
        self.xf = self.opti.parameter(13)

        self.J = self.opti.parameter(3, 3)
        self.A = self.opti.parameter(6, 6)
        self.m = self.opti.parameter(1)

        self.opti.set_value(self.J, J)
        self.opti.set_value(self.A, A)
        self.opti.set_value(self.m, m)

        # Set parameters
        self.opti.set_value(self.x0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])) 
        self.opti.set_value(self.xf, np.array([8, 3, 0, 0, 0, 0, 0.707, 0, 0, 0.707, 0, 0, 0]))
    
    def setup_problem(self):
        pos, vel, quat, ang_vel = self.x[:3, :], self.x[3:6, :], self.x[6:10, :], self.x[10:, :]
        N = self.N
        dt = self.dt
        m = self.m
        J = self.J
        A = self.A

        u = self.u


        for i in range(N-1):
            F = ca.mtimes(A[:3, :], u[:, i])
            M = ca.mtimes(A[3:, :], u[:, i])
            R_current = sc.Rotation.from_quat(quat[:, i])
            # Update position and velocity using the rotated force
            self.opti.subject_to(pos[:, i+1] == pos[:, i] + dt * vel[:, i])
            self.opti.subject_to(vel[:, i+1] == vel[:, i] + dt * (1/m) * ca.mtimes(R_current.as_matrix(), F))
            # Quaternion update
            self.opti.subject_to(quat[:, i+1] == quaternion_integration(quat[:, i], ang_vel[:, i], dt))
            # Angular velocity update (using the standard rigid-body dynamics)
            self.opti.subject_to(ang_vel[:, i+1] == ang_vel[:, i] + dt * ca.mtimes(ca.inv(J), M - ca.cross(ang_vel[:, i], ca.mtimes(J, ang_vel[:, i]))))
        
        # Control bounds (for each time step)
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(-2, u[:, i], 2))
            self.opti.set_initial(quat[:, i], np.array([0, 0, 0, 1]))
            # (Example: restrict z-coordinate between 0 and 5)
            self.opti.subject_to(self.opti.bounded(0, pos[2, i], 5))
            self.opti.subject_to(pos[1, i] >= 0)
        P_R_base = np.diag([1 / 0.24**2, 1 / 0.24**2, 1 / 0.10**2])
        
        for obs in self.map.obstacles:
            # For each obstacle, introduce dual (collision multiplier) variables per time step:
            # ξ ∈ ℝ³, μ_r ∈ ℝ, and ν ∈ ℝ.
            xi = self.opti.variable(3, N)
            mu_r = self.opti.variable(1, N)
            nu = self.opti.variable(1, N)

            V_O = obs.global_vertices()  # vertices of the polytope obstacle (3 x num_vertices)
            delta_min = obs.safety_distance

            # --- Initialize the dual variables to avoid singularities ---
            for i in range(N):
                # Following the suggestion in the paper for V-Sqrt/V-GI constraints,
                # initialize ξ away from zero (e.g. along the first coordinate)
                self.opti.set_initial(xi[:, i], np.array([2*delta_min, 0, 0]))
                self.opti.set_initial(mu_r[0, i], 0)
                self.opti.set_initial(nu[0, i], 0.1)

            for i in range(N):
                # In our setting the ellipsoidal robot is defined by its center pos[:,i] and
                # rotated shape matrix R * P_R_base * Rᵀ. For the dual constraint we need the inverse:
                c_R = pos[:, i]
                R_current = sc.Rotation.from_quat(quat[:, i]).as_matrix()
                P_R_inv = R_current @ ca.DM(np.linalg.inv(P_R_base)) @ R_current.T

                # Dual constraint corresponding to -1/4 ||ξ||² - ξᵀ c_R - μ_r - ν ≥ Δ_min².
                self.opti.subject_to(- (1/4) * ca.dot(xi[:, i], xi[:, i]) - ca.dot(xi[:, i], c_R) - mu_r[0, i] - nu[0, i] - delta_min**2 >= 0)
                # Constraint from the vertex representation of the polytope obstacle:
                self.opti.subject_to(V_O.T @ xi[:, i] + mu_r[0, i] >= 0)
                # Enforce the norm condition: ||ξ||² ≥ 4 Δ_min².
                self.opti.subject_to(ca.dot(xi[:, i], xi[:, i]) >= 4 * delta_min**2)
                # Dual constraint linking ν and the ellipsoid shape:
                self.opti.subject_to(nu[0, i]**2 >= ca.dot(xi[:, i], ca.mtimes(P_R_inv, xi[:, i])))
                self.opti.subject_to(nu[0, i] >= 0)



        # Here, we simply penalize the squared distance between the current and desired positions.
        cost = 0
        for i in range(N):
            cost += u[:, i].T @ u[:, i]  # penalize control effort
            p, p_f = self.x[:3, i], self.xf[:3]
            cost += ca.mtimes((p - p_f).T, (p - p_f)) * 10
        self.opti.minimize(cost)

        self.opti.solver("ipopt", {}, {"max_cpu_time" : 120, "linear_solver" : "ma27","nlp_scaling_method": "none",    "mu_strategy": "adaptive"})
        time_start = time()
        sol = self.opti.solve()

        if sol is not None: 
            x_ = sol.value(self.x)
            u_ = sol.value(self.u)
            filename = "solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez(filename, x=x_, u=u_)
            return {
                "status" : "success",
                "total_time " : time() - time_start,
                "error " : np.linalg.norm(x_[0:3, -1] - sol.value(self.xf[0:3])),
                "opts" : "None"
            }



def run_instance(param_set):
    problem = TrajectoryGeneration()

if __name__ == "__main__":
    map_env = Map()
    vertices_obstacle = np.array([
        [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
    ])
    position_obstacle = np.array([2.0, 0, 0])
    rotation_obstacle = np.eye(3)
    safety_dist_obstacle = 0.05
    obstacle1 = Obstacle(vertices_obstacle, position_obstacle, rotation_obstacle, safety_dist_obstacle)
    map_env.add_obstacle(obstacle1)

    position_obstacle = np.array([0.0, 2.3, 0.0]) + position_obstacle
    obstacle2 = Obstacle(vertices_obstacle, position_obstacle, rotation_obstacle, safety_dist_obstacle)
    map_env.add_obstacle(obstacle2)

    position_obstacle = np.array([-2.0, 0.0, 0.0]) + position_obstacle
    obstacle3 = Obstacle(vertices_obstacle, position_obstacle, rotation_obstacle, safety_dist_obstacle)
    map_env.add_obstacle(obstacle3)

    exponents = np.arange(-4, 4)
    scale_factors = 10.0 ** exponents

    problem  = TrajectoryGeneration(map_env)
    print(problem.setup_problem())
    