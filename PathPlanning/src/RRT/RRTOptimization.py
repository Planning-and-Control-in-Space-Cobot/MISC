import open3d as o3d
import pyvista as pv 
import numpy as np 
import fcl
from scipy.spatial.transform import Rotation as R, Slerp

import casadi as ca
import spatial_casadi as sc

from typing import List, Tuple

import os

class Robot: 
    def __init__ (self, J, A, m, ellipsoid_radius):
        self.J = J
        self.A = A
        self.m = m
        self.ellipsoid_radius = ellipsoid_radius
    
    def f(self, state, u, dt):
        def unflat(state, u):
            x = state[0:3]
            v = state[3:6]
            q = state[6:10]
            w = state[10:13]
            return x, v, q, w, self.A[0:3, :] @ u, self.A[3:6, :] @ u
        def flat(x, v, q, w):

            return ca.vertcat(x, v, q, w)
        def quat_mul(q1, q2):
            q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
            q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]
            q_ = ca.vertcat(
                ca.horzcat(q1w, q1z, -q1y, q1x),
                ca.horzcat(-q1z, q1w, q1x, q1y),
                ca.horzcat(q1y, -q1x, q1w, q1z),
                ca.horzcat(-q1x, -q1y, -q1z, q1w),
            )
            return q_ @ ca.vertcat(q2x, q2y, q2z, q2w)
        def quat_int(q, w, dt):
            w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)
            q_ = ca.vertcat(w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2))
            return quat_mul(q_, q)
        x, v, q, w, F, M = unflat(state, u)
        R = sc.Rotation.from_quat(q)
        x_next = x + v * dt
        v_next = v + dt * (1/ self.m) * R.as_matrix() @ F
        q_next = quat_int(q, w, dt)
        w_next = w + dt * ca.inv(self.J) @ (M - ca.cross(w, self.J @ w)) 
        return flat(x_next, v_next, q_next, w_next)
        

class Obstacle:
    def __init__(self, normal, closest_point_surface, iteration): 
        #assert(normal.shape == (3,)), "Normal vector should be of shape (3,)"
        #assert(np.linalg.norm(normal) == 1), "Normal vector should be of unit length"
        #assert(closest_point_surface.shape == (3,)), "Closest point on surface should be of shape (3,)"
        
        self.normal = normal
        self.closest_point_surface = closest_point_surface
        self.iteration = iteration
        self.closest_point_surface = closest_point_surface


class RRTPathOptimization:
    def __init__(self, 
                robot = Robot(np.eye(3), np.eye(3), 1, np.ones((3,)))):
        self.robot = robot

    def optimize(self,
            initial_path, 
            obstacles : List[Obstacle],  
            xi = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            xf=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 
            dt=1, 
            prev_u=None
        ):

        self.opti = ca.Opti()
        N  = len(initial_path)
        self.N = N

        # Define the optimization variables
        self.x = self.opti.variable(13, N) # [3 pos; 3 vel; 4 quat; 3 ang_vel]
        # pos - ellipsoid center position
        # vel - ellipsoid center velocity
        # quat - quaternion orientation of the ellipsoid
        # ang_vel - angular velocity of the ellipsoid
        self.u = self.opti.variable(6, N)
        # u - control inputs (forces and moments)
        self.dt = self.opti.variable(1)
        self.opti.subject_to(self.dt >= 0.01)
        self.opti.set_initial(self.dt, 5)


        # Dynamic Paremeters
        A = self.opti.parameter(6, 6)
        J = self.opti.parameter(3, 3)
        m = self.opti.parameter(1)
        ellipsoid_radius = self.opti.parameter(3, 3)
        self.xf = self.opti.parameter(13)
        self.xi = self.opti.parameter(13)
        self.opti.set_value(self.xi, xi)
        self.opti.set_value(self.xf, xf)
        self.opti.subject_to(self.x[:, -1] == self.xf) # Initial State
        self.opti.subject_to(self.x[:, 0] == self.xi) # Final State

        self.opti.set_value(A, self.robot.A) # A matrix
        self.opti.set_value(J, self.robot.J) # J matrix
        self.opti.set_value(m, self.robot.m) # mass
        self.opti.set_value(ellipsoid_radius, np.diag(self.robot.ellipsoid_radius)) # Ellipsoid radius

        # Define the dynamics of the system
        for i in range(N-1):
            self.opti.subject_to(self.x[:, i+1] == self.robot.f(self.x[:, i], self.u[:, i], self.dt))
        
        # Define the obstacle avodiance constraints
        # dist = normal (p_{closest_point} - p_{closest_ellipsoid}) - sqrt(n R Q R^T n^T) >= 0
        ## Add the obstacle avoidance constraints, it will be **at least** 1 per step in the horizon, but there can be more than 1 per step,
        ## When we backtrack we will have at least 1 step with more than 1 obs least 1 step with more than 1 obstacle
        num_obstacles = len(obstacles)
        self.num_obstacles = num_obstacles
        self.obstacle_surface_points = self.opti.parameter(3, num_obstacles)         
        self.normals = self.opti.parameter(3, num_obstacles)
        for i, obstacle in enumerate(obstacles):
            iter = obstacle.iteration
            self.opti.set_value(self.obstacle_surface_points[:, iter], obstacle.closest_point_surface)
            self.opti.set_value(self.normals[:, iter], obstacle.normal)

            R = sc.Rotation.from_quat(self.x[6:10, iter])
            dist = self.normals[:, iter].T @ (self.x[0:3, iter] - self.obstacle_surface_points[:, iter]) - \
                ca.sqrt(self.normals[:, iter].T @ R.as_matrix() @ ellipsoid_radius @ R.as_matrix().T @ self.normals[:, iter])
            self.opti.subject_to(dist > 0 + 1e-1) # Zero + safety margin
        
        # State and actuation constraints 
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))
        
        # Define the cost function
        cost = 0
        cost += 100 * self.dt
        for i in range(N): 
            cost += self.u[:, i].T @ self.u[:, i]

        self.opti.minimize(cost)

        self.opti.solver("ipopt",
            {}, 
            {
                "max_iter":1, 
                "acceptable_iter": 1,
                "print_level": 0,

            }
        )


        self.opti.set_initial(self.dt, dt)

        for i in range(self.N):
            self.opti.set_initial(self.x[0:3, i], initial_path[i].x)  
            self.opti.set_initial(self.x[3:6, i], initial_path[i].v)
            self.opti.set_initial(self.x[6:10, i], initial_path[i].q)
            self.opti.set_initial(self.x[10:13, i], initial_path[i].w)
        
        if prev_u is not None:
            self.opti.set_initial(self.u, prev_u)
        sol = self.opti.solve()

    def visualize_trajectory(self, initial_path, optimized_path, voxel_mesh):
        plotter = pv.Plotter()
        plotter.add_mesh(voxel_mesh, color="red", opacity=0.4)

        for s in initial_path:
            mesh = pv.ParametricEllipsoid(*self.robot.ellipsoid_radius)
            T = np.eye(4)
            T[:3, :3] = R.from_quat(s.q).as_matrix()
            T[:3, 3] = s.x
            mesh.transform(T)
            plotter.add_mesh(mesh, color="green", opacity=0.4)
        for s in optimized_path:
            mesh = pv.ParametricEllipsoid(*self.robot.ellipsoid_radius)
            T = np.eye(4)
            T[:3, :3] = R.from_quat(s.q).as_matrix()
            T[:3, 3] = s.x
            mesh.transform(T)
            plotter.add_mesh(mesh, color="blue", opacity=0.4)
        plotter.add_axes()
        plotter.export_html("OptimizedTrajectory.html")
        exit()
        plotter.show()

def main():
    executable_dir = os.path.dirname(os.path.abspath(__file__))
    J = np.load(os.path.join(executable_dir, "J_matrix.npy"))
    A = np.load(os.path.join(executable_dir, "A_matrix.npy"))
    m = np.load(os.path.join(executable_dir, "mass.npy"))
    ellipsoid_radius = np.load(os.path.join(executable_dir, "ellipsoidal_radius.npy"))

    robot = Robot(J, A, m, ellipsoid_radius)
            
    rrtOptimizer = RttPathOptimization(robot)
    rrtOptimizer.setup_optimization()


if __name__ == "__main__":
    main()
        