import open3d as o3d
import pyvista as pv
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R, Slerp

import casadi as ca
import spatial_casadi as sc

from typing import List, Tuple

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from RRT import RRTState
from Environment import EnvironmentHandler
import pyvista as pv

class Robot:
    def __init__(
        self, J: np.ndarray, A: np.ndarray, m: float, fcl_obj : fcl.CollisionGeometry 
    ):
        """
        Robot class to represent the robot in the optimization problem

        Parameters
            J (np.ndarray): inertia matrix of the robot
            A (np.ndarray): actuation matrix of the robot
            m (float): mass of the robot
            ellipsoid_radius (np.ndarray): radius of the ellipsoid that
                represents the robot

        Returns
            None
        """
        if (
            np.shape(J) != (3, 3)
            or np.shape(A) != (6, 6)
            or m <= 0
        ):
            raise ValueError(
                "J must be a 3x3 matrix, A must be a 6x6 matrix, ellipsoid_radius must be a 3x1 vector and m must be a positive scalar"
            )

        self.J = J
        self.A = A
        self.m = m
        self.fcl_obj = fcl_obj

    

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
            q_ = ca.vertcat(
                w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2)
            )
            return quat_mul(q_, q)

        x, v, q, w, F, M = unflat(state, u)
        R = sc.Rotation.from_quat(q)
        x_next = x + v * dt
        v_next = v + dt * (1 / self.m) * R.as_matrix() @ F
        q_next = quat_int(q, w, dt)
        w_next = w + dt * ca.inv(self.J) @ (M - ca.cross(w, self.J @ w))
        return flat(x_next, v_next, q_next, w_next)

class Obstacle:
    def __init__(
        self,
        closestPointRobot : np.ndarray, 
        closestPointObstacle : np.ndarray, 
        translation : np.ndarray, 
        iteration : int,
        safetyMargin : float = 0.01):
        '''
            This class represents an obstacle in the optimization problem

            Parameters
                closestPointRobot (np.ndarray): closest point on the robot
                closestPointObstacle (np.ndarray): closest point on the obstacle
                translation (np.ndarray): translation from the center of the robot to the closest point in the robot
                iteration (int) :  the iteration in the optimization problem that this obstacle plane should be considered
                safetyMargin (float): safety margin for the obstacle
        '''

        self.closestPointRobot = closestPointRobot
        self.closestPointObstacle = closestPointObstacle
        self.translation = translation
        self.iteration = iteration
        self.normal = (self.closestPointRobot - self.closestPointObstacle) / np.linalg.norm(self.closestPointRobot - self.closestPointObstacle)
        self.minDistance = np.linalg.norm(self.closestPointRobot - self.closestPointObstacle)
        self.safetyMargin = safetyMargin
    
    def generateSquare(self):
        """
        Generate 4 vertices of a square in 3D space lying in a plane.

        Parameters:
        - p0 (array-like): 3D point on the plane (shape: (3,))
        - n (array-like): 3D normal vector of the plane (shape: (3,))
        - d (float): Half side length of the square (i.e., square is 2d x 2d)

        Returns:
        - V (np.ndarray): 3x4 matrix with columns as the 4 square vertices in 3D
        """
        # Pick a helper vector that is not parallel to n
        helper = np.array([0, 1, 0]) if not np.allclose(self.normal, [0, 1, 0]) else np.array([1, 0, 0])

        # Generate two orthonormal vectors in the plane
        u = np.cross(self.normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u)
        v /= np.linalg.norm(v)

        minDistance_ = self.minDistance
        self.minDistance = max(self.minDistance, 2)

        # Compute square vertices
        V = np.array([
            self.closestPointObstacle + self.minDistance*u + self.minDistance*v,
            self.closestPointObstacle - self.minDistance*u + self.minDistance*v,
            self.closestPointObstacle - self.minDistance*u - self.minDistance*v,
            self.closestPointObstacle + self.minDistance*u - self.minDistance*v
        ]).T  # shape (3, 4)
        self.minDistance = minDistance_

        return V
        
class OptimizationState:
    """
    A class to represent the each state / step in the optimization, this class
    will also work as an interface for the optimization problem
    """

    def __init__(
        self,
        x: np.ndarray,
        q: np.ndarray,
        v: np.ndarray = np.zeros((3, 1)),
        w: np.ndarray = np.zeros((3, 1)),
        u: np.ndarray = np.zeros((6, 1)),
        i: np.ndarray = 0,
    ):
        self.x = x  # position
        self.v = v  # velocity
        self.q = q  # quaternion
        self.w = w  # angular velocity
        self.u = u  # control inputs
        self.i = i  # index of the state in the optimization problem

    def get_state(self) -> np.ndarray:
        """
        Return the state in a flatten numpy array

        Return
            np.ndarray: flatten state - (13,)
        """
        return np.hstack([self.x, self.v, self.q, self.w])

class RRTPathOptimization:
    def __init__(
        self,
        stateMinValues: np.ndarray,
        stateMaxValues: np.ndarray,
        env : EnvironmentHandler,
        robot : Robot,
    ):
        self.robot = robot
        self.stateMinValues = stateMinValues
        self.stateMaxValues = stateMaxValues

    def setup_optimization(
        self,
        initial_path: List[OptimizationState],
        obstacles: List[Obstacle],
        xi: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        xf: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ):
        """
        Setups the optimization problem, this function only needs to be called
        once, while we maintain the same number of obstacles and collisions

        Parameters
            initial_path (List[OptimizationState]): initial path to optimize
            obstacles (List[Obstacle]): list of obstacles to avoid
            collisions (List[Collision]): list of collisions to avoid
            xi (np.ndarray): initial state of the robot
            xf (np.ndarray): final state of the robot

        Returns
            None
        """
        self.opti = ca.Opti()

        self.num_obstacles = len(obstacles)
        self.N = len(initial_path)
        if self.N != self.num_obstacles + 2:
            raise ValueError("We need to consider one obstacle per step in the horizon except first and last, since they are the start and end states")

        # Define the optimization variables
        self.x = self.opti.variable(13, self.N)  # [3 pos; 3 vel; 4 quat; 3 ang_vel]
        self.u = self.opti.variable(6, self.N)

        # The start position of the robot in each of the steps
        self.startPos = self.opti.parameter(3, self.N)

        # The distance to the obstacle in each of the steps
        self.obstacleDistance = self.opti.parameter(1, self.N)

        self.dt = self.opti.variable(1)
        self.opti.subject_to(self.dt > 0)
        self.opti.subject_to(self.dt <0.2)

        # Optimization parameters
        self.xf = self.opti.parameter(13)
        self.xi = self.opti.parameter(13)
        self.opti.set_value(self.xi, xi)
        self.opti.set_value(self.xf, xf)
        self.opti.subject_to(self.x[:, -1] == self.xf)  # Initial State
        self.opti.subject_to(self.x[:, 0] == self.xi)  # Final State
        # Dynamic constraints
        for i in range(self.N - 1):
            # x[k+1] = f(x[k], u[k], dt)
            self.opti.subject_to(
                self.x[:, i + 1] == self.robot.f(self.x[:, i], self.u[:, i], self.dt)
            )

        self.xi_ = self.opti.variable(3,self.N - 2)
        self.mu_r_ = self.opti.variable(self.N - 2)
        self.nu_ = self.opti.variable(self.N - 2)

        self.distance = self.opti.parameter(1, self.num_obstacles)

        P_R_Base = np.diag([1 / 0.24**2, 1 / 0.24**2, 1 / 0.10**2])

        for i, obstacle in  enumerate(obstacles):

            self.opti.set_initial(self.xi_[:, i], np.array([obstacle.safetyMargin * 2, 0, 0]))
            self.opti.set_initial(self.mu_r_[i], 0)   
            self.opti.set_initial(self.nu_[i], 0)

            c_R = self.x[0:3, i + 1]
            R_current = sc.Rotation.from_quat(self.x[6:10, i + 1]).as_matrix()
            P_R_inv = R_current @ ca.DM(np.linalg.inv(P_R_Base)) @ R_current.T

            V_O = obstacle.generateSquare()

            self.opti.subject_to(- (1/4) * ca.dot(self.xi_[:, i], self.xi_[:, i]) - ca.dot(self.xi_[:, i], c_R) - self.mu_r_[i] - self.nu_[i] - obstacle.safetyMargin**2 >= 0)
            # Constraint from the vertex representation of the polytope obstacle:
            self.opti.subject_to(V_O.T @ self.xi_[:, i] + self.mu_r_[i] >= 0)
            # Enforce the norm condition: ||ξ||² ≥ 4 Δ_min².
            self.opti.subject_to(ca.dot(self.xi_[:, i], self.xi_[:, i]) >= 4 * obstacle.safetyMargin**2)
            # Dual constraint linking ν and the ellipsoid shape:
            self.opti.subject_to(self.nu_[i]**2 >= ca.dot(self.xi_[:, i], ca.mtimes(P_R_inv, self.xi_[:, i])))
            self.opti.subject_to(self.nu_[i] >= 0)

            self.opti.subject_to(ca.sumsqr(self.startPos[:, i] - self.x[0:3, i]) <= obstacle.minDistance**2)

        # State and actuation constraints
        for i in range(self.N):
            #self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))
            self.opti.subject_to(self.opti.bounded(self.stateMinValues, self.x[:, i], self.stateMaxValues))

        # Define the cost function
        cost = 0
        cost += 10 * self.dt**2
        # Penalize variation in attitude 
        for i in range(self.N - 1):
            cost += (1 - ca.dot(self.x[6:10, i], self.x[6:10, i + 1])**2)  

        for i in range(self.N):
            cost += self.u[:, i].T @ self.u[:, i]

        self.opti.minimize(cost)

        self.opti.solver("ipopt", {}, {
            "print_level": 5,
            "max_iter": 1000,
            "warm_start_init_point": "yes",        # Use initial guess
            "mu_strategy": "adaptive",
            "hessian_approximation": "limited-memory",
        })

        #self.opti.solver(
        #    "sqpmethod",
        #    {
        #        "max_iter": 1,
        #        # "max_inner_iter" : 1,
        #        "qpsol_options": {
        #            "nWSR": 10,
        #            "error_on_fail": False,
        #        },
        #    },
        #    {},
        #)

    def optimize(
        self,
        initial_path: List[OptimizationState],
        obstacles: List[Obstacle],
        dt: float = 1,
        prev_u: np.ndarray = None,
    ):
        if self.num_obstacles != len(obstacles):
            raise ValueError(
                "Number of obstacles has changed, please call the setup optimization function again with the correct number of obstacles"
            )
        self.opti.set_initial(self.dt, dt)

        for i in range(self.N):
            self.opti.set_initial(self.x[0:3, i], initial_path[i].x)
            self.opti.set_initial(self.x[3:6, i], initial_path[i].v)
            self.opti.set_initial(self.x[6:10, i], initial_path[i].q)
            self.opti.set_initial(self.x[10:13, i], initial_path[i].w)

            self.opti.set_value(self.startPos[:, i], initial_path[i].x)


        #for i, obstacle in enumerate(obstacles):
        #    self.opti.set_value(self.closestPointObstacle[:, i], obstacle.closestPointObstacle)
        #    self.opti.set_value(self.translation[:, i], obstacle.translation)
        #    self.opti.set_value(self.normal[:, i], obstacle.normal)
        #    self.opti.set_value(self.distance[i], obstacle.minDistance)

        if prev_u is not None:
            self.opti.set_initial(self.u, prev_u)

        sol = self.opti.solve_limited()
        return sol

    def getSolution(self, sol: ca.OptiSol) -> List[OptimizationState]:

        if sol is None:
            return []

        x = sol.value(self.x)
        u = sol.value(self.u)
        dt = sol.value(self.dt)

        return (
            [
                OptimizationState(
                    x[0:3, i], x[6:10, i], x[3:6, i], x[10:13, i], u[:, i], i
                )
                for i in range(x.shape[1])
            ],
            u,
            dt,
        )

    def visualize_trajectory(self, initial_path, optimized_path, voxel_mesh):
        plotter = pv.Plotter()
        plotter.add_mesh(voxel_mesh, color="red", opacity=0.4)

        for s in initial_path:
            mesh = pv.ParametricEllipsoid(0.24, 0.24, 0.10)
            T = np.eye(4)
            T[:3, :3] = R.from_quat(s.q).as_matrix()
            T[:3, 3] = s.x
            mesh.transform(T)
            plotter.add_mesh(mesh, color="green", opacity=0.4)
        for s in optimized_path:
            mesh = pv.ParametricEllipsoid(0.24, 0.24, 0.10)
            T = np.eye(4)
            T[:3, :3] = R.from_quat(s.q).as_matrix()
            T[:3, 3] = s.x
            mesh.transform(T)
            plotter.add_mesh(mesh, color="blue", opacity=0.4)
        plotter.add_axes()
        #plotter.export_html("OptimizedTrajectory.html")
        return plotter
