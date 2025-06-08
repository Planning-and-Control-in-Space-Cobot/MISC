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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, TypeVar
_T = TypeVar('_T', bound='Model')
from typing_extensions import override


class Model: 
    def __init__(self, name :  str, CollisionGeometry : fcl.CollisionGeometry = None, mesh : o3d.geometry.TriangleMesh = None):
        '''
        Base class for a model in the optimization problem, this class will be used to represent the robot dynamics, shape and collision geometry

        The init function of the model may be overloaded if necessary
        Parameters
            name (str): Name of the model
            collisionGeometry (fcl.CollisionGeometry): Collision geometry of the model, this will be used to compute the collision constraints in the optimization problem
            mesh (o3d.geometry.TriangleMesh): Mesh of the model, this will be used to visualize the model in the optimization problem
        '''
        self.collisionGeometry = None


    def getCollisionGeometry(self):
        '''
        Function to get the collision geometry of the model, '''
    

    @abstractmethod    
    def f(self, state, u, dt):
        '''
        Function to compute the next state of the Model given the current state, control inputs and time step


        Parameters
            state (np.ndarray): current state of the model
            u (np.ndarray): control inputs
            dt (float): time step
        Returns
            np.ndarray: next state of the model
        '''
        raise NotImplementedError("This method should be implemented in the subclass")    



class Robot(Model):
    @override
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

    
    @override
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
        safetyMargin : float = 0.1):
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

    def generateCube(self, ):
        """
        Generate 8 vertices of a cube in 3D space that lies in the half-space
        away from the obstacle (in the direction opposite to the normal).

        Parameters:
        - size (float): edge length of the cube

        Returns:
        - V (np.ndarray): 3x8 array where each column is a 3D vertex
        """
        size = self.minDistance + self.safetyMargin
        # Find center of cube, offset in the direction *away* from the obstacle
        offset = -self.normal * (size / 2.0)
        center = self.closestPointObstacle + offset

        # Create orthonormal basis (u, v, n)
        helper = np.array([0, 1, 0]) if not np.allclose(self.normal, [0, 1, 0]) else np.array([1, 0, 0])
        u = np.cross(self.normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u)
        v /= np.linalg.norm(v)
        n = self.normal

        # Half size in each direction
        hs = size / 2.0

        # Build 8 corners of the cube in the local frame and transform
        directions = [
            +u + v + n,  -u + v + n,  -u - v + n,  +u - v + n,
            +u + v - n,  -u + v - n,  -u - v - n,  +u - v - n,
        ]

        V = np.stack([center + hs * d for d in directions], axis=1)  # 3x8

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
        self.opti.subject_to(self.dt < 0.5)
        self.opti.subject_to(self.dt > 0)

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

            V_O = obstacle.generateCube()

            self.opti.subject_to(- (1/4) * ca.dot(self.xi_[:, i], self.xi_[:, i]) - ca.dot(self.xi_[:, i], c_R) - self.mu_r_[i] - self.nu_[i] - obstacle.safetyMargin**2 > 0 + 1e-3)
            # Constraint from the vertex representation of the polytope obstacle:
            self.opti.subject_to(V_O.T @ self.xi_[:, i] + self.mu_r_[i] > 0 + 1e-3)
            # Enforce the norm condition: ||ξ||² ≥ 4 Δ_min².
            self.opti.subject_to(ca.dot(self.xi_[:, i], self.xi_[:, i]) > 4 * obstacle.safetyMargin**2 + 1e-3)
            # Dual constraint linking ν and the ellipsoid shape:
            self.opti.subject_to(self.nu_[i]**2 > ca.dot(self.xi_[:, i], ca.mtimes(P_R_inv, self.xi_[:, i])) + 1e-3)
            self.opti.subject_to(self.nu_[i] > 0 + 1e-3)

            self.opti.subject_to(ca.sumsqr(self.startPos[:, i] - self.x[0:3, i]) <= obstacle.minDistance**2)

        # State and actuation constraints
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))
            self.opti.subject_to(self.opti.bounded(self.stateMinValues, self.x[:, i], self.stateMaxValues))

        # Define the cost function
        cost = 0
        cost += 10000 * self.dt
        # Penalize variation in attitude 
        #for i in range(self.N - 1):
        #    cost += (1 - ca.dot(self.x[6:10, i], self.x[6:10, i + 1])**2)  

        for i in range(self.N):
            cost += self.u[:, i].T @ 0.1 @ self.u[:, i]

        for i in range(1, self.N - 1):
            cost += (self.x[0:3] - self.xf[0:3]).T @ 0.1 @ (self.x[0:3] - self.xf[0:3])

        self.opti.minimize(cost)

        #self.opti.solver("ipopt", 
        #    {
        #        "print_time": False
        #    },
        #    {
        #        "print_level": 0,
        #        "max_iter": 10000,
        #        "warm_start_init_point": "yes",        # Use initial guess
        #        "linear_solver" : "ma97", 
        #        "mu_strategy": "adaptive",
        #        "hessian_approximation": "limited-memory",
        #    }
        #)

        self.opti.solver(
            "sqpmethod",
            {
                "max_iter": 1,
                # "max_inner_iter" : 1,
                "qpsol_options": {
                    "nWSR": 1,
                    "error_on_fail": False,
                },
            },
            {},
        )

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


        if prev_u is not None:
            self.opti.set_initial(self.u, prev_u)

        sol = self.opti.solve_limited()
        return sol
    def f_numpy(self, state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute next state using the same dynamics as robot.f, but in NumPy (no CasADi).
        """
        x = state[0:3]
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]

        F = self.robot.A[0:3, :] @ u
        M = self.robot.A[3:6, :] @ u

        # Position and velocity update
        x_next = x + v * dt
        R_q = R.from_quat(q).as_matrix()
        v_next = v + (dt / self.robot.m) * (R_q @ F)

        # Quaternion integration
        w_norm = np.linalg.norm(w) + 1e-3
        axis = w / w_norm
        theta = w_norm * dt
        dq = R.from_rotvec(axis * theta).as_quat()
        q_next = R.from_quat(dq) * R.from_quat(q)
        q_next = q_next.as_quat()

        # Angular velocity update
        J = self.robot.J
        w_next = w + dt * np.linalg.inv(J) @ (M - np.cross(w, J @ w))

        return np.hstack([x_next, v_next, q_next, w_next])

    def debug_constraints(self, sol: ca.OptiSol, obstacles: List[Obstacle]):
        import numpy as np

        x_val = sol.value(self.x)
        u_val = sol.value(self.u)
        dt_val = float(sol.value(self.dt))

        print("\n=== Decision Variables ===")
        print(f"dt = {dt_val:.6f}")
        for i in range(self.N):
            print(f"State {i}:")
            print(f"  pos      = {x_val[0:3, i].flatten()}")
            print(f"  vel      = {x_val[3:6, i].flatten()}")
            print(f"  quat     = {x_val[6:10, i].flatten()}")
            print(f"  ang_vel  = {x_val[10:13, i].flatten()}")
            print(f"  control  = {u_val[:, i].flatten()}")

        print("\n=== Obstacle Constraints ===")
        P_R_base = np.diag([1 / 0.24**2, 1 / 0.24**2, 1 / 0.10**2])
        for i, obs in enumerate(obstacles):
            c_R = x_val[0:3, i + 1]
            q_i = x_val[6:10, i + 1]
            R_i = R.from_quat(q_i).as_matrix()
            P_R_inv = R_i @ np.linalg.inv(P_R_base) @ R_i.T
            xi_val = sol.value(self.xi_[:, i])
            mu_r_val = float(sol.value(self.mu_r_[i]))
            nu_val = float(sol.value(self.nu_[i]))
            cube_vertices = obs.generateCube()

            print(f"\nObstacle iter {i}:")
            c1 = -0.25 * np.dot(xi_val, xi_val) - np.dot(xi_val, c_R) - mu_r_val - nu_val - obs.safetyMargin**2
            if c1 < 0:
                print(f"  Constraint 1 (support function): {float(c1):.6f} >= 0")

            c2_all = cube_vertices.T @ xi_val + mu_r_val
            for j, c2 in enumerate(c2_all):
                if c2 < 0:
                    print(f"  Constraint 2.{j+1} (vertex): {float(c2):.6f} >= 0")

            c3 = np.dot(xi_val, xi_val) - 4 * obs.safetyMargin**2
            if c3 < 0:
                print(f"  Constraint 3 (norm): {float(c3):.6f} >= 0")

            c4 = nu_val**2 - float(np.dot(xi_val, P_R_inv @ xi_val))
            if c4 < 0:  
                print(f"  Constraint 4 (dual): {float(c4):.6f} >= 0")

        print("\n=== Dynamics Constraints ===")
        for i in range(self.N - 1):
            state_i = x_val[:, i]
            u_i = u_val[:, i]
            actual = x_val[:, i + 1]
            predicted = self.f_numpy(state_i, u_i, dt_val)
            residual = actual - predicted
            print(f"  Dynamics {i}: max residual = {np.max(np.abs(residual)):.2e}")


        print("\n=== Actuation Bounds ===")
        for i in range(self.N):
            u_i = u_val[:, i]
            if not np.all((-3 <= u_i) & (u_i <= 3)):
                print(f"  u[{i}] out of bounds: {u_i}")

        print("\n=== State Bounds ===")
        for i in range(self.N):
            x_i = x_val[:, i]
            if not np.all((self.stateMinValues <= x_i) & (x_i <= self.stateMaxValues)):
                print(f"  x[{i}] out of bounds.")


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

    def visualize_trajectory(self, initial_path, optimized_path, voxel_mesh, obstacles=None):
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

        if obstacles is not None:
            for obs in obstacles:
                square = obs.generateSquare()
                faces = np.array([[4, 0, 1, 2, 3]])  # single quad face
                surf = pv.PolyData(square.T, faces)
                plotter.add_mesh(surf, color="yellow", opacity=0.3, style='wireframe')
                plotter.add_points(obs.closestPointObstacle, color="orange", point_size=10)
                plotter.add_arrows(obs.closestPointObstacle[np.newaxis, :],
                                obs.normal[np.newaxis, :], mag=0.3, color="orange")

        plotter.add_axes()
        return plotter
