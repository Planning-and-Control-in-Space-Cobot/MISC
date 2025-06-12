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

from Robot import Robot
from Obstacle import Obstacle

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
        self.env = env
        self.stateMinValues = stateMinValues
        self.stateMaxValues = stateMaxValues

    def setup_optimization(
        self,
        initial_path: List[OptimizationState],
        sampleRobot = False,
        stepSize = 0.30, 
        xi: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        xf: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ):
        """
        Setups the optimization problem, this function only needs to be called
        once, while we maintain the same number of obstacles and collisions

        Parameters
            initial_path (List[OptimizationState]): initial path to optimize
            sampleRobot (bool): whether to sample the robot or not, default is False
            collisions (List[Collision]): list of collisions to avoid
            stepSize (float): Max Movement from one step to another
            xi (np.ndarray): initial state of the robot
            xf (np.ndarray): final state of the robot

        Returns
            None
        """
        print(f"xi: {xi}, xf: {xf}")
        self.opti = ca.Opti()

        self.N = len(initial_path)

        # Define the optimization variables
        self.x = self.opti.variable(13, self.N)  # [3 pos; 3 vel; 4 quat; 3 ang_vel]
        self.u = self.opti.variable(6, self.N)


        # The distance to the obstacle in each of the steps
        self.obstacleDistance = self.opti.parameter(1, self.N) # Minimum distance to the obstacles in each step in the path (does not matter if we sample the robot or not!)

        self.dt = self.opti.variable(1)
        self.opti.subject_to(self.dt < 0.2)
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
    
        #self.distance = self.opti.parameter(1, self.num_obstacles)

        P_R_Base = np.diag([1 / 0.24**2, 1 / 0.24**2, 1 / 0.10**2])
        allObstacles = []
        # Obstacle avoidance constraints - First we need to get the obstacle of the initial path
        for i in range(1, self.N - 1):
            path = initial_path[i]
            obstacles = self.robot.getObstacles(self.env, path.x, R.from_quat(path.q), i, useSampling=sampleRobot)
            allObstacles.extend(obstacles)
            
            for obstacle in obstacles:
                xi = self.opti.variable(3)
                mu_r = self.opti.variable(1)
                nu = self.opti.variable(1)

                self.opti.set_initial(xi, np.array([obstacle.safetyMargin * 2, 0, 0]))
                self.opti.set_initial(mu_r, 0)
                self.opti.set_initial(nu, 0)

                c_R = self.x[0:3, i + 1]
                R_current = sc.Rotation.from_quat(self.x[6:10, i + 1]).as_matrix()
                P_R_inv = R_current @ ca.DM(np.linalg.inv(P_R_Base)) @ R_current.T

                V_O = obstacle.generateCube()

                self.opti.subject_to(- (1/4) * ca.dot(xi, xi) - ca.dot(xi, c_R) - mu_r - nu - obstacle.safetyMargin**2 > 0 + 1e-3)
                # Constraint from the vertex representation of the polytope obstacle:
                self.opti.subject_to(V_O.T @ xi + mu_r > 0 + 1e-3)
                # Enforce the norm condition: ||ξ||² ≥ 4 Δ_min².
                self.opti.subject_to(ca.dot(xi, xi) > 4 * obstacle.safetyMargin**2 + 1e-3)
                # Dual constraint linking ν and the ellipsoid shape:
                self.opti.subject_to(nu**2 > ca.dot(xi, ca.mtimes(P_R_inv, xi)) + 1e-3)
                self.opti.subject_to(nu > 0 + 1e-3)
        
        #for i in range(self.N - 1):
        #    self.opti.subject_to(
        #        ca.sumsqr(self.x[0:3, i + 1] - self.x[0:3, i]) <= stepSize ** 2
        #    )

        # State and actuation constraints
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))
            self.opti.subject_to(self.opti.bounded(self.stateMinValues, self.x[:, i], self.stateMaxValues))

        # Define the cost function
        cost = 0
        cost += 100 * self.dt
        # Penalize variation in attitude 
        #for i in range(self.N - 1):
        #    cost += (1 - ca.dot(self.x[6:10, i], self.x[6:10, i + 1])**2)  

        for i in range(self.N):
            cost += self.u[:, i].T @ 0.1 @ self.u[:, i]

        for i in range(1, self.N - 1):
            cost += (self.x[0:3, i] - self.xf[0:3]).T @ (self.x[0:3, i] - self.xf[0:3])

        self.opti.minimize(cost)

        self.opti.solver("ipopt", 
            {
         #       "print_time": False
            },
            {
        #        "print_level": 0,
                "max_iter": 1000,
                "warm_start_init_point": "yes",        # Use initial guess
                "linear_solver" : "ma97", 
                "mu_strategy": "adaptive",
                "hessian_approximation": "limited-memory",
            }
        )

        return allObstacles
        #self.opti.solver(
        #    "sqpmethod",
        #    {
        #        "max_iter": 1,
        #        # "max_inner_iter" : 1,
        #        "qpsol_options": {
        #            "nWSR": 1,
        #            "error_on_fail": False,
        #        },
        #    },
        #    {},
        #)

    def optimize(
        self,
        initial_path: List[OptimizationState],
        dt: float = 1,
        prev_u: np.ndarray = None,
    ):
        self.opti.set_initial(self.dt, dt)

        for i in range(self.N):
            self.opti.set_initial(self.x[0:3, i], initial_path[i].x)
            self.opti.set_initial(self.x[3:6, i], initial_path[i].v)
            self.opti.set_initial(self.x[6:10, i], initial_path[i].q)
            self.opti.set_initial(self.x[10:13, i], initial_path[i].w)

            print (f"i {i} - x {initial_path[i].x} - q {initial_path[i].q}")


        if prev_u is not None:
            self.opti.set_initial(self.u, prev_u)

        sol = self.opti.solve_limited()
        print(f"xi : {self.opti.value(self.x[:, 0])} vs {initial_path[0].x}")
        print(f"xf : {self.opti.value(self.x[:, -1])} vs {initial_path[-1].x}")
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
        plotter.add_mesh(voxel_mesh, color="red", opacity=0.1)

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
                cube = obs.generateCube()
                faces = np.array([[4, 0, 1, 2, 3],
                              [4, 4, 5, 6, 7],
                              [4, 0, 1, 5, 4],
                              [4, 2, 3, 7, 6],
                              [4, 0, 3, 7, 4],
                              [4, 1, 2, 6, 5]])
                surf = pv.PolyData(cube.T, faces=faces)
                plotter.add_mesh(surf, color="yellow", opacity=0.3, style='wireframe')
                plotter.add_points(obs.closestPointObstacle, color="orange", point_size=10)
                plotter.add_arrows(obs.closestPointObstacle[np.newaxis, :],
                                obs.normal[np.newaxis, :], mag=0.3, color="orange")

        plotter.add_axes()
        return plotter
