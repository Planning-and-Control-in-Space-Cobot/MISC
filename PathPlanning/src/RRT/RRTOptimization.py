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
        self, J: np.ndarray, A: np.ndarray, m: float, fcl_obj : fcl.CollisionGeometry, surfacePoints : np.ndarray 
    ):
        """
        Robot class to represent the robot in the optimization problem

        Parameters
            J (np.ndarray): inertia matrix of the robot
            A (np.ndarray): actuation matrix of the robot
            m (float): mass of the robot represents the robot
            fcl_obj (fcl.CollisionGeometry): fcl object of the robot
            surfacePoints (np.ndarray): A 3xN matrix that represents the translation from the center of the robot to the multiple surface points that will be used to represent the robot in the obstacle avoidance constraints of the optimization problem. NOTE: This translations are represent in the robot frame, so when used in the optimization problem they need to be rotated according to the robot orientation, otherwise they will not be correct.
       
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
        self.surfacePoints = surfacePoints

    def getNumSurfacePoints(self):
        '''
        Returns the number of surface points used to represent the robot in the optimization problem
    
        Returns
            int: number of surface points
        '''
        return self.surfacePoints.shape[0]

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
        robotSurfacePointIndex : int, 
        normal : np.ndarray,
        robotSurfacePoint : np.ndarray,
        closestPointObstacle: np.ndarray, 
        safetyMargin: float = 1e-1
    ):
        '''
        This class represents an obstacle in the optimization problem

        Parameters:
            robotSurfacePointIndex (int): Index to know the surface point of the robot that is being considered for this obstacle
            normal (np.ndarray) : Normal of the obstacle plane
            robotSurfacePoint (np.ndarray): The surface point of the robot that is b
            closestPointObstacle (np.ndarray): Closest point on the obstacle
            safetyMargin (float): Safety margin for the obstacle
        '''
        self.robotSurfacePointIndex = robotSurfacePointIndex
        self.normal = normal
        self.robotSurfacePoint = robotSurfacePoint
        self.closestPointObstacle = closestPointObstacle
        self.safetyMargin = safetyMargin
        self.distance = np.linalg.norm(self.closestPointObstacle - self.robotSurfacePoint)

class TimeStepObstacleConstraints:
    def __init__(self, 
                 n : int, 
                 numSurfacePoints : int, 
                 obstacles : List[Obstacle], 
                ):
        '''
        This class represent all the obstacles in a single time step of the optimization problem. We will have 1 obstacle per surface point being considered in the optimization problem.

        Parameters:
            n (int): The time step in the optimization problem
            numSurfacePoints (int): The number of surface points being considered in the optimization problem
            Obstacles (List[Obstacle]): List of obstacles in this time step
        '''
        self.n = n
        self.numSurfacePoints = numSurfacePoints
        self.obstacles = obstacles
        self.minObstacleDistance = min([obstacle.distance for obstacle in self.obstacles])
    
    def getMinDistance(self) -> float:
        '''
        Returns the minimum distance to the obstacles in this time step
        
        Returns
            float: minimum distance to the obstacles
        '''
        return self.minObstacleDistance

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
        obstacles: List[TimeStepObstacleConstraints],
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

        self.N = len(initial_path)
        # Check if we have N - 2 sets of obstacle avoidance constraints
        # N - 2 since we dont want to move the first and the last state, and so obstacle avoidance is not necessary
        if self.N != len(obstacles) + 2:
            raise ValueError( 
            "There should be N - 2 sets of obstacle avoidance constraints, where N is the number of states in the optimization problem, we should not move the first and last state.")

        # Define the optimization variables
        self.x = self.opti.variable(13, self.N)  # [3 pos; 3 vel; 4 quat; 3 ang_vel]
        self.u = self.opti.variable(6, self.N)

        # The start position of the robot in each of the steps
        self.startPos = self.opti.parameter(3, self.N) # Used to compute the trust region constraint of the optimization problem

        self.dt = self.opti.variable(1)
        self.opti.subject_to(self.dt > 0)
        self.opti.subject_to(self.dt < 0.2)

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

        self.numSurfacePoints = self.robot.getNumSurfacePoints()

        self.normal = self.opti.parameter(3, self.numSurfacePoints * (self.N - 2))
        self.closestPointObstacle = self.opti.parameter(3, self.numSurfacePoints * (self.N - 2))
        self.translation = self.opti.parameter(3, self.numSurfacePoints * (self.N - 2))
        self.minDistance = self.opti.parameter(1, self.N - 2)
        self.nObstacles = self.numSurfacePoints * (self.N  - 2)

        for n, obstacleSet in enumerate(obstacles):
            for i, obstacle in enumerate(obstacleSet.obstacles):
                # Get the index of the obstacle in the optimization problem 
                iter = n * self.numSurfacePoints + i
                normal = self.normal[:, iter]
                closestPointObstacle = self.closestPointObstacle[:, iter]
                translation = self.translation[:, iter]
                R = sc.Rotation.from_quat(self.x[6:10, n+1])

                lhs = normal.T @ (self.x[0:3, n+1] + R.as_matrix() @ translation)
                rhs = normal.T @ closestPointObstacle + obstacle.safetyMargin
                self.opti.subject_to(lhs <= rhs)

            self.opti.subject_to(
                ca.sumsqr(self.startPos[:, n + 1]  - self.x[0:3, n+1]) <= self.minDistance[n] ** 2 
            )

        # State and actuation constraints
        for i in range(self.N):
            #self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))
            self.opti.subject_to(self.opti.bounded(self.stateMinValues, self.x[:, i], self.stateMaxValues))

        cost = 0
        # Define the cost function
        for i in range(self.N - 1):
            cost += (1 - ca.dot(self.x[6:10, i], self.x[6:10, i + 1])**2)  

        for i in range(self.N):
            cost += self.u[:, i].T @ self.u[:, i]

        self.opti.minimize(cost)

        self.opti.solver("ipopt", {}, {
            "print_level": 5,
            "max_iter": 200,
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
        obstacles: List[TimeStepObstacleConstraints],
        dt: float = 1,
        prev_u: np.ndarray = None,
    ):
        if self.nObstacles != len(obstacles) * self.robot.getNumSurfacePoints():
            raise ValueError(
                "The number of obstacles in the optimization problem does not match the number os obstacles that was used during the setup of the optimization problem, will not be able to solve"
            ) 

        for n, obstacleSet in enumerate(obstacles):
            for i, obstacle in enumerate(obstacleSet.obstacles):
                # Get the index of the obstacle in the optimization problem
                iter = n * self.robot.getNumSurfacePoints() + i
                self.opti.set_value(self.normal[:, iter], obstacle.normal)
                self.opti.set_value(self.closestPointObstacle[:, iter], obstacle.closestPointObstacle)
                self.opti.set_value(self.translation[:, iter], obstacle.robotSurfacePoint)
            
            # Set the minimum distance to the obstacles
            self.opti.set_value(self.minDistance[n], obstacleSet.getMinDistance())
            # Set the start position of the robot in this time step
            self.opti.set_value(self.startPos[:, n + 1], initial_path[n + 1].x)

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
        plotter.add_mesh(voxel_mesh, color="red", opacity=1)

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
