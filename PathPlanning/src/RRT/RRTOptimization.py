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


class Robot:
    def __init__(
        self, J: np.ndarray, A: np.ndarray, m: float, ellipsoid_radius: np.ndarray
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
            or np.shape(ellipsoid_radius) != (3,)
            or m <= 0
        ):
            raise ValueError(
                "J must be a 3x3 matrix, A must be a 6x6 matrix, ellipsoid_radius must be a 3x1 vector and m must be a positive scalar"
            )

        self.J = J
        self.A = A
        self.m = m
        self.ellipsoid_radius = ellipsoid_radius

    def getEllipsoidMatrix(self):
        """
        Return the ellipsoidal matrix of the robot

        This is a 3x3 matrix that is
        |1/a^2   0    0   |
        |0    1/b^2   0   |
        |0       0  1/c^2 |

        Returns
            np.ndarray: ellipsoidal matrix of the robot
        """
        return np.diag(1 / (self.ellipsoid_radius**2))

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
        normal: np.ndarray,
        closest_point_surface: np.ndarray,
        distance: float,
        iteration: int,
    ):
        """
        Obstacle class to represent the obstacle in the optimization problem

        Parameters
            normal (np.ndarray): normal vector between the closest point on the
            surface of the robot and the surface of the obstacle
            closest_point_surface (np.ndarray): closest point on the surface of
            the obstacle
            distance (float): distance between the obstacle and the robot
            iteration (int): iteration in the optimization problem where
            the obstacle is considered
        """
        self.normal = normal
        self.distance = distance
        self.closest_point_surface = closest_point_surface
        self.iteration = iteration


class Collision:
    def __init__(self, closest_point_surface, normal, iteration):
        """
        Collision class to represent the collision between the robot and the
        obstacle after an optimization step, this is then used to backtrack the
        optimization to the previous iteration considering not only the
        obstacles by also the collisions

        Parameters
            closest_point_surface (np.ndarray): closest point on the surface of
            the robot
            normal (np.ndarray): normal vector between the closest point on the
            surface of the robot and the surface of the obstacle
        """
        self.closest_point_surface = closest_point_surface
        self.normal = normal
        self.iteration = iteration


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
        robot=Robot(J=np.eye(3), A=np.eye(6), m=1, ellipsoid_radius=np.ones((3,))),
    ):
        self.robot = robot
        self.stateMinValues = stateMinValues
        self.stateMaxValues = stateMaxValues

    def optimize(
        self,
        initial_path: List[OptimizationState],
        obstacles: List[Obstacle],
        collision: List[Collision] = [],
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

        for i, obstacle in enumerate(obstacles):
            # print(f"obstace closest_point_surface: {obstacle.iteration} {obstacle.closest_point_surface} normal: {obstacle.normal}")
            self.opti.set_value(
                self.obstacleSurfacePoints[:, i], obstacle.closest_point_surface
            )
            self.opti.set_value(self.normals[:, i], obstacle.normal)
            self.opti.set_value(self.obstacleDistance[i], obstacle.distance)

        if collision != []:
            for i, collision in enumerate(collision):
                self.opti.set_value(
                    self.collisionSurfacePoints[:, i], collision.closest_point_surface
                )
                self.opti.set_value(self.collisionNormals[:, i], collision.normal)

        if prev_u is not None:
            self.opti.set_initial(self.u, prev_u)

        try:
            sol = self.opti.solve_limited()
            return sol

        except RuntimeError as e:
            print(
                "⚠️ Optimization failed. Dumping debug variable values per iteration:\n"
            )

            for i in range(self.N):
                try:
                    print(f"--- Iteration {i} ---")
                    print("x (position):", self.opti.debug.value(self.x[0:3, i]))
                    print("v (velocity):", self.opti.debug.value(self.x[3:6, i]))
                    print("q (quaternion):", self.opti.debug.value(self.x[6:10, i]))
                    print(
                        "R (rotation matrix):\n",
                        R.from_quat(self.opti.debug.value(self.x[6:10, i])).as_matrix(),
                    )
                    print(
                        "w (angular velocity):", self.opti.debug.value(self.x[10:13, i])
                    )
                    print("u (control):", self.opti.debug.value(self.u[:, i]))
                    print("startPos:", self.opti.debug.value(self.startPos[:, i]))

                    print("Obstacle normal:", self.opti.debug.value(self.normals[:, i]))
                    print(
                        "Obstacle surface point:",
                        self.opti.debug.value(self.obstacleSurfacePoints[:, i]),
                    )
                    print(
                        "Obstacle distance:",
                        self.opti.debug.value(self.obstacleDistance[i]),
                    )
                    print()
                except Exception as debug_exception:
                    print(f"⚠️ Failed to get debug info at step {i}: {debug_exception}")

            print("\n=== Other Variables ===")
            try:
                print("dt:", self.opti.debug.value(self.dt))
            except Exception as ex:
                print("Could not debug dt:", ex)

            if collision != []:
                for i in range(self.num_collisions):
                    try:
                        print(
                            f"Collision {i} surface point:",
                            self.opti.debug.value(self.collisionSurfacePoints[:, i]),
                        )
                        print(
                            f"Collision {i} normal:",
                            self.opti.debug.value(self.collisionNormals[:, i]),
                        )
                    except Exception as col_ex:
                        print(f"Could not debug collision {i}:", col_ex)

            raise RuntimeError(f"Solver failed: {str(e)}")

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

    def setup_optimization(
        self,
        initial_path: List[OptimizationState],
        obstacles: List[Obstacle],
        collisions: List[Collision] = [],
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
        self.num_collisions = len(collisions)
        self.N = len(initial_path)

        if self.N != self.num_obstacles:
            raise ValueError("We need to consider one obstacle per step in the horizon")

        # Define the optimization variables
        self.x = self.opti.variable(13, self.N)  # [3 pos; 3 vel; 4 quat; 3 ang_vel]
        self.u = self.opti.variable(6, self.N)

        # The start position of the robot in each of the steps
        self.startPos = self.opti.parameter(3, self.N)

        # The distance to the obstacle in each of the steps
        self.obstacleDistance = self.opti.parameter(1, self.N)

        self.dt = self.opti.variable(1)
        self.opti.subject_to(self.dt > 0.0)

        # Optimization parameters
        self.xf = self.opti.parameter(13)
        self.xi = self.opti.parameter(13)
        self.opti.set_value(self.xi, xi)
        self.opti.set_value(self.xf, xf)
        self.opti.subject_to(self.x[:, -1] == self.xf)  # Initial State
        self.opti.subject_to(self.x[:, 0] == self.xi)  # Final State

        self.obstacleSurfacePoints = self.opti.parameter(3, self.num_obstacles)
        self.normals = self.opti.parameter(3, self.num_obstacles)

        # Dynamic constraints
        for i in range(self.N - 1):
            # x[k+1] = f(x[k], u[k], dt)
            self.opti.subject_to(
                self.x[:, i + 1] == self.robot.f(self.x[:, i], self.u[:, i], self.dt)
            )

        # Obstacle avoidance constraints
        for i, obstacle in enumerate(obstacles):
            iter = obstacle.iteration
            R = sc.Rotation.from_quat(self.x[6:10, iter])
            normal = self.normals[:, iter]
            closest_point_surface = self.obstacleSurfacePoints[:, iter]

            supp = ca.sqrt(
                normal.T
                @ R.as_matrix()
                @ self.robot.getEllipsoidMatrix()
                @ R.as_matrix().T
                @ normal
            )

            center_proj = (
                normal.T 
                @ self.x[0:3, iter]
            )

            beta = normal.T @ closest_point_surface

            d_i = center_proj - beta - supp
            self.opti.subject_to(d_i > 0 + 1e-1)  # Zero + safety margin

            # trust region
            #self.opti.subject_to(
            #    ca.sumsqr(self.x[0:3, iter] - self.startPos[:, iter])
            #    < self.obstacleDistance[iter] ** 2 / 20
            #)

        # Collision avoidance constraints - only when we already had a collision
        if collisions != []:
            self.collisionSurfacePoints = self.opti.parameter(3, self.num_collisions)
            self.collisionNormals = self.opti.parameter(3, self.num_collisions)
            for i, collision in enumerate(collisions):
                iter = collision.iteration
                R = sc.Rotation.from_quat(self.x[6:10, iter])
                normal = self.collisionNormals[:, i]
                closest_point_surface = self.collisionSurfacePoints[:, i]

                supp = ca.sqrt(
                    normal.T
                    @ R.as_matrix()
                    @ self.robot.getEllipsoidMatrix()
                    @ R.as_matrix().T
                    @ normal
                )
                center_proj = (
                    normal.T @ self.x[0:3, iter]
                )

                beta = normal.T @ closest_point_surface
                d_i = center_proj - beta - supp
                self.opti.subject_to(d_i > 0 + 1e-1)

        # State and actuation constraints
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))

            # State boundaries
            # Pos
            self.opti.subject_to(
                self.opti.bounded(
                    self.stateMinValues[0:3], self.x[0:3, i], self.stateMaxValues[0:3]
                )
            )
            # Vel
            self.opti.subject_to(
                self.opti.bounded(
                    self.stateMinValues[3:6], self.x[3:6, i], self.stateMaxValues[3:6]
                )
            )
            # Ang Vel
            self.opti.subject_to(
                self.opti.bounded(
                    self.stateMinValues[10:13],
                    self.x[10:13, i],
                    self.stateMaxValues[10:13],
                )
            )

        # Define the cost function
        cost = 0
        cost += self.dt

        for i in range(self.N):
            pass
            # cost += self.u[:, i].T @ self.u[:, i]

        self.opti.minimize(cost)

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

    def visualize_trajectory(self, initial_path, optimized_path, voxel_mesh):
        plotter = pv.plotter()
        plotter.add_mesh(voxel_mesh, color="red", opacity=0.4)

        for s in initial_path:
            mesh = pv.parametricellipsoid(*self.robot.ellipsoid_radius)
            t = np.eye(4)
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
