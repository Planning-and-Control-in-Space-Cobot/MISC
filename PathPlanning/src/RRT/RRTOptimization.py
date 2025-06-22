import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation as R

import casadi as ca
import spatial_casadi as sc

import numba 

from typing import List

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from Environment import EnvironmentHandler

from typing import TypeVar

_T = TypeVar("_T", bound="Model")

from Robot import Robot
from Obstacle import Obstacle


class OptimizationState:
    """A class to represent the each state / step in the optimization, this class
    will also work as an interface for the optimization problem
    """
    def __init__(
        self,
        x: np.ndarray,
        q: np.ndarray,
        v: np.ndarray = np.zeros((3)),
        w: np.ndarray = np.zeros((3)),
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
        """Return the state in a flatten numpy array

        Return:
            np.ndarray: flatten state - (13,)
        """
        return np.hstack([self.x, self.v, self.q, self.w])


class RRTPathOptimization:
    def __init__(
        self,
        stateMinValues: np.ndarray,
        stateMaxValues: np.ndarray,
        env: EnvironmentHandler,
        robot: Robot,
    ):
        self.robot = robot
        self.env = env
        self.stateMinValues = stateMinValues
        self.stateMaxValues = stateMaxValues

    @staticmethod
    def checkDataValidity(
        minStateValues : np.ndarray, 
        maxStateValues : np.ndarray, 
        initialPath : np.ndarray, 
        initialActuation : np.ndarray, 
        obstaclesNormals : np.ndarray,
        obstacleClosestPoints : np.ndarray, 
        obstacleSafetyMargins : np.ndarray, 
        obstacleMaxDistances : np.ndarray, 
        numObstacles : int,
        xi : np.ndarray, 
        xf : np.ndarray 
    ):
        """Check the validity of the data passed to the optimization problem.

        To ensure the problem is working as expected, we need to make sure all
        the data is valid, otherwise the results may vary from the expected
        ones and introduce errors that are hard to debug.

        Parameters:
            minStateValues (np.ndarray): Minimum values for the state variables
            maxStateValues (np.ndarray): Maximum values for the state variables
            initialPath (np.ndarray): Initial path to optimize
            initialActuation (np.ndarray): Initial actuation values
            obstaclesNormals (np.ndarray): Normals of the obstacles
            obstacleClosestPoints (np.ndarray): Closest points of the obstacles
            obstacleSafetyMargins (np.ndarray): Safety margins for the obstacles
            obstacleMaxDistances (np.ndarray): Maximum distances to the 
                obstacles
            xi (np.ndarray): Initial state of the robot
            xf (np.ndarray): Final state of the robot
        
        Returns:
            bool: True if all data is valid, False otherwise
        """
        ## Check min state values:
        if minStateValues.shape != (13,):
            print("Invalid min state values shape, expected (13,)")
            return False
        if np.any(minStateValues[6:10] < -1):
            print(f"Invalid Quaternion in min state value")
            return False
        if np.any(np.isnan(minStateValues)):
            print("NaN values in min state values")
            return False
        if np.any(np.isinf(minStateValues)):        
            print("Inf values in min state values")
            return False
        ## Check max state values:
        if maxStateValues.shape != (13,):
            print("Invalid max state values shape, expected (13,)")
            return False
        if np.any(maxStateValues[6:10] > 1):
            print(f"Invalid Quaternion in max state value")
            return False
        if np.any(np.isnan(maxStateValues)):
            print("NaN values in max state values")
            return False
        if np.any(np.isinf(maxStateValues)):
            print("Inf values in max state values")
            return False
        
        ## Check if min state values are less than max state values
        if np.any(minStateValues > maxStateValues):
            print("Min state values are greater than max state values")
            return False

        ## Check initial Path
        if initialPath.shape[1] != 13:
            print("Invalid initial path shape, expected (13, N)")
            return False

        if np.any(np.isnan(initialPath)):
            print("NaN values in initial path")
            return False
    
        if np.any(np.isinf(initialPath)):
            print("Inf values in initial path")
            return False

        for path in initialPath:
            if not np.allclose(np.linalg.norm(path[6:10]), 1, atol = 1e-1): 
                print(f"Invalid Quaternion in initial path {path} {np.linalg.norm(path[6:10])}")
                return False
            if np.any(path < minStateValues - 1e-2) or np.any(path > maxStateValues + 1e-2):
                print(f"Invalid state in initial path {path}, out of bounds")
                print(f"Min state values: {minStateValues}, Max state values: {maxStateValues}")
                return False
        
        ## Check initial actuation
        if initialActuation.shape[0] != 6:
            print("Invalid initial actuation shape, expected (6, N)")
            return False
        if np.any(np.isnan(initialActuation)):
            print("NaN values in initial actuation")
            return False
        if np.any(np.isinf(initialActuation)):
            print("Inf values in initial actuation")
            return False

        ## Check obstacle normals
        if  np.any(np.isnan(obstaclesNormals)):
            print("NaN values in obstacles normals")
            return False
        if np.any(np.isinf(obstaclesNormals)):
            print("Inf values in obstacles normals")
            return False    
        if np.linalg.norm(obstaclesNormals, axis=1).any() != 1:
            print("Invalid Quaternion in obstacles normals")
            return False
        if obstaclesNormals.shape[1] != 3:
            print("Invalid obstacles normals shape, expected (N, 3)")
            return False
        
        ## Check obstacle closest points
        if np.any(np.isnan(obstacleClosestPoints)):
            print("NaN values in obstacles closest points")
            return False
        if np.any(np.isinf(obstacleClosestPoints)):
            print("Inf values in obstacles closest points")
            return False
        if obstacleClosestPoints.shape[1] != 3:
            print("Invalid obstacles closest points shape, expected (N, 3)")
            return False
        
        ## Check obstacle safety margins
        if np.any(np.isnan(obstacleSafetyMargins)):
            print("NaN values in obstacles safety margins")
            return False
        if np.any(np.isinf(obstacleSafetyMargins)):
            print("Inf values in obstacles safety margins")
            return False
        if np.any(obstacleSafetyMargins < 0):
            print("Negative values in obstacles safety margins")
            return False
        if obstacleSafetyMargins.size != numObstacles:
            print(f"Invalid obstacle safety margins, expect 1 per obstacle")
            return False
        
        ## Check obstacle max distances
        if np.any(np.isnan(obstacleMaxDistances)):
            print("NaN values in obstacles max distances")
            return False
        if np.any(np.isinf(obstacleMaxDistances)):
            print("Inf values in obstacles max distances")
            return False
        if np.any(obstacleMaxDistances < 0):
            print("Negative values in obstacles max distances")
            return False

        ## Check initial state
        if xi.shape != (13,):
            print("Invalid initial state shape, expected (13,)")
            return False
        if not np.allclose(np.linalg.norm(xi[6:10]), 1, atol = 1e-1):
            print(f"Invalid Quaternion in initial state")
            return False
        if np.any(np.isnan(xi)):
            print("NaN values in initial state")
            return False
        if np.any(np.isinf(xi)):
            print("Inf values in initial state")
            return False
        ## initialState must be within the bounds
        if np.any(xi < minStateValues) or np.any(xi > maxStateValues):
            print(f"Invalid initial state {xi}, out of bounds")
            return False
        
        ## Check final state
        if xf.shape != (13,):
            print("Invalid final state shape, expected (13,)")
            return False
        if not np.allclose(np.linalg.norm(xf[6:10]), 1, atol = 1e-1):
            print(f"Invalid Quaternion in final state {xf[6:10]}")
            return False
        if np.any(np.isnan(xf)):
            print("NaN values in final state")
            return False
        if np.any(np.isinf(xf)):
            print("Inf values in final state")
            return False
        ## finalState must be within the bounds
        if np.any(xf < minStateValues) or np.any(xf > maxStateValues):
            print(f"Invalid final state {xf}, out of bounds")
            return False
        return True

    def setup_optimization(
        self,
        initial_path: List[OptimizationState],
        obstacles : List[Obstacle],
        maxDistances : List[float],
        initialU : np.ndarray,
        xi: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        xf: np.ndarray = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ):
        """Setups the optimization problem, this function only needs to be called
        once, while we maintain the same number of obstacles and collisions

        Parameters
            initial_path (List[OptimizationState]): initial path to optimize
            obstacles (List[Obstacle]): List of obstacle in the environment we need to avoid
            maxDistance (List[Float]): List of the maximum distance to the obstacle in each step (the max distance is the maximum of the minimum distance)
            xi (np.ndarray): initial state of the robot
            xf (np.ndarray): final state of the robot

        Returns:
            None
        """
        ## Check data validity 
        import time 
        startTime = time.time()
        if not self.checkDataValidity(
            self.stateMinValues, 
            self.stateMaxValues, 
            np.vstack([s.get_state() for s in initial_path]), 
            initialU,
            np.array([o.normal for o in obstacles]),
            np.array([o.closestPointObstacle for o in obstacles]),
            np.array([o.safetyMargin for o in obstacles]),
            np.array(maxDistances),
            len(obstacles),
            xi,
            xf
        ):
            raise ValueError("Invalid data passed to the optimization problem")     
        print(f"Data validity check took {time.time() - startTime:.2f} seconds")



        self.opti = ca.Opti()
        self.N = len(initial_path)

        # Define the optimization variables
        self.x = self.opti.variable(
            13, self.N
        )  # [3 pos; 3 vel; 4 quat; 3 ang_vel]
        self.u = self.opti.variable(6, self.N)
    
        #self.initialPos = self.opti.parameter(3, self.N - 2)
        #self.maxDistance =  self.opti.parameter(1, self.N - 2)

        # The distance to the obstacle in each of the steps
        #self.obstacleDistance = self.opti.parameter(
        #    1, self.N
        #)  # Minimum distance to the obstacles in each step in the path (does not matter if we sample the robot or not!)

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
                self.x[:, i + 1]
                == self.robot.f(self.x[:, i], self.u[:, i], self.dt)
            )

        # Obstacle avoidance constraints - First we need to get the obstacle of the initial path
        for i in range(1, self.N - 1):
            # Get the decision variables for the current step
            pos = self.x[0:3, i]
            R_q = sc.Rotation.from_quat(self.x[6:10, i])

            # Gets the obstacles in the environment (from initial path not decision variables)
            maxDistance = maxDistances[i]
            _obstacles = [o for o in obstacles if o.iteration == i]
            for obstacle in _obstacles:
                # half plane constraints:
                for v in self.robot.getVertices():
                    # How to know if the signal for the half plane is positive or negative?
                    #
                    self.opti.subject_to(
                        obstacle.normal.reshape((1, 3))
                        @ (R_q.as_matrix() @ v + pos)
                        > obstacle.normal.reshape((1, 3)) @ obstacle.closestPointObstacle.reshape((3, 1)) + obstacle.safetyMargin
                    )

            self.opti.subject_to(
                ca.sumsqr(self.x[0:3, i] - initial_path[i].x) < maxDistance**2
            )

        # State and actuation constraints
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(-3, self.u[:, i], 3))
            self.opti.subject_to(
                self.opti.bounded(
                    self.stateMinValues, self.x[:, i], self.stateMaxValues
                )
            )
            self.opti.subject_to(ca.sumsqr(self.x[6:10, i]) == 1)

        #self.opti.subject_to(

        # Define the cost function
        self.cost = 0
        self.cost += 10000 * self.dt
        for i in range(self.N):
            self.cost += self.u[:, i].T @ 0.1 @ self.u[:, i]

        for i in range(1, self.N - 1):
            self.cost += (self.x[0:3, i] - self.xf[0:3]).T @ (
                self.x[0:3, i] - self.xf[0:3]
            )

        self.opti.minimize(self.cost)

        self.opti.solver(
            "ipopt",
            {
                "print_time": False
            },
            {
                "print_level": 0,
                "max_iter": 100,
                "warm_start_init_point": "yes",  # Use initial guess
                "linear_solver": "ma97",
                "mu_strategy": "adaptive",
                "hessian_approximation": "limited-memory",
            },
        )

        #self.opti.solver(
        #   "sqpmethod",
        #   {
        #       "max_iter": 1,
        #       # "max_inner_iter" : 1,
        #       "qpsol_options": {
        #           "nWSR": 1000,
        #           "error_on_fail": False,
        #       },
        #   },
        #   {},
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

        if prev_u is not None:
            self.opti.set_initial(self.u, prev_u)

        sol = self.opti.solve_limited()
        return sol

    def getSolution(self, sol: ca.OptiSol):
        if sol is None:
            return []

        x = sol.value(self.x)
        u = sol.value(self.u)
        dt = sol.value(self.dt)
        cost = sol.value(self.cost)

        return (
            [
                OptimizationState(
                    x[0:3, i], x[6:10, i], x[3:6, i], x[10:13, i], u[:, i], i
                )
                for i in range(x.shape[1])
            ],
            u,
            dt,
            cost
        )

    def visualize_trajectory(
        self, initial_path, optimized_path, voxel_mesh, obstacles=None, steps : List[int] = [] 
    ):
        if steps == []:
            steps = range(len(initial_path))
        
        plotter = pv.Plotter()
        plotter.add_mesh(voxel_mesh, color="red", opacity=0.1)

        for i, s in enumerate(initial_path):
            if i not in steps:
                continue
            mesh = self.robot.getPVMesh(s.x, R.from_quat(s.q))
            plotter.add_mesh(
                mesh, color="green", opacity=0.5
            )




        for i, s in enumerate(optimized_path):
            if i not in steps:
                continue
            mesh = self.robot.getPVMesh(s.x, R.from_quat(s.q))
            plotter.add_mesh(
                mesh, color="blue", opacity=0.5, show_edges=True
            )



        #if obstacles is not None:
        #    for obs in obstacles:
        #        cube = obs.generateCube()
        #        faces = np.array(
        #            [
        #                [4, 0, 1, 2, 3],
        #                [4, 4, 5, 6, 7],
        #                [4, 0, 1, 5, 4],
        #                [4, 2, 3, 7, 6],
        #                [4, 0, 3, 7, 4],
        #                [4, 1, 2, 6, 5],
        #            ]
        #        )
        #        surf = pv.PolyData(cube.T, faces=faces)
        #        plotter.add_mesh(
        #            surf, color="yellow", opacity=0.3, style="wireframe"
        #        )
        #        plotter.add_points(
        #            obs.closestPointObstacle, color="orange", point_size=10
        #        )
        #        plotter.add_arrows(
        #            obs.closestPointObstacle[np.newaxis, :],
        #            obs.normal[np.newaxis, :],
        #            mag=0.3,
        #            color="orange",
        #        )

        plotter.add_axes()
        return plotter