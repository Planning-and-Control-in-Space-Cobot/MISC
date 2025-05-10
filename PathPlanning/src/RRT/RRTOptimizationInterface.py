import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
import scipy

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from RRT import CollisionEnvironment3D as RRTEnv
from RRT import RRTState
from RRTOptimization import (
    RRTPathOptimization as RRTOpt,
    Robot,
    Obstacle,
    Collision,
    OptimizationState,
)

from typing import List, Tuple


class RRTOptimizationInterface:
    def __init__(
        self,
        env: RRTEnv,
        robot: Robot,
        initial_path: List[RRTState],
        stateMinValues: np.ndarray,
        stateMaxValues: np.ndarray,
        dt: float = 5,
        max_iter: int = 10,
    ):
        self.env = env
        self.robot = robot
        self.initialPath = [self._RRTStateToOptimizationState(s) for s in initial_path]
        self.stateMinValues = stateMinValues
        self.stateMaxValues = stateMaxValues
        self.dt = dt
        self.max_iter = max_iter

    def _isPathValid(self, Path: List[OptimizationState]) -> bool:
        """
        Check if the path is valid by checking for collisions with the environment.
        """
        r_ = self.env.buildEllipsoid()
        for state in Path:
            if self.env.numCollisions(r_, state.x, state.q) > 0:
                return False
        return True

    def _getColisions(self, Path: List[OptimizationState]) -> List[Tuple[int, np.ndarray]]:
        """
        Get the list of collisions in the path considering the shape of the
        robot with attitude and position of the robot and the environment.
        """
        colisions = []
        r_ = self.env.buildEllipsoid()
        for i, state in enumerate(Path):
            if self.env.numCollisions(r_, state.x, state.q) > 0:
                colision_point = self.env.getCollisionPoints(r_, state.x, state.q)
                colisions.append((i, colision_point))
        return colisions

    def _fromColisionListToObstacleList(
        self,
        colisions: List[Tuple[int, np.ndarray]],
        previousPath: List[OptimizationState],
    ) -> List[Collision]:
        """
        Convert the list of collisions to a list of obstacles for the
        optimization problem.

        Parameters:
            colisions (List[Tuple[int, np.ndarray]]): List of collisions with
            the index of the state and the point of collision. These colisions
            happen for the optimized path and not for the previous path.

            previousPath (List[OptimizationState]): The initial path that was
            passed to this optimization problem.
        Returns:
            obstacles (List[Obstacle]): The list of obstacles that will be used
            in the optimization.
        """
        obstacles = []
        point = self.env.buildSinglePoint()
        ellipsoid_robot = self.env.buildEllipsoid()

        for path_iter, colision_point in colisions:
            # The colision happens for the optimization problem j at index i We
            # will redo optimization problem j adding a new constraint to the
            # colision point
            # New colisions constraint
            collision_point = colision_point.reshape((3,))
            pos = previousPath[path_iter].x
            q = previousPath[path_iter].q

            normal = self.env.getNormalPlane(
                ellipsoid_robot, pos, q, point, colision_point.T
            )

            obstacles.append(Collision(normal, colision_point, path_iter))
        return obstacles

    def _constructObstacleList(self, path: List[OptimizationState]) -> List[Obstacle]:
        """
        Construct the list of obstacles that will be used in the optimization as
        the obstacle avoidance constraint. Each step in the optimization problem
        need to have at least one constraint for obstacle avoidance, even if
        currently it is not in collision with any obstacle.

        Parameters:
            path (List[OptimizationState]): The path that was passed to this
            optimization problem.

        Returns:
            obstacles (List[Obstacle]): The list of obstacles that will be used
            in the optimization.
        """

        obstacles = []
        ellipsoid_robot = self.env.buildEllipsoid()

        for i, state in enumerate(path):
            pos = state.x
            quat = state.q

            normal = self.env.getNormalPlane(ellipsoid_robot, pos, quat)
            distance, closestPointRobot, closestPointEnv = self.env.getClosestPoints(ellipsoid_robot, pos, quat)
            #print(f"i - {i} distance - {distance} closestPointRobot - {closestPointRobot} closestPointEnv - {closestPointEnv}")

            obstacles.append(Obstacle(normal, closestPointEnv, distance, i))
        return obstacles

    def _RRTStateToOptimizationState(self, state: RRTState) -> OptimizationState:
        """
        Convert the RRT state to optimization state.

        This is used since the inital path we receive originates from an RRT and
        then needs to be converted to a optimization state so that it can be
        used in the optimization phase

        This function assumes the velocity and angular velocity are zero in the
        state, since our RRT does not consider dynamics, this is the only case
        we can ensure the dynamic constraint are satisfied.

        Parameters:
            state (RRTState): The state that was passed to this optimization
            problem.

        Returns:
            state (OptimizationState): The state that was passed to this
            optimization problem.
        """
        return OptimizationState(
            state.x,  # pos
            state.q,  # quat
            np.zeros((3,)),  # vel
            np.zeros((3,)),  # ang vel
            np.zeros((6,)),  # u
            state.i,  # index
        )

    def optimizePath(self) -> List[OptimizationState]:
        """
        Optimize the path using the RRT path optimization algorithm.

        This function already includes all the necessary logic to handle
        colisions with obstacles, meaning you only need to call the function and
        receive the optimized path
        """
        obstacles = self._constructObstacleList(self.initialPath)

        opt = RRTOpt(self.stateMinValues, self.stateMaxValues, self.robot)
        opt.setup_optimization(
            self.initialPath,
            self._constructObstacleList(self.initialPath),
            [],
            self.initialPath[0].get_state(),
            self.initialPath[-1].get_state(),
        )

        N = len(self.initialPath)

        previousPath = [self._RRTStateToOptimizationState(p) for p in self.initialPath]
        previousObstacles = obstacles
        previousCollisions = []
        previousDt = self.dt
        previousU = np.zeros((6, N))

        lastSolution = (previousPath, previousObstacles, previousDt, previousU)

        ellipsoid_robot = self.env.buildEllipsoid()
        previouslyFailed = False

        """
        for i, p in enumerate(self.initialPath):
            if self.env.numCollisions(ellipsoid_robot, p.x, p.q) > 0:
                print("Collision detected {i} with point {p.x} {p.q}")
        """

        for i in range(self.max_iter):
            print(f"Iteration {i} of {self.max_iter}", end="\n\n\n")
            sol = opt.optimize(previousPath, previousObstacles, previousCollisions, previousDt, previousU)


            if sol is None:
                raise ValueError(
                    "No solution found, something went wrong with the optimization stage"
                )

            optimizedPath, optimizedU, optimizedDt = opt.getSolution(sol)
            print(f"OptimizedDt - {optimizedDt}")

            if self._isPathValid(optimizedPath):
                print(f"Path is valid")
                newObstacles = self._constructObstacleList(
                    optimizedPath
                )  # No need to append colisions to the state, since the trajectory is valid => No colisions
                previousCollisions = [] 
                newU = optimizedU
                newDt = optimizedDt
                lastSolution = (optimizedPath, newObstacles, newDt, newU)
                if previouslyFailed:
                    opt.setup_optimization(
                        optimizedPath,
                        newObstacles,
                        [],
                        self.initialPath[0].get_state(),
                        self.initialPath[-1].get_state(),
                    )
                previouslyFailed = False

            else:
                previouslyFailed = True
                newCollisions_ = self._fromColisionListToObstacleList(
                    self._getColisions(optimizedPath), previousPath
                )
                
                previousCollisions.extend(newCollisions_)
                print(f"Path is not valid -  {len(previousCollisions)} - {len(newCollisions_)} ")
                for i in range(len(newCollisions_)):
                    print(f" - {newCollisions_[i].iteration} - {newCollisions_[i].closest_point_surface}")
                newObstacles = previousObstacles
                newU = previousU
                newDt = previousDt
                optimizedPath = previousPath
                opt.setup_optimization(
                    optimizedPath,
                    newObstacles,
                    previousCollisions,
                    self.initialPath[0].get_state(),
                    self.initialPath[-1].get_state(),
                )

            previousPath = optimizedPath
            previousObstacles = newObstacles
            previousDt = newDt
            previousU = newU
        
        return lastSolution
    
    def visualizeTrajectory(self,  plotter : pv.Plotter, lastSolution ):
        """
        Visualize the trajectory in the environment.

        Parameters:
            lastSolution (List[OptimizationState]): The optimized path that was
            passed to this optimization problem.
            plotter (pv.Plotter): The plotter that will be used to visualize the
            trajectory.
        """
        optimizedPath, obstacles, dt, u = lastSolution
        r_ = pv.ParametricEllipsoid(
            self.robot.ellipsoid_radius[0],
            self.robot.ellipsoid_radius[1],
            self.robot.ellipsoid_radius[2],
        )
        plotter = pv.Plotter()
        print(self.env.voxel_mesh)
        plotter.add_mesh(
            self.env.voxel_mesh, color="white",  label="Voxel Grid"

        )
        print(f"raddi : {self.robot.ellipsoid_radius}")

        for state, initial_state in zip(optimizedPath, self.initialPath):
            r__ = r_.copy()
            pos = state.x
            q = state.q
            tranform = np.eye(4)
            tranform[:3, :3] = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()
            tranform[:3, 3] = pos
            r__.transform(tranform)

            ini_pos = initial_state.x
            ini_q = initial_state.q
            ini_tranform = np.eye(4)
            ini_tranform[:3, :3] = scipy.spatial.transform.Rotation.from_quat(ini_q).as_matrix()
            ini_tranform[:3, 3] = ini_pos
            r___ = r_.copy()
            r___.transform(ini_tranform)
            plotter.add_mesh(r___, color="blue",  label="Optimized Path") 
            plotter.add_mesh(r__, color="red",  label="Robot")
        
        plotter.reset_camera()

        plotter.show()

        
        return plotter
    