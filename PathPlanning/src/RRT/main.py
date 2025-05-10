import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from RRT import RRTPlanner3D as RRTPlanner, RRTState

from Environment import EnvironmentHandler as RRTEnv
from RRTOptimization import Robot
from RRTOptimizationInterface import RRTOptimizationInterface


def main():
    point_cloud_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "map.pcd"
    )
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    rrtEnv = RRTEnv(pcd)

    start = RRTState(np.array([0, -3, 1.0]), np.array([0, 0, 0, 1]))
    goal = RRTState(np.array([20.0, 15.0, 1.0]), np.array([0, 0, 0, 1]))
    planner = RRTPlanner(rrtEnv)
    path = planner.plan(start, goal)

    if path == []:
        return
    """
    r_ = rrtEnv.buildEllipsoid()
    for i, p in enumerate(path):
        if (numCollisions := rrtEnv.numCollisions(r_, p.x, p.q)) > 0:
            print(f"Collision at {i} with {numCollisions} collisions")
    print(f"Path is valid -  Into optimization")
    """

    A = np.load(os.path.join(os.path.dirname(__file__), "A_matrix.npy"))
    J = np.load(os.path.join(os.path.dirname(__file__), "J_matrix.npy"))
    m = np.load(os.path.join(os.path.dirname(__file__), "mass.npy"))
    ellipsoid_radius = np.load(
        os.path.join(os.path.dirname(__file__), "ellipsoidal_radius.npy")
    )

    robot = Robot(J, A, m, ellipsoid_radius)
    stateMinValues = np.array([0, 0, 0, -5, -5, -5, -1, -1, -1, -1, -2, -2, -2])
    stateMaxValues = np.array([25, 25, 2, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2])

    rrtOpt = RRTOptimizationInterface(
        rrtEnv, robot, path, stateMinValues, stateMaxValues, dt=5, max_iter=100
    )
    lastSolution = rrtOpt.optimizePath()

    #plotter = rrtEnv.visualizeMap(plotter)
    plotter = rrtOpt.visualizeTrajectory(None, lastSolution)


if __name__ == "__main__":
    main()
