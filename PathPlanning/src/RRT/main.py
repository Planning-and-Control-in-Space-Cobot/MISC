import numpy as np
import scipy as sp
import pyvista as pv
import argparse
import open3d as o3d

import time
import os 
import sys

from Environment import EnvironmentHandler
from RRT import RRTPlanner3D, RRTState
import rrtcxx

def main():
    parser = argparse.ArgumentParser(description="RRT Path Planning")
    parser.add_argument("--env", type=str, default="default_env", help="Environment name")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of RRT iterations")

    args = parser.parse_args()
    env_name = args.env
    iterations = args.iterations

    env = EnvironmentHandler
    if not os.path.exists(os.path.join(os.path.dirname(__file__), env_name)):
        raise FileNotFoundError(f"Environment '{env_name}' does not exist.")

    pcd = o3d.io.read_point_cloud(os.path.join(os.path.dirname(__file__), env_name))
    env = EnvironmentHandler(pcd)

    robot = env.buildBox()
    robotMesh = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))
    start = RRTState(np.array([2.75, 2, 1]), np.array([0, 0.707, 0, 0.707]))
    goal = RRTState(np.array([2.75, 2, 7]), np.array([0, 0.707, 0, 0.707]))
    minPos = np.array([1.5, 1, 0])
    maxPos = np.array([4.0, 6, 10])
    ## Now do same path with rrtcxx and compare time taken

    distance, pt1, pt2, normal = env.distance(robot, start.x, sp.spatial.transform.Rotation.from_quat(start.q))
    print(f"Distance: {distance}, Point1: {pt1}, Point2: {pt2}, Normal: {normal}")
    distance, pt1, pt2, normal = env.distance(robot, goal.x, sp.spatial.transform.Rotation.from_quat(goal.q))
    print(f"Distance: {distance}, Point1: {pt1}, Point2: {pt2}, Normal: {normal}")

    start = rrtcxx.State(np.array([0.5, 3.0, 1.0]), np.array([0, 0, 0, 1]))
    goal = rrtcxx.State(np.array([0.5, 5.0, 6.0]), np.array([0, 0, 0, 1]))
    boundsX = np.array([0.0, 3.0])
    boundsY = np.array([3.0, 5.0])
    boundsZ = np.array([0.0, 7.0])
    
    startTime = time.time()
    payloadTranslation = np.array([0.0, 0.0, 0.0])
    payloadSize = np.array([0.45, 0.45, 0.12])
    usePayload = False
    planner = rrtcxx.RRTPlanner3D(env.triangleVertex, env.triangleIndex, payloadTranslation, payloadSize, usePayload, 100000, 0.2, 0.1, 0.0, 3.0,
                                   3.0, 6.0, 0.0, 7.0)
    path = planner.plan(start, goal)
    endTime = time.time()
    print(f"RRTCXX Planning time: {endTime - startTime:.2f} seconds")
    pv_ = pv.Plotter()
    pv_.add_mesh(env.voxel_mesh, color='white', show_edges=True, opacity=0.5)
    
    
    
    if path == []:
        return
    for p in path:
        x = p.position
        q = p.q

        T = np.eye(4)
        T[:3, :3] = sp.spatial.transform.Rotation.from_quat(q, scalar_first=True).as_matrix()
        T[:3, 3] = x
        cube = robotMesh.copy()
        cube.transform(T)
        pv_.add_mesh(cube, color='blue', show_edges=True)
        

    pv_.show()


    positions = np.array([p.position for p in path])
    orientations = np.array([p.q for p in path])

    np.savez(os.path.join(os.path.dirname(__file__), "rrt_path.npz"), positions=positions, orientations=orientations)






if __name__ == "__main__":
    main()