import os
import sys
import time
import argparse

import pyvista as pv 
import numpy as np 
import pickle as pkl
import scipy.spatial.transform as trf
import matplotlib.pyplot as plt
import open3d as o3d

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from Environment import EnvironmentHandler
from Robot import Robot
#from RRTOptimization import OptimizationState


def main():
    parser = argparse.ArgumentParser(
        description="RRT Path Planning and Optimization"
    )
    parser.add_argument(
        "--map",
        "-m",
        type=str,
        default="map.pcd",
        help="Path to the point cloud map file",
    )
    parser.add_argument(
        "--results", 
        "-r",
        type=str,
        default="results.pkl",
        help="Path to save the optimization results",
    )

    args = parser.parse_args()
    scriptDir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(scriptDir, args.map)):
        print(f"Map file {args.map} does not exist.")
        sys.exit(1)

    if not os.path.exists(os.path.join(scriptDir, args.results)):
        print(f"Results file {args.results} does not exist.")
        sys.exit(1)

    env = EnvironmentHandler(
        o3d.io.read_point_cloud(os.path.join(scriptDir, args.map)),
    )

    robot = Robot(
        np.eye(3), 
        np.eye(6), 
        1
    ) # Create robot object with placeholder parameters, since dynamics are not 
    # used in this example, and will only be used for visualization of a 
    # trajectory that has already been computed.
     
    with open(os.path.join(scriptDir, args.results), "rb") as f:
        results = pkl.load(f)
    
    initialPath = results["initialPath"]
    realPath = results["realPath"]
    optimalPaths = results["optimalPaths"]


    pv_ = pv.Plotter()
    pv_.add_mesh(
        env.voxel_mesh, 
        color="white",
        show_edges=True,
        opacity=0.5,
    )

    for p, _, _ in realPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q) 
        
        pv_.add_mesh(
            robot.getPVMesh(x, R), 
            color="blue",
            show_edges=True,
            opacity=0.5,
        )


    numOptimalPaths = len(optimalPaths)
    print(f"Number of optimal paths: {numOptimalPaths}")

    for p in initialPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q) 
        
        pv_.add_mesh(
            robot.getPVMesh(x, R), 
            color="red",
            show_edges=True,
            opacity=0.5,
        )
    
    for p in optimalPaths[-1][0]:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        pv_.add_mesh(
            robot.getPVMesh(x, R), 
            color="green",
            show_edges=True,
            opacity=0.5,
        )


    pv_.add_text(
        "Initial Path", 
        position="upper_left", 
        color="red"
    )

    pv_.add_text(
        "Converged Optimal Path",
        position="lower_left",
        color="green"
    )

    pv_.add_text(
        "Real Path", 
        position="upper_right", 
        color="blue"
    )
    
    pv_.show_grid()
    pv_.show()


    print("Plotting Data related to distance between real path and each optimal path")
    distances = []
    for i, (optimalPath, dt, cost) in enumerate(optimalPaths):
        totalDistance = 0
        for p, _, sop in realPath:
            if sop != optimalPath:
                continue

            index = min (
                range(len(optimalPath)), 
                key = lambda j: np.linalg.norm(p.x - optimalPath[j].x)
            )
            #print(f"Optimal Path {i} with Closest Point Index {index} to real path index {realPath.index((p, _, _))}")
            totalDistance += np.linalg.norm(p.x - optimalPath[index].x)

        distances.append(totalDistance)
    
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(numOptimalPaths), 
        distances, 
        color='skyblue'
    )
    plt.xlabel('Optimal Path Index')
    plt.ylabel('Distance to Real Path')
    plt.title('Distance from Real Path to Each Optimal Path')
    plt.xticks(range(numOptimalPaths), [f'Path {i+1}' for i in range(numOptimalPaths)])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    optimalPathInRealPath = []
    for p, dt, op in realPath:
        if op not in optimalPathInRealPath:
            optimalPathInRealPath.append(op)
    
    print(f"Optimal Paths in Real Path: {len(optimalPathInRealPath)}")



if __name__ == "__main__":
    main()