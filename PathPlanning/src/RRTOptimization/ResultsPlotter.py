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
            print(f'sop {type(sop)} optimalPath {type(optimalPath)}')
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


    plt.figure()
    plt.subplot(2, 2, 1)
    times = [dt for _, dt, _ in realPath]
    times[0] = 0
    comulativeTimes = np.cumsum(times)

    pos = np.array([p.x for p, _, _ in realPath])
    Rs = [trf.Rotation.from_quat(p.q) for p, _, _ in realPath]
    plt.plot(comulativeTimes, pos[:, 0], label='X Position')
    plt.plot(comulativeTimes, pos[:, 1], label='Y Position')
    plt.plot(comulativeTimes, pos[:, 2], label='Z Position')
    for i in range(len(realPath) - 1):
        if realPath[i][2] != realPath[i + 1][2]:
            plt.axvline(x=comulativeTimes[i], color='gray', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Position Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(comulativeTimes, times, label='Time Step', color='orange')
    w = np.array([p.w for p, _, _ in realPath])
    plt.plot(comulativeTimes, w[:, 0], label='X Angular Velocity', color='purple')
    plt.plot(comulativeTimes, w[:, 1], label='Y Angular Velocity', color='green')
    plt.plot(comulativeTimes, w[:, 2], label='Z Angular Velocity', color= 'red')
    for i in range(len(realPath) - 1):  
        if realPath[i][2] != realPath[i + 1][2]:
            plt.axvline(x=comulativeTimes[i], color='gray', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Time Step (s) / Angular Velocity (rad/s)')
    plt.title('Time Step and Angular Velocity Over Time')
    plt.legend()
    plt.grid()
    # Uncomment the following lines if you want to plot time step vs time


    #plt.xlabel('Time (s)')
    #plt.ylabel('Time Step (s)')
    #plt.title('Time Step Over Time')
    #plt.legend()
    #plt.grid()

    plt.subplot(2, 2, 3)
    vel = np.array([p.v for p, _, _ in realPath])
    plt.plot(comulativeTimes, vel[:, 0], label='X Velocity')
    plt.plot(comulativeTimes, vel[:, 1], label='Y Velocity')
    plt.plot(comulativeTimes, vel[:, 2], label='Z Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    acc = np.array([trf.Rotation.from_quat(p.q).as_euler('xyz', degrees=True) for p, _, _ in realPath])
    plt.plot(comulativeTimes, acc[:, 0], label='Roll')
    plt.plot(comulativeTimes, acc[:, 1], label='Pitch')
    plt.plot(comulativeTimes, acc[:, 2], label='Yaw')
    for i in range(len(realPath) - 1):
        if realPath[i][2] != realPath[i + 1][2]:
            plt.axvline(x=comulativeTimes[i], color='gray', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Attitude (degrees)')
    plt.title('Attitude Over Time')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(comulativeTimes, times, label='Time Step', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Time Step (s)')
    plt.title('Time Step Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    u = np.array([p.u for p, _, _ in realPath])
    plt.plot(comulativeTimes, u[:, 0], label='Motor 0')
    plt.plot(comulativeTimes, u[:, 1], label='Motor 1')
    plt.plot(comulativeTimes, u[:, 2], label='Motor 2')
    plt.plot(comulativeTimes, u[:, 3], label='Motor 3')
    plt.plot(comulativeTimes, u[:, 4], label='Motor 4')
    plt.plot(comulativeTimes, u[:, 5], label='Motor 5')

    plt.plot(comulativeTimes, 3 * np.ones_like(comulativeTimes), label='Max Motor Thrust', linestyle='--', color='red')
    plt.plot(comulativeTimes, -3 * np.ones_like(comulativeTimes), label='Min Motor Thrust', linestyle='--', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Thrust (N)')
    plt.title('Motor Thrust Over Time')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()





    plt.show()


    optimalPathInRealPath = []
    for p, dt, op in realPath:
        if op not in optimalPathInRealPath:
            optimalPathInRealPath.append(op)
    
    print(f"Optimal Paths in Real Path: {len(optimalPathInRealPath)}")



if __name__ == "__main__":
    main()