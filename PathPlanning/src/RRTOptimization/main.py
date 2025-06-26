import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
import scipy.spatial.transform as trf
import pickle

from colorama import Fore, Style


# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


from Environment import EnvironmentHandler
from RRTOptimization import (
    Robot,
    RRTPathOptimization,
    Obstacle,
    OptimizationState,
)

import time as time

import argparse


def createPyBox(axis):
    """Create a PyVista box mesh with the given axis."""
    box = pv.Box(
        bounds=(
            -axis[0] / 2,
            axis[0] / 2,
            -axis[1] / 2,
            axis[1] / 2,
            -axis[2] / 2,
            axis[2] / 2,
        )
    )
    return box

def drawEnvironmentAndNormals(environment, obstacles, robot, path):
    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)

    for obs in obstacles:
        pos = obs.closestPointObstacle
        normal = obs.normal
        pv_.add_arrows(pos, normal, mag=0.3, color="orange")
    
    for p in path:
        x = p.x
        q = p.q
        R = trf.Rotation.from_quat(q)
        robot_mesh = robot.getPVMesh(x, R)
        if robot_mesh is not None:
            pv_.add_mesh(robot_mesh, color="blue", opacity=0.4)
    
    pv_.show()
    return

def drawEnvironmentAndObstacles(environment, obstacles):
    """Draws the environment and the obstacles in a PyVista plotter.

    Parameters:
        environment (EnvironmentHandler): The environment handler containing the voxel mesh.
        obstacles (list[Obstacle]): List of obstacles to be visualized.

    Returns:
        pv.Plotter: The PyVista plotter with the environment and obstacles.
    """
    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)

    for obs in obstacles:
        pos = obs.closestPointObstacle
        normal = obs.normal
        pv_.add_arrows(pos, normal, mag=0.3, color="orange")
    pv_.show()

    return pv_

def main():
    parser = argparse.ArgumentParser(
        description="RRT Path Planning and Optimization"
    )

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        description="RRT Path Planning and Optimization"
    )


    parser.add_argument(
        "--path", 
        "-p",
        type=str,
        default="path.npz",
        help="Path to the saved path file",
    )
    parser.add_argument(
        "--map",
        "-m",
        type=str,
        default="map.pcd",
        help="Path to the point cloud map file",
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(script_dir, args.path)):
        raise ValueError(
            f"Path file '{args.path}' does not exist. Please provide a valid path file."
        )

    if not os.path.exists(os.path.join(script_dir, args.map)):
        raise ValueError(
            f"Map file '{args.map}' does not exist. Please provide a valid point cloud map file."
        )

    pcd = o3d.io.read_point_cloud(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.map)
    )
    environment = EnvironmentHandler(pcd)
    # environment._debugPointCloud(#

    path = np.load(os.path.join(script_dir, args.path))
    originalPosition = path["positions"]
    originalOrientation = path["orientations"]
    initialPath = [
        OptimizationState(originalPosition[i], originalOrientation[i], i=i)
        for i in range(len(originalPosition))
    ]

    minV = np.array([-5, -5, -5])
    maxV = np.array([5, 5, 5])
    minW = np.array([-2, -2, -2])
    maxW = np.array([2, 2, 2])
    dt = 0.2

    for i in range(len(initialPath) - 1):
        initialPath[i].v = np.clip((initialPath[i+1].x - initialPath[i].x) / dt / 2, minV, maxV)
        print(f"i{i} v: {initialPath[i].v}")
        #initialPath[i].w = np.clip((initialPath[i+1] - initialPath[i]) / dt, minW, maxW)

    print("Creating Robot object")
    A = np.load(os.path.join(script_dir, "A_matrix.npy"))
    J = np.load(os.path.join(script_dir, "J_matrix.npy"))
    m = np.load(os.path.join(script_dir, "mass.npy"))

    robot = Robot(
        J,
        A,
        m,
        environment.buildBox(),
        pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06)),
    )

    # min_corner=[1.5, 1, 0], max_corner=[4., 6, 10
    stateLowerBound = np.hstack([
        np.array([1.5, 1, 0]),  # x, y, z
        minV,
        np.array([-1, -1, -1, -1]), 
        minW,
    ])

    stateUpperBound = np.hstack([
        np.array([4.0, 6, 10]),  # x, y, z
        maxV,
        np.array([1, 1, 1, 1]),
        maxW,
    ])


    # stateLowerBound = np.array([0, -25, 0, -5, -5, -5, -1, -1, -1, -1, -2, -2, -2])
    # stateUpperBound = np.array([25, 25, 2, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2])
    
    numOptimizedSteps = 20 
    stride = int(numOptimizedSteps / 2)
    numTimesOptimized = 5
    rrtOpt = RRTPathOptimization(
        stateLowerBound, stateUpperBound, environment, robot
    )

    prevU = np.zeros((6, numOptimizedSteps))
    dt = 0.2
    windowStart = 0
    optimizedPath = []
    prevUs = []
    dts = []
    _initialPath = initialPath.copy()
    obstacles, maxDistances, _, _ = robot.getObstacles(environment, _initialPath[0:numOptimizedSteps], [], [], [])
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    """ 
    while len(optimizedPath) < len(initialPath):
        print(f"Optimizing path going from {windowStart} to {windowStart + numOptimizedSteps}")
        numSuccessfulOptimizations = 0
        while numSuccessfulOptimizations < numTimesOptimized:
            if windowStart + numOptimizedSteps < len(_initialPath):
                optimizationPath = _initialPath[windowStart:windowStart + numOptimizedSteps]
            else:
                optimizationPath = _initialPath[windowStart:]
                optimizationPath.extend((numOptimizedSteps - len(optimizationPath)) * [initialPath[-1]])
            
            xi = np.zeros(13)
            xi[0:3] = optimizationPath[0].x
            xi[3:6] = optimizationPath[0].v
            xi[6:10] = optimizationPath[0].q
            xf = np.zeros(13)
            xf[0:3] = _initialPath[-1].x
            xf[3:6] = _initialPath[-1].v
            xf[6:10] = _initialPath[-1].q

            #print(f"xi {xi}")
            #print(f"xf {xf}")
            #print(f"Len Obstacles: {len(obstacles)}")
            rrtOpt.setup_optimization(
                optimizationPath, 
                obstacles, 
                maxDistances, 
                prevU, 
                xi, 
                xf
            )
            print(f"{Fore.BLUE} Window Start: {windowStart} numOptimization:{numSuccessfulOptimizations} {Style.RESET_ALL}")

            sol = rrtOpt.optimize(optimizationPath, dt, prevU)
            _optimizationPath, _prev_u, _dt, cost = rrtOpt.getSolution(sol)
            print(f"{Fore.YELLOW}Cost: {cost}{Style.RESET_ALL}")
    
            _obstacles, _maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
                environment, 
                _optimizationPath, 
                optimizationPath, 
                obstacles, 
                maxDistances
            )

            if not anyCollision:
                numSuccessfulOptimizations += 1
                print(
                    f"{Fore.GREEN}No collision detected during optimization! windowStart {windowStart} numOptimizedSteps {numSuccessfulOptimizations}{Style.RESET_ALL}"
                )
                prevU = _prev_u
                dt = _dt
                _initialPath[windowStart:windowStart + numOptimizedSteps] = _optimizationPath


                if numSuccessfulOptimizations == numTimesOptimized:
                    print(f"{Fore.GREEN} InitialPaht Len: {len(initialPath)} OptimizedPath Len: {len(optimizedPath)} {Style.RESET_ALL}")
                    if windowStart % 10 == 0:
                        pv_ = rrtOpt.visualize_trajectory(
                            initialPath, optimizedPath, environment.voxel_mesh)
                        for p in optimizedPath:
                            print(f"path : {p.x} {trf.Rotation.from_quat(p.q).as_euler('xyz', degrees=True)}")
                        pv_.show()
                    prevUs.append(_prev_u[:, 0:stride])
                    dts.append(dt)
                    optimizedPath.extend(_optimizationPath[0:stride])

                    prevU[:, :-stride] = prevU[:, stride:]
                    prevU[:, -stride:] = np.zeros((6,stride))

                    windowStart += stride 
                    obstacles, maxDistances, _, _ = robot.getObstacles(
                        environment,
                        _optimizationPath,
                        _initialPath[windowStart:windowStart + numOptimizedSteps],
                        obstacles,
                        maxDistances
                    )
            else:
                print(
                    f"{Fore.RED}Collision detected during optimization!{Style.RESET_ALL}"
                )
                drawEnvironmentAndNormals(environment, collisionObstacles, robot, _optimizationPath)

                obstacles = _obstacles
    """

    optimizationPath = initialPath.copy()
    xi = np.zeros(13)
    xf = np.zeros(13)
    xi[0:3] = optimizationPath[0].x
    xi[6:10] = optimizationPath[0].q
    numValidOptimizations = 0

    optimizationPath = initialPath.copy()
    xi = np.zeros(13)
    xf = np.zeros(13)
    xi[0:3] = optimizationPath[0].x
    xi[6:10] = optimizationPath[0].q
    print(f"xi: {xi}")
    xf[0:3] = optimizationPath[-1].x
    xf[6:10] = optimizationPath[-1].q
    print(f"xf: {xf}")

    prev_u = np.zeros((6, len(optimizationPath)))

    dt = 0.2
    optimizedPaths = []

    obstacles, maxDistances, _,  _  = robot.getObstacles(
        environment,
        optimizationPath, 
        [], 
        [], 
        []
    )

    for i in range(20):
        print(f'Num obstacle considered in this step : {len(obstacles)}')

        startTime = time.time()
        rrtOpt.setup_optimization(optimizationPath, obstacles, maxDistances, prev_u, xi=xi, xf=xf)

        sol = rrtOpt.optimize(
            optimizationPath, dt=dt, prev_u=prev_u
        )
        optimizationTime = time.time() - startTime

        _optimizationPath, _prev_u, _dt, cost = rrtOpt.getSolution(sol)
        print(f"Time taken for optimization: {optimizationTime:.2f} seconds")

        #if np.abs(_dt - dt) < 0.05:
        #    rrtOpt.convexOptimization(
        #        optimizationPath, obstacles, maxDistances, prev_u, dt, xi, xf
        #    )



        # Evaluate the trajectory to ensure it is valid
        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            _optimizationPath, 
            optimizationPath, 
            obstacles, 
            maxDistances    
        )

        if not anyCollision:
            pv_ = rrtOpt.visualize_trajectory(
                optimizationPath, _optimizationPath, environment.voxel_mesh, None, []
            )
            pv_.show()
            optimizationPath = _optimizationPath
            prev_u = _prev_u
            dt = _dt
            optimizedPaths.append((_optimizationPath, _prev_u, _dt, cost))

        else:
            drawEnvironmentAndNormals(environment, collisionObstacles, robot, _optimizationPath)

            
    pv_ = rrtOpt.visualize_trajectory(
        initialPath, optimizedPaths[-1][0], environment.voxel_mesh, None, [] 
    )
    pv_.show()

    with open(os.path.join(script_dir, "optimizedPath.pkl"), "wb") as f:
        pickle.dump(optimizedPaths, f)
    
    print(f"Optimized path saved to {os.path.join(script_dir, 'optimizedPath.pkl')}")



            

if __name__ == "__main__":
    main()
