import os
import sys
import time 
import argparse
from colorama import Fore, Style
from typing import List, Tuple
from multiprocessing import Process, Manager, Lock
import multiprocessing as mp


import numpy as np
import open3d as o3d
import pyvista as pv
import scipy.spatial.transform as trf
import matplotlib.pyplot as plt
import pickle

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from Environment import EnvironmentHandler

from Obstacle import Obstacle
from Robot import Robot
from OptimizationState import OptimizationState
from LocalOptimalPlanner import LocalOptimalPlanner
from GlobalOptimalPlanner import GlobalOptimalPlanner

from Simulator import Simulator
from Simulator.SpaceCobotModel import SpaceCobot

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

    # Get full path file for the path
    pathFile = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),\
        args.path
    )

    # Get full path file for the map
    mapFile = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.map
    )

    # Check if the path and map file exists
    if not os.path.exists(pathFile):
        raise ValueError(
            f"Path '{pathFile}' does not exist. Please provide a valid path."
        )
    if not os.path.exists(mapFile):
        raise ValueError(
            f"Map '{mapFile}' does not exist. Please provide a valid point cloud map file."
        )

    # Load the map into the environment
    environment = EnvironmentHandler(mapFile)

    # Load the path from the file (npz) and save in OptimizationState format
    path = np.load(os.path.join(script_dir, args.path))
    originalPosition = path["positions"]
    originalOrientation = path["orientations"]
    initialPath = [
        OptimizationState(originalPosition[i], originalOrientation[i])
        for i in range(len(originalPosition))
    ]

    minV = np.array([-5, -5, -5])
    maxV = np.array([5, 5, 5])
    minW = np.array([-4, -4, -4])
    maxW = np.array([4, 4, 4])
    dt = 0.2

    # Set velocities and angular velocities for the initial path
    for i in range(len(initialPath) - 1):
        v = (initialPath[i+1].x - initialPath[i].x) / dt
        initialPath[i].v = np.clip(v, minV, maxV)
        w = (1 / dt) * (trf.Rotation.from_quat(initialPath[i+1].q) * 
            trf.Rotation.from_quat(initialPath[i].q).inv()).as_rotvec()
        initialPath[i].w = np.clip(w, minW, maxW)

    _initialPath = initialPath.copy()

    # Loading the robot dynamics parameters
    A = np.load(os.path.join(script_dir, "A_matrix.npy"))
    J = np.load(os.path.join(script_dir, "J_matrix.npy"))
    m = np.load(os.path.join(script_dir, "mass.npy"))

    # Creating the robot object from the loaded dynamics parameters
    robot = Robot(
        J,
        A,
        m,
    )

    # Lower and upper bounds for the state space in the optimization problem
    stateLowerBound = np.hstack([
        np.array([-2.0, -2.0, -2.0]),  # x, y, z
        minV,
        np.array([-1, -1, -1, -1]), 
        minW,
    ])

    stateUpperBound = np.hstack([
        np.array([3.0, 6.0, 7.0]),  # x, y, z
        maxV,
        np.array([1, 1, 1, 1]),
        maxW,
    ])

    spaceCobot = SpaceCobot(m, J, A)
    simulator = Simulator(
        spaceCobot
    )

    globalOptimizationStable = False    
    collisions = []

    globalOptimalPlanner = GlobalOptimalPlanner(
        stateLowerBound, 
        stateUpperBound, 
        environment,
        robot
    )

    while not globalOptimizationStable:
        obstacles, maxDistances = robot.getObstacles(
            environment,
            initialPath
        )

        obstacles.extend(collisions)

        try: 
            timeStart = time.time()
            optimizedPath, timeStep = globalOptimalPlanner.optimize(
                initialPath, 
                obstacles, 
                maxDistances, 
                dt, 
                initialPath[0], 
                initialPath[-1]
            )
        except Exception as e:
            print(Fore.RED + f"Error during optimization: {e}" + Style.RESET_ALL)

        timeEnd = time.time()
        print(Fore.YELLOW + f"Optimization took {timeEnd - timeStart:.4f} seconds" + Style.RESET_ALL)

        print(Fore.RED + f"start Point {initialPath[0].x} end Point {initialPath[-1].x}" + Style.RESET_ALL)
        print(Fore.RED + f"start Point {optimizedPath[0].x} end Point {optimizedPath[-1].x}" + Style.RESET_ALL)


        if robot.collisionFree(optimizedPath, environment):
            dtVariation = np.abs(timeStep - dt) / dt * 100
            print(Fore.GREEN + f"Optimization successful with time step variation: {dtVariation:.2f}% - {timeStep}" + Style.RESET_ALL)
            dt = timeStep 
            optimizedPath[0] = initialPath[0]
            optimizedPath[-1] = initialPath[-1]
            initialPath = optimizedPath
        
        else:
            print(Fore.RED + "Optimization failed due to collisions" + Style.RESET_ALL)
            newCollisions = robot.getCollision(
                environment,
                optimizedPath
            )

            pv_ = pv.Plotter()
            pv_ = environment.visualizeCoalMesh(pv_)

            for p in initialPath:
                x = p.x
                R = trf.Rotation.from_quat(p.q)
                robotMesh = robot.getPVMesh(x, R)
                pv_.add_mesh(
                    robotMesh,
                    color="green",
                    show_edges=True,
                    opacity=0.5,
                )
            

            for o in obstacles:
                normal = o.normal
                point = o.closestPointObstacle
                arrow = pv.Arrow(
                    start=point,
                    direction=normal,
                    scale=0.1,
                )
                plane = pv.Plane(
                    center=point,
                    direction=normal,
                    i_size=0.5,
                    j_size=0.5,
                )
                pv_.add_mesh(plane, color="yellow", opacity=0.5)
                pv_.add_mesh(arrow, color="yellow")


            for p in optimizedPath:
                x = p.x
                R = trf.Rotation.from_quat(p.q)
                robotMesh = robot.getPVMesh(x, R)
                pv_.add_mesh(
                    robotMesh,
                    color="blue",
                    show_edges=True,
                    opacity=0.5,
                )
                
            collisions.extend(newCollisions)
            for c in collisions:
                normal = c.normal
                point = c.closestPointObstacle
                arrow = pv.Arrow(
                    start=point,
                    direction=normal,
                    scale=0.1,
                )

                plane = pv.Plane(
                    center=point,
                    direction=normal,
                    i_size=0.5,
                    j_size=0.5,
                )
                pv_.add_mesh(plane, color="red", opacity=0.5)
                pv_.add_mesh(arrow, color="red")

            pv_.show_axes()
            pv_.show_grid()
            pv_.add_axes_at_origin()
            pv_.show()


    optimizationPath = initialPath.copy()
    xi = optimizationPath[0]
    xf = optimizationPath[-1]

    dt = 0.2

    numSuccessfulOptimizationsRequired = 10
    optimizationHorizon = 10
    lookAhead = 10
    windowStart = 0
    prevU = np.zeros((6, optimizationHorizon + lookAhead))

    manager = Manager()
    sharedData = manager.dict()
    lock = Lock()
    sharedData["currentPosition"] = initialPath[0]
    sharedData["currentTrajectory"] = None
    sharedData["currentTimeStep"] = dt
    sharedData["atEnd"] = False
    sharedData["allOptimalPaths"] = []

    #globalOptimalPlanner = GlobalOptimalPlanner(
    #    stateMinValues=stateLowerBound, 
    #    stateMaxValues=stateUpperBound,
    #    env=environment,
    #    robot=robot,
    #)

    localOptimalPlanner = LocalOptimalPlanner(
        stateMinValues=stateLowerBound,
        stateMaxValues=stateUpperBound,
        env=environment,
        robot=robot,
    )

    localSize = 10

    optimizedPaths = []
    prevCost = 0
    firstOptimization = True

    global currentPosition 
    currentPosition = initialPath[0]

    globalOptimizationProcess = Process(
        target=globalOptimization, 
        args=(
            stateLowerBound, 
            stateUpperBound, 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), args.map), 
            J, 
            A,
            m, 
            initialPath,
            sharedData, 
            lock
        )
    )


    localOptimizationProcess = Process(
        target=localOptimization,
        args=(
            stateLowerBound, 
            stateUpperBound, 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), args.map),
            J,
            A,
            m,
            simulator,
            sharedData,
            lock,
            localSize,
        )
    )

    globalOptimizationProcess.start()
    localOptimizationProcess.start()
    globalOptimizationProcess.join()
    localOptimizationProcess.join()

    with open(os.path.join(script_dir, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "initialPath": initialPath,
                "realPath": sharedData["finalTrajectory"],
                "optimalPaths": sharedData["allOptimalPaths"]
            }, f
        )
    return

    with open(os.path.join(script_dir, "results.pkl"), "wb") as f:
        pickle.dump({
            "initialPath" : initialPath,
            "realPath" : finalTrajectory, 
            "optimalPaths" : allOptimalPaths,
        }, f)

    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)
    pv_.add_axes()
    pv_.show_grid()
    for p, _, _ in finalTrajectory:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="blue",
            show_edges=True,
            opacity=0.5,
        )
    pv_.show()
    return

if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    mp.set_start_method("fork")

    main()
