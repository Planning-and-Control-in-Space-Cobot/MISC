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


def run_global_optimization(globalOptimalPlanner,
                            robot,
                            environment,
                            initialPath,
                            obstacles, 
                            maxDistances,
                            dt,
                            xi, 
                            xf):

    optimizationSucessful = False
    while not optimizationSucessful:
        Starttime = time.time()
        optimizedTrajectory, newDt, cost = globalOptimalPlanner.optimize(
            initialPath, 
            obstacles, 
            maxDistances, 
            dt, 
            xi, 
            xf
        )
        print(f"Global optimization took {time.time() - Starttime:.3f} seconds")
        print(f"Optimized trajectory length: {len(optimizedTrajectory)}")
        print(f"Optimized trajectory cost: {cost}")
        print(f"Optimized trajectory dt: {newDt}")

        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            optimizedTrajectory, 
            initialPath, 
            obstacles, 
            maxDistances    
        )

        if not anyCollision:
            initialPath = optimizedTrajectory
            dt = newDt
            prevCost = cost
            optimizationSucessful = True
            print("Total Cost :", cost)

    return initialPath, dt, prevCost

def globalOptimization(stateLowerBound, 
                        stateUpperBound, 
                        pcdPath, 
                        J, 
                        A,
                        m,
                       initialPath : List[OptimizationState], 
                       sharedData : dict, 
                       lock) -> None:
    """Performs global optimization on the initial path.
    
    This function uses the GlobalOptimalPlanner to optimize the initial path
    based on the environment and obstacles detected. It publishes the optimized 
    trajectory to global variables that can be accessed by other parts of the 
    code.

    Parameters:
        globalOptimizer (GlobalOptimalPlanner):
            Instance of the GlobalOptimalPlanner class used to define and solve
            the optimization problem.
        environment (EnvironmentHandler):
            Instance of the EnvironmentHandler class that provides the 
            environment and obstacles for the optimization.
        initialPath (list[OptimizationState]):
            List of optimization states representing the initial path to be 
            optimized.
        robot (Robot):
            Instance of the Robot class that provides the robot model and 
            dynamics for the optimization.
    Returns:
        None
    """
    
    _allOptimalPaths =  []

    prevCost = 0
    prevDt = 0.2
    
    collisionObstacles = []
    i=0

    print(f"State Min Values: {stateLowerBound}")
    print(f"State Max Values: {stateUpperBound}")

    environment = EnvironmentHandler(
        o3d.io.read_point_cloud(pcdPath)
    )
    robot = Robot(
        J=J,
        A=A,
        m=m,
    )

    globalOptimizer = GlobalOptimalPlanner(
        stateMinValues=stateLowerBound, 
        stateMaxValues=stateUpperBound,
        env=environment,
        robot=robot,
    )
    while True:
        print(f"{Fore.YELLOW}Global Optimization Iteration {i}{Style.RESET_ALL}")
        i += 1
        with lock:
            _currentPosition = sharedData["currentPosition"]
            _atEnd = sharedData["atEnd"]
        
        if _atEnd:
            print(Fore.GREEN + "Global optimization finished." + Style.RESET_ALL)
            allOptimalPaths = _allOptimalPaths
            break
            
        if currentPosition is None:
            print(Fore.YELLOW + f"Waiting for the current position to be set" + Style.RESET_ALL)
            time.sleep(0.1)
            continue

        startPointIndex = min(
            range(len(initialPath)), 
            key=lambda j: np.linalg.norm(initialPath[j].x - _currentPosition.x)
        )

        _initialPath = [_currentPosition] + initialPath[startPointIndex + 1:]
        obstacles, maxDistances = robot.getObstacles(environment, _initialPath)

        newCollisionsObstacles = []
        for j, p in enumerate(initialPath):
            if p not in _initialPath:
                continue
            indexNewPath = _initialPath.index(p)
            for k, col in collisionObstacles:
                if col.iteration == j:
                    newCollisionsObstacles.append(
                        Obstacle(
                            closestPointObstacle=col.closestPointObstacle,
                            closestPointRobot=col.closestPointRobot,
                            normal=col.normal,
                            iteration=indexNewPath,
                        )
                    )
                    break
        obstacles += newCollisionsObstacles

        print(Fore.YELLOW +  f"Length of the path is {len(_initialPath)} and starting point is {startPointIndex}" + Style.RESET_ALL)

        timeStart = time.time()
        try:
            _optimalTrajectory, _newDt, _cost = globalOptimizer.optimize(
                _initialPath, 
                obstacles, 
                maxDistances, 
                prevDt, 
                _currentPosition, 
                _initialPath[-1], 
            )

        except Exception as e:
            print(Fore.RED + f"Error during global optimization: {e}" + Style.RESET_ALL)
            continue

        timeEnd = time.time()
        print(Fore.YELLOW + f"Global optimization took {timeEnd - timeStart:.3f} seconds" + Style.RESET_ALL)

        collisionFree = robot.collisionFree(_optimalTrajectory, environment)

        _allOptimalPaths.append((_optimalTrajectory, _newDt, _cost))

        if collisionFree:
            print(Fore.CYAN + "No Collision found, updating global variables." + Style.RESET_ALL)
            with lock:
                sharedData["currentTrajectory"] = _optimalTrajectory
                sharedData["currentTimeStep"] = _newDt
            
            costVariation = np.abs(prevCost - _cost) / _cost
            timeVariation = np.abs(prevDt - _newDt) / _newDt
            
            if costVariation < 0.01 and timeVariation < 0.01:
                print(Fore.GREEN + "Global optimization converged." + Style.RESET_ALL)
                with lock:
                    sharedData["allOptimalPaths"] = _allOptimalPaths
                break
            else:
                initialPath = _optimalTrajectory
                prevCost = _cost
                prevDt = _newDt
        else:
            pass

def localOptimization(
        stateLowerBound : np.ndarray,
        stateUpperBound : np.ndarray,
        pcdPath : str, 
        J, 
        A, 
        m,
        simulator : Simulator,
        sharedData : dict, 
        lock,
        localHorizon : int = 10
    ):
    """Performs local optimization on the current global optimal trajectory.

    This function uses the LocalOptimalPlanner to optimize the local trajectory, 
    this is a subset of the global optimal trajectory based on the environment 
    and the obstacles detected. After the optimization, it uses the step to
    update the current position, so that the global optimization know the 
    current position of the robot, and only optimized the trajectory from there 
    to the the goal node

    Parameters:
        localOptimizer (LocalOptimalPlanner):
            Instance of the LocalOptimalPlanner class used to define and solve
            the optimization problem.
        environment (EnvironmentHandler):
            Instance of the EnvironmentHandler class that provides the 
            environment and obstacles for the optimization.
        robot (Robot):
            Instance of the Robot class that provides the robot model and 
            dynamics for the optimization.
        localHorizon (int, optional):
            Horizon size for the local optimization. Defaults to 10.
        
    Returns:
        None
        
    """
    environment = EnvironmentHandler(
        o3d.io.read_point_cloud(pcdPath)
    )
    robot = Robot(
        J=J,
        A=A,
        m=m,
    )
    localOptimizer = LocalOptimalPlanner(
        stateMinValues=stateLowerBound, 
        stateMaxValues=stateUpperBound,
        env=environment,
        robot=robot,
    )

    with lock:
        _currentTrajectory = sharedData["currentTrajectory"]
        _currentPosition = sharedData["currentPosition"]

    realPath = [(_currentPosition, -1, None)]

    i = 0 
    while True:
        print(Fore.BLUE + f"Local Optimization Iteration {i}" + Style.RESET_ALL)

        with lock:
            if _currentTrajectory != sharedData["currentTrajectory"]:
                print(Fore.MAGENTA + f"Current trajectory input changed" + Style.RESET_ALL)

            _currentTrajectory = sharedData["currentTrajectory"]
            _currentTimeStep = sharedData["currentTimeStep"]
            _currentPosition = sharedData["currentPosition"]
            _atEnd = sharedData["atEnd"]

        if _currentTrajectory is None:
            print(Fore.YELLOW + f"Waiting for the current Trajectory to be set" + Style.RESET_ALL)
            time.sleep(1)
            continue

        print(Fore.BLUE + f"Size Global Trajectory: {len(_currentTrajectory)}" + Style.RESET_ALL)

        i += 1

        startPointIndex = min(
            range(len(_currentTrajectory)),
            key=lambda j: np.linalg.norm(_currentTrajectory[j].x - _currentPosition.x),
        )

        print(Fore.BLUE + f"Start point index: {startPointIndex}" + Style.RESET_ALL)

        initialLocalTrajectory = [_currentPosition]
        initialLocalTrajectory.extend(_currentTrajectory[startPointIndex + 1:])

        if len(initialLocalTrajectory) > localHorizon:
            initialLocalTrajectory = initialLocalTrajectory[:localHorizon]
        else:
            initialLocalTrajectory.extend([initialLocalTrajectory[-1]] * (localHorizon - len(initialLocalTrajectory)))
        


        obstacles, maxDistances = robot.getObstacles(
            environment,
            initialLocalTrajectory, 
        )

        #print(Fore.BLUE + f"Len Initial Trajectory: {len(initialTrajectory)} num Obstacles {len(obstacles)} local horizon {localHorizon}" + Style.RESET_ALL)

        startTime = time.time()
        _optimalTrajectory, _, _ = localOptimizer.optimize(
            initialLocalTrajectory,
            obstacles, 
            maxDistances, 
            _currentTimeStep, 
            _currentPosition, 
            _currentTrajectory[-1],
            0
        )

        endTime = time.time()
        print(Fore.BLUE + f"Local optimization took {endTime - startTime:.3f} seconds with simulated Time Step of {_currentTimeStep}" + Style.RESET_ALL)


        state = simulator.simulate(
            _optimalTrajectory[0].get_state(),
            _optimalTrajectory[0].u,
            _currentTimeStep
        )[:, -1]

        state = OptimizationState(
            x=state[0:3],
            v=state[3:6],
            q=state[6:10],
            w=state[10:13], 
        )

        realPath[-1][0].u = _optimalTrajectory[0].u
        realPath.append((state, _currentTimeStep, _currentTrajectory))

        if np.linalg.norm((state.x - _currentTrajectory[-1].x)) < 0.1:
            print(Fore.GREEN + "Reached the goal node." + Style.RESET_ALL)
            with lock:
                sharedData["atEnd"] = True
                sharedData["finalTrajectory"] = realPath
            break

        collisionFree = robot.collisionFree(
            [rp[0] for rp in realPath],
            environment,
        )
        print(Fore.BLUE + f"Collision Found in Real Path: {collisionFree}" + Style.RESET_ALL)


        with lock:
            sharedData["currentPosition"] = state


            #drawLOTWithGOTAndFCP(
            #    environment, 
            #    robot, 
            #    _currentTrajectory,
            #    _optimalTrajectory,
            #    realPath
            #)

def drawLOTWithGOTAndFCP(

        environment, 
        robot, 
        globalOptimalPath, 
        localOptimalTrajectory, 
        fullCorrectPath
    ):
    """Draws the Full problem.

    Draws the local optimal trajectory, with the global optimal trajectory, and 
    the full path taken until the step

    Parameters:
        environment (EnvironmentHandler):
            Instance of the EnvironmentHandler class that provides the 
            environment and obstacles for the optimization.
        robot (Robot):
            Instance of the Robot class that provides the robot model and 
            dynamics for the optimization.
        globalOptimalPath (List[OptimizationState]):
            List of optimization states representing the global optimal path.
        localOptimalTrajectory (List[OptimizationState]):
            List of optimization states representing the local optimal trajectory.
        fullCorrectPath (List[Tuple[List[OptimizationState], float]]):
            List of optimization states representing the full correct path taken.
    """
    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)

    for p in globalOptimalPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="red",
            show_edges=True,
            opacity=0.5,
        )

    for p in localOptimalTrajectory:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="blue",
            show_edges=True,
            opacity=0.5,
        )
    for p, _ in fullCorrectPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="green",
            show_edges=True,
            opacity=0.5,
        )
    pv_.add_text("Global Optimal Path", position="upper_left", color="red")
    pv_.add_text("Local Optimal Trajectory", position="upper_right", color="blue")
    pv_.add_text("Full Correct Path", position="lower_left", color="green")
    pv_.add_axes()
    pv_.show_grid()
    pv_.show()

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

    print("Creating Robot object")
    A = np.load(os.path.join(script_dir, "A_matrix.npy"))
    J = np.load(os.path.join(script_dir, "J_matrix.npy"))
    m = np.load(os.path.join(script_dir, "mass.npy"))

    robot = Robot(
        J,
        A,
        m,
    )

    stateLowerBound = np.hstack([
        np.array([0.0, 3.0, 0.0]),  # x, y, z
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
