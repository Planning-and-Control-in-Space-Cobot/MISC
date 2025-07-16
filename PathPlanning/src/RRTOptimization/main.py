import os
import sys
import time 
import argparse
from colorama import Fore, Style
from typing import List, Tuple
from threading import Thread, Lock

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

initialTrajectory = None
currentPosition = None
currentTimeStep = None
atEnd = False
finalTrajectory = None
allOptimalPaths = []
optimizationLock = Lock()


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

def drawOptimizationProblem(environment, obstacles, robot, path, fullProblem=False):
    """Draws the optimization problem inputs

    Draws the environment, the robot in each pose and the obstacles detected for 
    each pose in the path. Has the possibility to draw the full optimization
    problem or each step in the path and their constraints
    """
    if fullProblem:
        pv_ = pv.Plotter90
        pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)
        pv_.add_axes()
        pv_.show_grid()

        for p in enumerate(path):
            x = p.x
            R = trf.Rotation.from_quat(p.q)
            robotMesh = robot.getPVMesh(x, R)
            pv_.add_mesh(
                robotMesh,
                color="blue",
                show_edges=True,
                opacity=0.5,
            )
        
        for o in obstacles:
            plane = pv.Plane(
                center=o.closestPointObstacle,
                direction=o.normal
            )
            arrow = pv.Arrow(
                start=o.closestPointObstacle,
                direction=o.normal,
                scale=0.1,
                tip_length=0.2,
            )
            pv_.add_mesh(
                plane,
                color="red",
                show_edges=True,
                line_width=1.0,
            )

            pv_.add_mesh(
                arrow,
                color="red",
                show_edges=True,
                line_width=1.0,
            )

        pv_.show()

    else:
        for i, p in enumerate(path):

            pv_ = pv.Plotter()
            pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)
            pv_.add_axes()
            pv_.show_grid()

            x = p.x
            R = trf.Rotation.from_quat(p.q)
            robotMesh = robot.getPVMesh(x, R)
            pv_.add_mesh(
                robotMesh,
                color="blue",
                show_edges=True,
                opacity=0.5,
            )
            _obstacles = [o for o in obstacles if o.iteration == i]
            print(
                f"Drawing step {i} of the optimization problem "
                f"with pose {p.x} and orientation {p.q}"
                f" with {len(_obstacles)} obstacles detected."
            )
            for o in _obstacles:
                plane = pv.Plane(
                    center=o.closestPointObstacle,
                    direction=o.normal
                )
                arrow = pv.Arrow(
                    start=o.closestPointObstacle,
                    direction=o.normal,
                    scale=0.1,
                    tip_length=0.2,
                )
                pv_.add_mesh(
                    plane,
                    color="red",
                    show_edges=True,
                    line_width=1.0,
                )

                pv_.add_mesh(
                    arrow,
                    color="red",
                    show_edges=True,
                    line_width=1.0,
                )

                arrow = pv.Arrow(
                    start=o.closestPointRobot,
                    direction=-o.normal,
                    scale=0.1,
                    tip_length=0.2,
                )
                pv_.add_mesh(
                    arrow,
                    color="blue",
                    show_edges=True,
                    line_width=1.0,
                )
            pv_.show()


    return

def localVsGlobalVsOriginal(
    environment,
    originalPath,
    localOptimizedPath,
    globalOptimizedPath,
    robot,
):
    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)
    pv_.add_axes()
    pv_.show_grid()

    for p in originalPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="green",
            show_edges=True,
            opacity=0.5,
        )
    
    for p in localOptimizedPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="blue",
            show_edges=True,
            opacity=0.5,
        )
    for p in globalOptimizedPath:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="red",
            show_edges=True,
            opacity=0.5,
        )
    pv_.add_text("Original Path", position="upper_left", color="green")
    pv_.add_text("Local Optimized Path", position="upper_right", color="blue")
    pv_.add_text("Global Optimized Path", position="lower_left", color="red")
    pv_.show()

def drawLocalOptimalTrajectory(
        environment, 
        trajectory, 
        robot, 
        localOptimalTrajectory
    ):
    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=1.0)
    pv_.add_axes()
    pv_.show_grid()

    for p, _ in trajectory:
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="blue",
            show_edges=True,
            opacity=0.5,
        )
    
    for lo in localOptimalTrajectory:
        x = lo.x
        R = trf.Rotation.from_quat(lo.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="orange",
            show_edges=True,
            opacity=0.5,
        )
    pv_.show()

def drawRealPathWithObstaclesAndCollisions(
        environment,
        robot, 
        realPath, 
        collision, 
        obstacles, 
        index = None,
        start = 0,
):
    if index is None:
        index = range(len(realPath))
        
    pv_ = pv.Plotter()
    pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)

    for i, (p, _) in enumerate(realPath):
        if i not in index:
            continue
        x = p.x
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(x, R)
        pv_.add_mesh(
            robotMesh,
            color="blue",
            show_edges=True,
            opacity=0.5,
        )
    
    for o in obstacles:
        if not o.iteration + start in index:
            continue
    
        plane = pv.Plane(
            center=o.closestPointObstacle,
            direction=o.normal
        )
        arrow = pv.Arrow(
            start=o.closestPointObstacle,
            direction=o.normal,
            scale=0.1,
            tip_length=0.2,
        )
        pv_.add_mesh(
            plane,
            color="red",
            show_edges=True,
            line_width=1.0,
        )
        pv_.add_mesh(
            arrow,
            color="red",
            show_edges=True,
            line_width=1.0,
        )   


    for c in collision:
        if not c.iteration in index:
            continue

        plane = pv.Plane(
            center=c.closestPointObstacle,
            direction=c.normal
        )
        arrow = pv.Arrow(
            start=c.closestPointObstacle,
            direction=c.normal,
            scale=0.1,
            tip_length=0.2,
        )
        pv_.add_mesh(
            plane,
            color="green",
            show_edges=True,
            line_width=1.0,
        )
        pv_.add_mesh(
            arrow,
            color="green",
            show_edges=True,
            line_width=1.0,
        )
    
    pv_.add_axes()
    pv_.show_grid()
    pv_.show()

def globalOptimization(globalOptimizer : GlobalOptimalPlanner, 
                       environment : EnvironmentHandler,
                       initialPath : List[OptimizationState], 
                       robot : Robot) -> None:
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
    
    global currentPosition, currentTimeStep, initialTrajectory, atEnd, allOptimalPaths
    _currentPosition = None 
    _currentTimeStep = None
    _initialTrajectory = None
    _atEnd = False

    _allOptimalPaths =  []

    prevCost = 0
    prevDt = 0.2

    obstacles, maxDistances, _, _ = robot.getObstacles(
        environment, 
        initialPath, 
        [], 
        [], 
        []
    )

    i=0
    while True:
        print(f"{Fore.YELLOW}Global Optimization Iteration {i}{Style.RESET_ALL}")
        i += 1
        with optimizationLock:
            _currentPosition = currentPosition
            _atEnd = atEnd
        
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

        distanceCurrentStart = np.linalg.norm(
            initialPath[startPointIndex].x - _currentPosition.x
        )

        if distanceCurrentStart < 0.1:
            _initialPath = initialPath[startPointIndex:]
        else :
            _initialPath = [currentPosition] + initialPath[startPointIndex:]

        print(Fore.YELLOW +  f"Length of the path is {len(_initialPath)}" + Style.RESET_ALL)

        timeStart = time.time()
        try:
            _optimalTrajectory, _newDt, _cost = globalOptimizer.optimize(
                _initialPath, 
                obstacles, 
                maxDistances, 
                prevDt, 
                _currentPosition, 
                _initialPath[-1]
            )
        except Exception as e:
            print(Fore.RED + f"Error during global optimization: {e}" + Style.RESET_ALL)
            continue

        timeEnd = time.time()
        print(Fore.YELLOW + f"Global optimization took {timeEnd - timeStart:.3f} seconds" + Style.RESET_ALL)

        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            _optimalTrajectory, 
            initialPath, 
            obstacles, 
            maxDistances    
        )

        _allOptimalPaths.append((_optimalTrajectory, _newDt, _cost))

        if not anyCollision:
            print(Fore.CYAN + "No Collision found, updating global variables." + Style.RESET_ALL)
            with optimizationLock:
                currentTimeStep = _newDt
                initialTrajectory = _optimalTrajectory
            
            costVariation = np.abs(prevCost - _cost) / _cost
            timeVariation = np.abs(prevDt - _newDt) / _newDt
            
            if costVariation < 0.01 and timeVariation < 0.01:
                print(Fore.GREEN + "Global optimization converged." + Style.RESET_ALL)
                break
            else:
                initialPath = _optimalTrajectory
                prevCost = _cost
                prevDt = _newDt

def localOptimization(
        localOptimizer : LocalOptimalPlanner, 
        environment : EnvironmentHandler,
        robot : Robot, 
        simulator : Simulator,
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
    global currentPosition, currentTimeStep, initialTrajectory, atEnd, finalTrajectory
    _currentPosition = None
    _currentTimeStep = None
    _currentTrajectory = None
    _atEnd = False

    with optimizationLock:
        _currentPosition = currentPosition

    realPath = [(_currentPosition, -1)]

    i = 0 
    while True:
        print(Fore.BLUE + f"Local Optimization Iteration {i}" + Style.RESET_ALL)

        with optimizationLock:
            if _currentTrajectory != initialTrajectory:
                print(Fore.MAGENTA + f"Current trajectory input changed" + Style.RESET_ALL)


            _currentTrajectory = initialTrajectory
            _currentTimeStep = currentTimeStep
            _currentPosition = currentPosition
            _atEnd = atEnd

        if _currentTrajectory is None:
            print(Fore.YELLOW + f"Waiting for the current Trajectory to be set" + Style.RESET_ALL)
            time.sleep(0.1)
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
        


        obstacles, maxDistances, _, _ = robot.getObstacles(
            environment,
            initialLocalTrajectory, 
            [], 
            [], 
            []
        )

        print(Fore.BLUE + f"Len Initial Trajectory: {len(initialTrajectory)} num Obstacles {len(obstacles)} local horizon {localHorizon}" + Style.RESET_ALL)

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
        print(Fore.BLUE + f"Local optimization took {endTime - startTime:.3f} seconds" + Style.RESET_ALL)


        state = simulator.simulate(
            _optimalTrajectory[0].get_state(),
            _optimalTrajectory[0].u,
            _currentTimeStep
        )[:, -1]

        state = OptimizationState(
            x=state[0:3],
            v=state[3:6],
            q=state[6:10],
            w=state[10:13]
        )

        realPath.append((state, _currentTimeStep))

        if np.linalg.norm((state.x - _currentTrajectory[-1].x)) < 0.1:
            print(Fore.GREEN + "Reached the goal node." + Style.RESET_ALL)
            finalTrajectory = realPath
            atEnd = True
            break


        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            [rp[0] for rp in realPath],
            [],
            [],
            []
        )

        print(Fore.BLUE + f"Collision Found in Real Path: {anyCollision}" + Style.RESET_ALL)


        with optimizationLock:


            #drawLOTWithGOTAndFCP(
            #    environment, 
            #    robot, 
            #    _currentTrajectory,
            #    _optimalTrajectory,
            #    realPath
            #)
            currentPosition = state


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

    obstacles, maxDistances, _,  _  = robot.getObstacles(
        environment,
        optimizationPath, 
        [], 
        [], 
        []
    )

    print(f"Number of obstacles detected: {len(obstacles)}")
    print(f"Number of max distances: {len(maxDistances)}")

    numSuccessfulOptimizationsRequired = 10
    optimizationHorizon = 10
    lookAhead = 10
    windowStart = 0
    prevU = np.zeros((6, optimizationHorizon + lookAhead))

    globalOptimalPlanner = GlobalOptimalPlanner(
        stateMinValues=stateLowerBound, 
        stateMaxValues=stateUpperBound,
        env=environment,
        robot=robot,
    )

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

    globalOptimizationThread = Thread(
        target=globalOptimization, 
        args=(
            globalOptimalPlanner,
            environment,
            initialPath,
            robot
        )
    )

    globalOptimizationThread.start()

    localOptimizationThread = Thread(
        target=localOptimization,
        args=(
            localOptimalPlanner,
            environment,
            robot,
            simulator,
            localSize,
        )
    )

    localOptimizationThread.start()
    globalOptimizationThread.join()
    localOptimizationThread.join()


    with open(os.path.join(script_dir, "results.pkl"), "wb") as f:
        pickle.dump({
            "initialPath" : initialPath,
            "realPath" : finalTrajectory, 
            "optimalPaths" : allOptimalPaths,
        }, f)

    return


    globalOptimalPath1 = None
    globalOptimalPath2 = None
    
    globalOptimalPath1, dt, cost = run_global_optimization(
        globalOptimalPlanner, 
        robot, 
        environment,
        initialPath, 
        obstacles, 
        maxDistances, 
        dt, 
        xi, 
        xf
    )

    globalOptimalPath2, dt, cost = run_global_optimization(
        globalOptimalPlanner, 
        robot, 
        environment,
        globalOptimalPath1, 
        obstacles, 
        maxDistances, 
        dt, 
        xi, 
        xf
    )
    
    localPathSize = 10
    realPath = [(globalOptimalPath1[0], dt)]
    for i in range(20):
        currentPositionIndex = min(
            range(len(globalOptimalPath1)),
            key=lambda j: np.linalg.norm(globalOptimalPath1[j].x - realPath[-1][0].x)
        )
    
        localInitialPath = globalOptimalPath1[currentPositionIndex+1:currentPositionIndex+1 + localSize]
        localInitialPath.insert(0, realPath[-1][0])
        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            localInitialPath, 
            None, 
            [], 
            []
        )

        localOptimalTrajectory, _, stepDt = localOptimalPlanner.optimize(
            localInitialPath, 
            obstacles,
            maxDistances, 
            dt, 
            localInitialPath[0],
            localInitialPath[-1], 
            0
        )

        state = simulator.simulate(
            localOptimalTrajectory[0].get_state(), 
            localOptimalTrajectory[0].u, 
            stepDt
        )[:, -1]
        print(f"State after simulation: {state}")
        print(f"Local Optimal Trajectory: {localOptimalTrajectory[2].get_state()}")
        state = OptimizationState(
            x = state[0:3],
            v = state[3:6],
            q = state[6:10],
            w = state[10:13]
        )

        realPath.append((state, stepDt))
    for i in range(60):
        currentPositionIndex = min(
            range(len(globalOptimalPath2)),
            key=lambda j: np.linalg.norm(globalOptimalPath2[j].x - realPath[-1][0].x)
        )

        localInitialPath = globalOptimalPath2[currentPositionIndex+1:currentPositionIndex+1 + localSize]
        localInitialPath.insert(0, realPath[-1][0])
        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            localInitialPath, 
            None, 
            [], 
            []
        )

        localOptimalTrajectory, _, stepDt = localOptimalPlanner.optimize(
            localInitialPath, 
            obstacles,
            maxDistances, 
            dt, 
            localInitialPath[0],
            localInitialPath[-1], 
            0
        )



        state = simulator.simulate(
            localOptimalTrajectory[0].get_state(), 
            localOptimalTrajectory[0].u, 
            stepDt
        )[:, -1]
        state = OptimizationState(
            x = state[0:3],
            v = state[3:6],
            q = state[6:10],
            w = state[10:13]
        )

        realPath.append((state, stepDt))
    drawLOTWithGOTAndFCP(
        environment, 
        robot, 
        globalOptimalPath2,
        localOptimalTrajectory,
        realPath
    )


        
   # with open(os.path.join(script_dir, "optimizedPathNoExpand.pkl"), "wb") as f:
   #     
   #     pickle.dump(initialPath, f)
    
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    
    print(Fore.GREEN + "Start Node : " + Style.RESET_ALL, initialPath[0].get_state())
    print(Fore.GREEN + "Goal Node : " + Style.RESET_ALL, initialPath[-1].get_state())
        
    atGoalNode = False
    goalNode = initialPath[-1]
    localPathSize = 10
    realPath = [(initialPath[0], dt)]
    
    optimizationTimes = []
    obstacleConsidered = []
    i = 0 

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    while not atGoalNode:
        # Do local optimization and simulation
        # Get closest node in global optimal path
        closestNodeIndex = i 
        #print(Fore.YELLOW + f"Iteration {i}" + Style.RESET_ALL)
        closestNodeIndex = min(
            range(len(initialPath)), 
            key=lambda j: np.linalg.norm(initialPath[j].x - realPath[-1][0].x)
        )



        if closestNodeIndex+1 + localPathSize < len(initialPath):
            print(Fore.YELLOW + f"Closest node index: {closestNodeIndex}, local path size: {localPathSize}" + Style.RESET_ALL)
            localInitialPath = initialPath[closestNodeIndex+1:closestNodeIndex+1 + localPathSize - 1] 
            localInitialPath.insert(0, realPath[-1][0])
        else:
            print(Fore.YELLOW + "Not enough nodes in initial path, extending with last node" + Style.RESET_ALL)
            localInitialPath = [realPath[-1][0]]
            localInitialPath.extend(initialPath[closestNodeIndex:])
            diff = (closestNodeIndex + localPathSize) - len(initialPath)
            localInitialPath.extend(diff * [initialPath[-1]])

        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            localInitialPath, 
            None, 
            [], 
            []
        )

        print(f"Any Collision: {anyCollision}")
        print(f"Number of obstacles considered: {len(obstacles)}")
        if anyCollision:
            drawLocalOptimalTrajectory(
                environment,
                realPath, 
                robot, 
                localInitialPath
            )
                

        startTime = time.time()
        localOptimalTrajectory, _, stepDt = localOptimalPlanner.optimize(
            localInitialPath, 
            obstacles,
            maxDistances, 
            dt, 
            localInitialPath[0],
            localInitialPath[-1], 
            0
        )
        endTime = time.time()
        print(f"Step took {endTime - startTime:.3f} seconds")
        

        
        optimizationTimes.append(endTime - startTime)
        obstacleConsidered.append(len(obstacles))

        state = simulator.simulate(localOptimalTrajectory[0].get_state(), localOptimalTrajectory[0].u, stepDt)[:, -1]
        state = OptimizationState(
            x = state[0:3],
            v = state[3:6],
            q = state[6:10],
            w = state[10:13]
        )
        print(Fore.YELLOW + f"i {i} closestNodeIndex {closestNodeIndex}" + Style.RESET_ALL)
        print(Fore.RED + f'Real Next State : {state.get_state()}' + Style.RESET_ALL)
        print(Fore.RED + f'Predicted Next State : {localOptimalTrajectory[1].get_state()}' + Style.RESET_ALL)
        print(Fore.RED + f'Initial Next State : {initialPath[closestNodeIndex + 1].get_state()}' + Style.RESET_ALL)
        print(Fore.YELLOW + f'Initial Actuation : {initialPath[closestNodeIndex].u} real actuation {localOptimalTrajectory[0].u}' )
        i += 1

        realPath.append((state, stepDt))
        collisions = []
        indexes = None

        for j, (p, _) in enumerate(realPath):
            x = p.x
            R = trf.Rotation.from_quat(p.q)
            
            isCollision, depth, pt1, pt2, normal = environment.collide(robot.fcl_obj, x, R)
            
            if isCollision:
                print(f"x : {x} R : {R.as_quat()}")
                print(f"x : {initialPath[j].x} R : {trf.Rotation.from_quat(initialPath[j].q).as_quat()}")
                print(f"localOptimalTrajectory {j} : {localOptimalTrajectory[1].get_state()}")
                print(f"NumericalF {j} : {robot.numericalF(localOptimalTrajectory[0].get_state(), localOptimalTrajectory[0].u, stepDt)}")
                print(f"State in realPath {j} : {p.get_state()}")
                print(f"Collision detected at step {j} with depth {depth}")
                indexes = indexes if indexes is not None else []
                indexes.append(j)
                collisions.append(Obstacle(
                    closestPointObstacle=pt2,
                    closestPointRobot=pt1,
                    distance=-depth,
                    normal=normal,
                    iteration=j,
                ))
            
        if indexes is not None:
            drawRealPathWithObstaclesAndCollisions(
                environment, 
                robot, 
                realPath, 
                collisions, 
                obstacles, 
                indexes, 
                i
            )

        if np.linalg.norm(state.x - goalNode.x) < 0.1:
            atGoalNode = True

    plt.boxplot(optimizationTimes)
    plt.title("Optimization Times")
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Time (seconds)")
    plt.grid()
    plt.show()

    drawLocalOptimalTrajectory(
        environment,
        realPath,
        robot,
        []
    )

    for i in range(50):
        print(f'Num obstacle considered in this step : {len(obstacles)}')

        startTime = time.time()
        optimizedTrajectory, newDt, cost = globalOptimalPlanner.optimize(
            initialPath, 
            obstacles, 
            maxDistances, 
            dt, 
            xi, 
            xf
        )


        # Evaluate the trajectory to ensure it is valid
        obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
            environment, 
            optimizedTrajectory, 
            initialPath, 
            obstacles, 
            maxDistances    
        )
        print(f"Number of obstacles considered: {len(obstacles)}")
        if len(obstacles)  > 500:
            for obs in obstacles:
                print(f"Obstacle {obs.iteration} with closest point {obs.closestPointObstacle} and normal {obs.normal}")
                
        if obstacles is None:
            print(Fore.RED + "No obstacles detected" + Style.RESET_ALL)
            break
        

        optimizedPaths.append((initialPath, optimizedTrajectory, obstacles, collisionObstacles, maxDistances, newDt))

        if not anyCollision:
            firstOptimization = False
            print(Fore.GREEN + 
                  f"time maxValue : 0.01 - {np.abs(newDt - dt) / dt:.5f}\n"
                  f"cost maxValue : 0.1 - {np.abs(cost - prevCost) / prevCost:.5f}" + Style.RESET_ALL,
                  end="\n\n")
            if np.abs(newDt - dt) / dt < 0.01 and (np.abs(cost - prevCost) / prevCost) < 0.1: 
                globalOptimalPlanner.visualizeTrajectory(
                    optimizationPath, 
                    optimizedTrajectory, 
                    environment.voxel_mesh,    
                )
                return
            initialPath = optimizedTrajectory
            dt = newDt
            prevCost = cost
            print(Fore.GREEN + 
                f"Optimization for {len(initialPath)} with "
                f"{len(obstacles)} obstacles successful,"
                f"in {time.time() - startTime:.2f} seconds."
                f"dt {dt} cost {cost:.2f}" + Style.RESET_ALL
            )
        else:
            pass
            #drawCollisions(
            #    environment, 
            #    _optimizationPath, 
            #    optimizationPath, 
            #    collisionObstacles, 
            #    obstacles, 
            #    robot
            #)

            
    with open(os.path.join(script_dir, "optimizedPath.pkl"), "wb") as f:
        pickle.dump(optimizedPaths, f)
    
    print(f"Optimized path saved to {os.path.join(script_dir, 'optimizedPath.pkl')}")


if __name__ == "__main__":
    main()
