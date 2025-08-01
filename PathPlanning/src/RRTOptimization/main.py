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
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import pickle

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from Environment import EnvironmentHandler as EnvironmentHandler

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
    try:
        _allOptimalPaths = []

        prevCost = 0
        prevDt = 0.2
        
        collisionObstacles = []
        i=0

        print(f"State Min Values: {stateLowerBound}")
        print(f"State Max Values: {stateUpperBound}")

        environment = EnvironmentHandler(
            pcdPath
        )

        pv_ = pv.Plotter()
        pv_ = environment.visualizeCoalMesh(pv_)
        for obs in environment.pyvistaMeshes:
            pv_.add_mesh(obs, color='white', show_edges=True, opacity=1)
        pv_.add_axes()
        pv_.show()



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

        newCollisionsObstacles = []
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
                
            if _currentPosition is None:
                print(Fore.YELLOW + f"Waiting for the current position to be set" + Style.RESET_ALL)
                time.sleep(0.1)
                continue

            startPointIndex = min(
                range(len(initialPath)), 
                key=lambda j: np.linalg.norm(initialPath[j].x - _currentPosition.x)
            )

            _initialPath = [_currentPosition] + initialPath[startPointIndex + 1:]

            obstacles, maxDistances = robot.getObstacles(environment, _initialPath)

            print(Fore.RED + f"Num Collision Obstacles: {len(newCollisionsObstacles)}" + Style.RESET_ALL)
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
                print(Fore.RED + f"{type(e)}" + Style.RESET_ALL)
                print(Fore.RED + f"{e}")
                continue

            timeEnd = time.time()
            print(Fore.YELLOW + f"Global optimization took {timeEnd - timeStart:.3f} seconds" + Style.RESET_ALL)

            collisionFree = robot.collisionFree(_optimalTrajectory, environment)

            _allOptimalPaths.append((_optimalTrajectory, _newDt, _cost))

            if collisionFree:
                newCollisionsObstacles = []
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
                newCollisionsObstacles.extend(robot.getCollision(environment, _optimalTrajectory))
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
                for no in newCollisionsObstacles:
                    print(f"Obstacle at iteration {no.iteration} with distance {no.minDistance} normal {no.normal}")
                
                #drawEnvironmentWithNormals(
                #    environment, 
                #    _optimalTrajectory, 
                #    newCollisionsObstacles, 
                #    robot
                #)

                print(newCollisionsObstacles)
                print(Fore.RED + "Collision found, retrying global optimization." + Style.RESET_ALL)
                pass
    except Exception as e:
        print(Fore.RED + f"An error occurred during global optimization: {e}" + Style.RESET_ALL)

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
        pcdPath
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

    sharedData["localPlanner"] = []
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

        distance = np.linalg.norm(
            _currentTrajectory[startPointIndex].x - _currentPosition.x
        )

        print(f"Start Point Index {startPointIndex} distance {distance} {_currentPosition.x}")

        if distance > 0.5:
            trajectory, fullTrajectory = generateTransitionTrajectory(
                _currentPosition, 
                _currentTrajectory,
                _currentTimeStep
            )
            """

            with lock:
                pv_ = pv.Plotter()
                pv_ = environment.visualizeCoalMesh(pv_)

                for state in realPath:
                    x = state[0].x
                    R = trf.Rotation.from_quat(state[0].q)
                    robotMesh = robot.getPVMesh(x, R)
                    pv_.add_mesh(
                        robotMesh,
                        color="green",
                        show_edges=True,
                        opacity=0.5,
                    )
                
                for state in _currentTrajectory:
                    x = state.x
                    R = trf.Rotation.from_quat(state.q)
                    robotMesh = robot.getPVMesh(x, R)
                    pv_.add_mesh(
                        robotMesh,
                        color="blue",
                        show_edges=True,
                        opacity=0.5,
                    )
                
                for state in trajectory:
                    x = state.x
                    R = trf.Rotation.from_quat(state.q)
                    robotMesh = robot.getPVMesh(x, R)
                    pv_.add_mesh(
                        robotMesh,
                        color="red",
                        show_edges=True,
                        opacity=0.5,
                    )
                
                pv_.add_axes()
                pv_.show_grid()
                pv_.add_text("Current Position", position="upper_left", color="green")
                pv_.add_text("Current Trajectory", position="upper_right", color="blue")
                pv_.add_text("Transition Trajectory", position="lower_left", color="red")
                pv_.show()
                """
            initialLocalTrajectory = trajectory
        else:
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

        

        startTime = time.time()
        try:
            _optimalTrajectory, _, _ = localOptimizer.optimize(
                initialLocalTrajectory,
                obstacles, 
                maxDistances, 
                _currentTimeStep, 
                _currentPosition, 
                _currentTrajectory[-1],
                0
            )
        except Exception as e:
            sharedData["localPlanner"].append(
                {"obstacles": obstacles, "maxDistances": maxDistances, 
                "trajectory": initialLocalTrajectory, "currentPosition": _currentPosition,
                "globalPath" : _currentTrajectory, "currentTimeStep": _currentTimeStep, "optimalTrajectory": None , "realTrajectory": realPath} 
            )
            print(Fore.RED + f"Local optimization failed: {e}" + Style.RESET_ALL)
            with lock:
                sharedData["atEnd"] = True
                sharedData["finalTrajectory"] = realPath
            continue
        if _optimalTrajectory is not None:
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
            sharedData["localPlanner"].append(
                {"obstacles": obstacles, "maxDistances": maxDistances, 
                "trajectory": initialLocalTrajectory, "currentPosition": _currentPosition,
                "globalPath" : _currentTrajectory, "currentTimeStep": _currentTimeStep, "optimalTrajectory": _optimalTrajectory, "realTrajectory": realPath} 
            )

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

def generateTransitionTrajectory(
        currentState : OptimizationState,
        globalOptimalPath : List[OptimizationState], 
        dt :  float
) -> List[OptimizationState]:
    def generate_transition_trajectory(current_state, target_state, N=10):
        """
        Generate a smooth trajectory between two states using cubic interpolation for position,
        SLERP for orientation, and linear interpolation for velocities.

        Parameters:
            current_state (OptimizationState): Initial state with (x, v, q, w)
            target_state (OptimizationState): Final state to reach
            N (int): Number of steps in the trajectory (including start and end)

        Returns:
            List[OptimizationState]: List of interpolated states
        """
        # Normalize time from 0 to 1
        ts = np.linspace(0, 1, N)

        # Position and velocity interpolation (cubic)
        x0, v0 = np.array(current_state.x), np.array(current_state.v)
        x1, v1 = np.array(target_state.x), np.array(target_state.v)

        a0 = x0
        a1 = v0
        a2 = 3*(x1 - x0) - 2*v0 - v1
        a3 = -2*(x1 - x0) + v0 + v1

        positions = np.array([a0 + a1*t + a2*t**2 + a3*t**3 for t in ts])
        velocities = np.array([a1 + 2*a2*t + 3*a3*t**2 for t in ts])

        # Orientation SLERP
        q0 = np.array(current_state.q)
        q1 = np.array(target_state.q)
        slerp = Slerp([0, 1], R.from_quat([q0, q1]))
        quaternions = slerp(ts).as_quat()

        # Angular velocity interpolation (linear)
        w0 = np.array(current_state.w)
        w1 = np.array(target_state.w)
        angular_velocities = np.linspace(w0, w1, N)

        # Build trajectory
        trajectory = []
        for i in range(N):
            traj_state = OptimizationState(
                x=positions[i],
                v=velocities[i],
                q=quaternions[i],
                w=angular_velocities[i],
                u=np.zeros(6),
                i=i
            )
            trajectory.append(traj_state)

        return trajectory

    closestPointIndex = min(
        range(len(globalOptimalPath)),
        key=lambda i: np.linalg.norm(globalOptimalPath[i].x - currentState.x)
    )
    print(Fore.BLUE + f"ClosestPointIndex: {closestPointIndex} vs len{len(globalOptimalPath)}" + Style.RESET_ALL)
    nextState = globalOptimalPath[closestPointIndex + 1]

    distanceNextState = np.linalg.norm(nextState.x - currentState.x)
    numStatesPos = int(np.ceil(distanceNextState / (0.1)))

    q_rel = trf.Rotation.from_quat(currentState.q).inv() * trf.Rotation.from_quat(nextState.q)
    numStatesAttitude = int(np.ceil(np.degrees(q_rel.magnitude()) / 15))
    numStates = max(numStatesPos, numStatesAttitude)

    print(f"Num States position {numStatesPos}, numStatesAttitude {numStatesAttitude}")

    interpolatedStates = generate_transition_trajectory(
        currentState, 
        nextState, 
        numStates
    )

    interpolatedStates.extend(globalOptimalPath[closestPointIndex + 1:])
    finalTrajectory = interpolatedStates.copy()

    if len(interpolatedStates) < 10:
        interpolatedStates.extend([globalOptimalPath[-1]] * (10 - len(interpolatedStates)))
    else:
        interpolatedStates = interpolatedStates[:10]

    for i, state in enumerate(interpolatedStates):
        print(type(state))
    return interpolatedStates, finalTrajectory

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

def drawEnvironmentWithNormals(environment, path, obstacles, robot):
    pv_ = pv.Plotter()

    envMeshes = environment.pyvistaMeshes
    for mesh in envMeshes:
        pv_.add_mesh(mesh, color='white', show_edges=True, opacity=0.5)
    
    #for p in path:
    #    x = p.x
    #    q = p.q

    #    T = np.eye(4)
    #    T[:3, :3] = trf.Rotation.from_quat(q).as_matrix()
    #    T[:3, 3] = x
    #    cube = robot.getPVMesh(x, trf.Rotation.from_quat(q))
    #    pv_.add_mesh(cube, color='blue', show_edges=True)
    
    for obs in obstacles:
        plane = pv.Plane(
            center=obs.closestPointObstacle,
            direction=obs.normal,
            i_size=0.1,
            j_size=0.1,
        )
    
        arrow = pv.Arrow(
            start=obs.closestPointObstacle,
            direction=obs.normal,
            scale=1.1,
            tip_length=0.05,
        )
        pv_.add_mesh(plane, color='red', show_edges=True, opacity=0.5)
        pv_.add_mesh(arrow, color='green', show_edges=True, opacity=0.5)

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

    environmentPath = os.path.join(script_dir, args.map)
    environment = EnvironmentHandler(environmentPath)

    path = np.load(os.path.join(script_dir, args.path))
    originalPosition = path["positions"]
    originalOrientation = path["orientations"]
    initialPath = [
        OptimizationState(originalPosition[i], originalOrientation[i])
        for i in range(len(originalPosition))
    ]
    A = np.load(os.path.join(script_dir, "A_matrix.npy"))
    J = np.load(os.path.join(script_dir, "J_matrix.npy"))
    m = np.load(os.path.join(script_dir, "mass.npy"))
    robot = Robot(
        J,
        A,
        m,
    )

    print("Setting up the optimization parameters")

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

    stateLowerBound = np.hstack([
        np.array([-1.5, -1.5, -2.5]),  # x, y, z
        minV,
        np.array([-1, -1, -1, -1]), 
        minW,
    ])

    stateUpperBound = np.hstack([
        np.array([2.5, 2.5, 5.0]),  # x, y, z
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
            environmentPath,
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
            environmentPath,
            J,
            A,
            m,
            simulator,
            sharedData,
            lock,
            localSize,
        )
    )

    try:
        globalOptimizationProcess.start()
        localOptimizationProcess.start()
        globalOptimizationProcess.join()
        localOptimizationProcess.join()
    except KeyboardInterrupt:
        print(Fore.RED + "\nKeyboardInterrupt detected. Terminating processes..." + Style.RESET_ALL)
        globalOptimizationProcess.terminate()
        localOptimizationProcess.terminate()
        globalOptimizationProcess.join()
        localOptimizationProcess.join()
    finally:
        print(Fore.YELLOW + "Saving results to results.pkl..." + Style.RESET_ALL)
        try:
            with open(os.path.join(script_dir, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "initialPath": initialPath,
                        "realPath": sharedData.get("finalTrajectory", []),
                        "optimalPaths": sharedData.get("allOptimalPaths", [])
                    },
                    f,
                )
            print(Fore.GREEN + "Results successfully saved." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Failed to save results: {e}" + Style.RESET_ALL)

    


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
