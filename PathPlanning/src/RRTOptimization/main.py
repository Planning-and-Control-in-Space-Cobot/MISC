import os
import sys
import time 
import argparse
from colorama import Fore, Style

import numpy as np
import open3d as o3d
import pyvista as pv
import scipy.spatial.transform as trf
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

def drawCollisions(environment, path, pPath, collisionObstacles, pObstacles, robot):


    for c in collisionObstacles:
        pv_ = pv.Plotter()
        pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)
        i = c.iteration

        plane = pv.Plane(
            center=c.closestPointObstacle,
            direction=c.normal
        )
        arrow = pv.Arrow(
            start=c.closestPointObstacle,
            direction=c.normal,
            scale=.1,
            tip_length=.2,
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
        
        p =  path[i]
        R = trf.Rotation.from_quat(p.q)
        robotMesh = robot.getPVMesh(p.x, R)
        pv_.add_mesh(
            robotMesh,
            color="blue",
            show_edges=True,
            opacity=0.5,
        )

        pp = pPath[i]
        R = trf.Rotation.from_quat(pp.q)
        robotMesh = robot.getPVMesh(pp.x, R)
        pv_.add_mesh(
            robotMesh,
            color="green",
            show_edges=True,
            opacity=0.5,
        )

        pObs = [pObs for pObs in pObstacles if pObs.iteration == i]
        
        for po in pObs:
            for v in robot.getVertices():
                print(f"Vertex {v} : "
                f"{po.normal @ (trf.Rotation.from_quat(p.q).apply(v) + p.x)} :"
                f"{po.normal @ po.closestPointObstacle + po.safetyMargin} : "
                f"{(po.normal @ (trf.Rotation.from_quat(p.q).apply(v) + p.x)) - (po.normal @ po.closestPointObstacle + po.safetyMargin)}")
            print("\n")

            plane = pv.Plane(
                center=po.closestPointObstacle,
                direction=po.normal
                
            )
            arrow = pv.Arrow(
                start=po.closestPointObstacle,
                direction=po.normal,
                scale=0.1,
                tip_length=0.2,
            )
            pv_.add_mesh(
                plane,
                color="orange",
                show_edges=True,
                line_width=1.0,
            )

            pv_.add_mesh(
                arrow,
                color="orange",
                show_edges=True,
                line_width=1.0,
            )
        pv_.add_axes()
        pv_.show_grid()
        pv_.show()

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
    """
    while len(optimizedPaths) < len(initialPath):
        numSuccessfulOptimizations =  0
        firstRun = True
        while numSuccessfulOptimizations < numSuccessfulOptimizationsRequired:
            print(f"Optimizing path from step {windowStart} to {windowStart + optimizationHorizon} with {windowStart + optimizationHorizon + lookAhead} lookahead steps")
            if windowStart + optimizationHorizon + lookAhead < len(initialPath):
                initialTrajectory = optimizationPath[windowStart:windowStart + optimizationHorizon + lookAhead]
            else:
                diff = (windowStart + optimizationHorizon + lookAhead) - len(initialPath)
                initialTrajectory = optimizationPath[windowStart:]
                initialTrajectory.extend(diff * [initialPath[-1]])

            if firstRun:
                obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
                    environment, 
                    initialTrajectory,
                    optimizationPath, 
                    obstacles, 
                    maxDistances
                )
                firstRun = False
            xi = optimizedPaths[-1].get_state()
            xf = initialTrajectory[-1].get_state()
            print(f"xi: {xi}")
            print(f"xf: {xf}")
            print(f"InitialTrajectory {initialTrajectory[0].get_state()}")

            rrtOpt.setup_optimization(
                initialTrajectory, 
                obstacles, 
                maxDistances, 
                prevU, 
                xi=xi, 
                xf=xf
            )

            sol = rrtOpt.optimize(
                initialTrajectory, 
                dt,
                prevU
            )

            if  sol is None:
                print(Fore.RED + "Optimization failed, trying again..." + Style.RESET_ALL)
                exit(1)
            
            _optimizationPath, _prev_u, _dt, cost = rrtOpt.getSolution(sol)
            print(f"Cost of the optimized path: {cost:.2f}")

            # Evaluate the trajectory to ensure it is valid
            print(f"Obstacles considered in this step: {len(obstacles)}")
            obstacles, maxDistances, anyCollision, collisionObstacles = robot.getObstacles(
                environment, 
                _optimizationPath, 
                optimizationPath, 
                obstacles, 
                maxDistances    
            )
            print(f"Number of obstacles considered: {len(obstacles)}")

            if not anyCollision:
                numSuccessfulOptimizations += 1
                print(Fore.GREEN + f"Optimization successful, number of successful optimizations: {numSuccessfulOptimizations}" + Style.RESET_ALL)
                print(f"Len of path len {len(optimizationPath)}")
                optimizationPath[windowStart:windowStart + optimizationHorizon + lookAhead] = _optimizationPath
                print(f"Len of path after optimization {len(optimizationPath)}")

                prevU = _prev_u
                dt = _dt
                #for i, o in enumerate(_optimizationPath):
                #    print(f"{i} - {o.get_state()}")
                #optimizedPaths.append((_optimizationPath, _prev_u, _dt, cost))

                if numSuccessfulOptimizations >= numSuccessfulOptimizationsRequired:
                    firstRun = True

                    optimizedPaths.extend(_optimizationPath[0:optimizationHorizon])
                    if windowStart % 5 == 0:
                        pv_ = rrtOpt.visualize_trajectory(
                            initialPath,
                            optimizedPaths, 
                            environment.voxel_mesh,
                            None, 
                            [],
                            [initialTrajectory[-1].i]
                        )
                        pv_.show()

                    print(Fore.GREEN + "Optimization successful, moving to the next step" + Style.RESET_ALL)
                    windowStart += optimizationHorizon
                    prevU[:, 0:lookAhead] = prevU[:, :lookAhead]
                    prevU[:, :lookAhead] = np.zeros((6, lookAhead))

            else:
                pv_ = rrtOpt.visualize_trajectory(
                    initialPath, 
                    _optimizationPath, 
                    environment.voxel_mesh, 
                    None,
                )
                pv_.show()
                print(Fore.RED + "Collision detected, trying again..." + Style.RESET_ALL)
                """
                #drawEnvironmentAndNormals(environment, collisionObstacles, robot, _optimizationPath)

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
        """
        if not firstOptimization:
            collision = []
            for i in range(len(initialPath) - localSize):
                localInitialPath = initialPath[i:+i+localSize]
                localXi = localInitialPath[0]
                localXf = localInitialPath[-1]
                localObstacles = [obs for obs in obstacles if obs.iteration < i + localSize and obs.iteration >= i]

                print(Fore.YELLOW + f"Local optimization for step {i} with {len(localObstacles)} obstacles" + Style.RESET_ALL)


                localTime = time.time()
                localOptimizdetrajectory, localDt, localCost = localOptimalPlanner.optimize(
                    localInitialPath, 
                    localObstacles,
                    maxDistances[i:i+localSize],
                    newDt, 
                    localXi, 
                    localXf, 
                    i
                )
                print(Fore.YELLOW + f"Local optimization took {time.time() - localTime:.3f} seconds" + Style.RESET_ALL)
                
                _, _, localAnyCollision, _ = robot.getObstacles(
                    environment, 
                    localOptimizdetrajectory, 
                    initialPath, 
                    obstacles, 
                    maxDistances    
                )
                if localAnyCollision:
                    collision.append(i)

            print(Fore.RED + f"Collisions detected in start {collision}" + Style.RESET_ALL)
            print(Fore.GREEN + f"Good local trajectory in {[i for i in range(len(initialPath) - localSize) if i not in collision]}" + Style.RESET_ALL)
        """

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
