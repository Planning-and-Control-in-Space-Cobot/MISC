import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
import scipy.spatial.transform as trf
import pickle

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from RRT import RRTPlanner3D as RRTPlanner, RRTState

from Environment import EnvironmentHandler 
from RRTOptimization import Robot, RRTPathOptimization, Obstacle, OptimizationState

from time import sleep

import matplotlib.pyplot as mplt

def main():
    args = sys.argv[1:] # all but the first argument
    
    use_saved_path = len(args) == 0 or (len(args) == 1 and args[0] != "new")
    onlyRRT = len(args) == 0 or (len(args) == 1 and args[0] == "rrt")
    print(f"Use saved path: {use_saved_path}, Only RRT: {onlyRRT}")
    
    point_cloud_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "map.pcd")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    rrtEnv = EnvironmentHandler(pcd)

    if onlyRRT:
        start = RRTState(np.array([0, -3, 1.0]), np.array([0, 0, 0, 1]))
        goal = RRTState(np.array([20.0, 15.0, 1.0]), np.array([0, 0, 0, 1]))
        robot = rrtEnv.buildEllipsoid()
        planner = RRTPlanner(rrtEnv, robot)
        path = planner.plan(start, goal)
        if path == []:
            print("No path found")
            return
        for s in path:
            print(f"x: {s.x}, q: {s.q}")
        planner.visualize_path(path)
        exit()
    

    if use_saved_path:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "path.pkl")):
            raise ValueError("Path file does not exist, please run with 'new' to generate a new path")
        with open(os.path.join(os.path.dirname(__file__), "path.pkl"), "rb") as f:
            path = pickle.load(f)
    else:
        start = RRTState(np.array([0, -3, 1.0]), np.array([0, 0, 0, 1]))
        goal = RRTState(np.array([20.0, 15.0, 1.0]), np.array([0, 0, 0, 1]))
        robot = rrtEnv.buildEllipsoid()
        planner = RRTPlanner(rrtEnv, robot)
        path = planner.plan(start, goal)

        with open(os.path.join(os.path.dirname(__file__), "path.pkl"), "wb") as f:
            pickle.dump(path, f)

    if path == []:
        return
    path = [OptimizationState(p.x, p.q) for p in path]
    A = np.load(os.path.join(os.path.dirname(__file__), "A_matrix.npy"))
    J = np.load(os.path.join(os.path.dirname(__file__), "J_matrix.npy"))
    m = np.load(os.path.join(os.path.dirname(__file__), "mass.npy"))
    fcl_obj = rrtEnv.buildEllipsoid()

    robot = Robot(J, A, m, fcl_obj)
    stateLowerBound = np.array([-25, -25, 0, -5, -5, -5, -1, -1, -1, -1, -2, -2, -2])
    stateUpperBound = np.array([25, 25, 2, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2])
    
    rrtOpt = RRTPathOptimization(stateLowerBound, stateUpperBound, rrtEnv, robot)
    xi = np.zeros(13)
    xf = np.zeros(13)

    xi[0:3] = path[0].x
    xi[6:10] = path[0].q

    xf[0:3] = path[-1].x
    xf[6:10] = path[-1].q

    def getObstacles(path):
        obstacles = []
        for i, p in enumerate(path[1:-1]):
            x = p.x
            q = p.q
            R = trf.Rotation.from_quat(q)
            if rrtEnv.numCollisions(fcl_obj, x, q) > 0:
                return None 

            _, p_obj, p_env = rrtEnv.getClosestPoints(robot.fcl_obj, x, q)
            translation = p_obj - x
            translation = R.as_matrix().T @ translation
            obstacle = Obstacle(p_obj, p_env, translation, i) 
            obstacles.append(obstacle)
        
        return obstacles
    obstacles = getObstacles(path)
    rrtOpt.setup_optimization(path, obstacles, xi, xf) 
    dt = 1
    prev_u = np.zeros((6, len(path)))
    originalPath = path.copy()
    for i in range (1):
        print(f"Iteration {i}")
        obstacles = getObstacles(path)
        if obstacles is None:
           exit(1) 
        sol = rrtOpt.optimize(path, obstacles, dt, prev_u)        
        path, prev_u, u  = rrtOpt.getSolution(sol)

    plt = rrtOpt.visualize_trajectory(originalPath, path, rrtEnv.voxel_mesh)
    def format_vec(v, width=6, prec=3):
        return "[" + " ".join(f"{x:{width}.{prec}f}" for x in v) + "]"
    def compute_plane_distance(collision_point, prevEnvPoint, normal):
        num = np.dot(collision_point - prevEnvPoint, normal)
        denom = np.linalg.norm(normal)
        return num / denom if denom > 1e-12 else 0.0  # avoid div by 0


    for i, p in enumerate(path):
        x = p.x
        q = p.q 
        R = trf.Rotation.from_quat(q)
        p_x = path[i].x
        num_cols = rrtEnv.numCollisions(fcl_obj, x, q)
        if num_cols != 0:
            collision_point = rrtEnv.getCollisionPoints(fcl_obj, x, q)
            normal = obstacles[i].normal
            prevEnvPoint = obstacles[i].closestPointObstacle
            prevRobotPoint = obstacles[i].closestPointRobot
            distancePrevRobotPointToCollisionPoint = np.linalg.norm(collision_point - prevRobotPoint)
            prevRobotPointPosition = obstacles[i].closestPointRobot + R.apply(obstacles[i].translation)

            start_pos = p_x
            end_pos = x 
            distance_ = np.linalg.norm(start_pos - end_pos)

            lhs = np.dot(normal, x + R.apply(obstacles[i].translation))
            rhs = np.dot(normal, prevEnvPoint) + obstacles[i].safetyMargin
            print(f"i {i} lhs: {lhs:.2f} rhs: {rhs:.2f} diff: {lhs - rhs:.2f}")

            #print(
            #    f"num collisions: {i:<2}  {num_cols} " 
            #    f"normal = {format_vec(normal)} "
            #    f"collision Point = {format_vec(collision_point)} -{distancePrevRobotPointToCollisionPoint} { prevRobotPointPosition} "
            #    f"prevEnvPoint = {format_vec(prevEnvPoint)} "
            #    f"obsacleMinDistance = {obstacles[i].minDistance:.2f} "
            #    f"mov_distance = {distance_:.6f} "
            #)

            plane = pv.Plane(prevEnvPoint, normal)
            plt.add_mesh(plane, color="yellow")

    np.set_printoptions(
        linewidth=200,           # Increase max characters per line
        suppress=True,           # Suppress scientific notation
        precision=3,            # Set precision for floats
    )

    # Extract position components
    x_ = [p.x[0] for p in path]
    x__ = [p.x[0] for p in originalPath]

    y_ = [p.x[1] for p in path]
    y__ = [p.x[1] for p in originalPath]

    z_ = [p.x[2] for p in path]
    z__ = [p.x[2] for p in originalPath]

    # Extract quaternion components
    qx_ = [p.q[0] for p in path]
    qx__ = [p.q[0] for p in originalPath]

    qy_ = [p.q[1] for p in path]
    qy__ = [p.q[1] for p in originalPath]

    qz_ = [p.q[2] for p in path]
    qz__ = [p.q[2] for p in originalPath]

    qw_ = [p.q[3] for p in path]
    qw__ = [p.q[3] for p in originalPath]

    # Create subplots
    mplt.figure(figsize=(10, 14))

    # X
    mplt.subplot(7, 1, 1)
    mplt.plot(x_, label="optimized path")
    mplt.plot(x__, label="original path")
    mplt.ylabel("x")
    mplt.title("X over iterations")
    mplt.legend()

    # Y
    mplt.subplot(7, 1, 2)
    mplt.plot(y_, label="optimized path")
    mplt.plot(y__, label="original path")
    mplt.ylabel("y")
    mplt.title("Y over iterations")
    mplt.legend()

    # Z
    mplt.subplot(7, 1, 3)
    mplt.plot(z_, label="optimized path")
    mplt.plot(z__, label="original path")
    mplt.ylabel("z")
    mplt.title("Z over iterations")
    mplt.legend()

    # Qx
    mplt.subplot(7, 1, 4)
    mplt.plot(qx_, label="optimized path")
    mplt.plot(qx__, label="original path")
    mplt.ylabel("qx")
    mplt.title("Qx over iterations")
    mplt.legend()

    # Qy
    mplt.subplot(7, 1, 5)
    mplt.plot(qy_, label="optimized path")
    mplt.plot(qy__, label="original path")
    mplt.ylabel("qy")
    mplt.title("Qy over iterations")
    mplt.legend()

    # Qz
    mplt.subplot(7, 1, 6)
    mplt.plot(qz_, label="optimized path")
    mplt.plot(qz__, label="original path")
    mplt.ylabel("qz")
    mplt.title("Qz over iterations")
    mplt.legend()

    # Qw
    mplt.subplot(7, 1, 7)
    mplt.plot(qw_, label="optimized path")
    mplt.plot(qw__, label="original path")
    mplt.xlabel("iteration")
    mplt.ylabel("qw")
    mplt.title("Qw over iterations")
    mplt.legend()

    # Layout
    mplt.tight_layout()
    mplt.show()


    plt.show()


    


if __name__ == "__main__":
    main()
