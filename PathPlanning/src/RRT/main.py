import os
import open3d as o3d
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
import scipy.spatial.transform as trf
import pickle
import trimesh as tm

from typing import List, Tuple

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from RRT import RRTPlanner3D as RRTPlanner, RRTState

from Environment import EnvironmentHandler as RRTEnv
from RRTOptimization import Robot, RRTPathOptimization, Obstacle, OptimizationState, TimeStepObstacleConstraints
from MeshSampler import MeshSampler

from time import sleep
def createEllipsoid(a : float, b : float, c : float, subdivisions :int = 3) -> tm.Trimesh:
    '''
    This function creates an ellipsoidal mesh composed of triangle that is encapsulated in a Trimesh object.

    The ellipsoid is considered to be centered at the origin and aligned with the axis, if needed it can be rotated

    Parameters:
        a (float): The radius of the ellipsoid along the x-axis.
        b (float): The radius of the ellipsoid along the y-axis.
        c (float): The radius of the ellipsoid along the z-axis.
        subdivisions (int): The number of subdivisions for the mesh. Default is 10.

    Return:
        tm.Trimesh: A Trimesh object representing the ellipsoid.
    '''
    # Create an icosphere mesh 
    mesh = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    # Scale the vertices to form an ellipsoid 
    vertices = mesh.vertices @ np.diag([a, b, c])
    # Create new mesh with the scaled vertices
    return tm.Trimesh(vertices=vertices, faces=mesh.faces)


def main():
    robotEllipsoid = createEllipsoid(0.24, 0.24, 0.1, 3)

    meshSampler = MeshSampler(robotEllipsoid)
    translations = meshSampler.sample_uniform(50) # This is the translation from the center of the robot to the surface point since ellipsoid is centered at origin and axis aligned

    args = sys.argv[1:] # all but the first argument
    
    use_saved_path = len(args) == 0 or (len(args) == 1 and args[0] != "new")
    
    point_cloud_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "map.pcd")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    rrtEnv = RRTEnv(pcd)
    
    if use_saved_path:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "path.pkl")):
            raise ValueError("Path file does not exist, please run with 'new' to generate a new path")
        with open(os.path.join(os.path.dirname(__file__), "path.pkl"), "rb") as f:
            path = pickle.load(f)
    else:


        start = RRTState(np.array([0, -3, 1.0]), np.array([0, 0, 0, 1]))
        goal = RRTState(np.array([20.0, 15.0, 1.0]), np.array([0, 0, 0, 1]))
        planner = RRTPlanner(rrtEnv)
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

    robot = Robot(J, A, m, fcl_obj, translations)
    stateLowerBound = np.array([-25, -25, 0, -5, -5, -5, -1, -1, -1, -1, -2, -2, -2])
    stateUpperBound = np.array([25, 25, 2, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2])
    
    rrtOpt = RRTPathOptimization(stateLowerBound, stateUpperBound, rrtEnv, robot)
    xi = np.zeros(13)
    xf = np.zeros(13)

    xi[0:3] = path[0].x
    xi[6:10] = path[0].q

    xf[0:3] = path[-1].x
    xf[6:10] = path[-1].q


    def getObstacles(path : List[OptimizationState], surfacePoints : np.ndarray, rrtEnv : RRTEnv): 
        point =  rrtEnv.buildSinglePoint()
        obstacleSets = []
        for i, p in enumerate(path[1:-1]):
            obstacleTimeStep = []
            for j, translation in enumerate(surfacePoints):
                x = p.x + translation # Sum the translation to the robot position
                q = p.q
                R = trf.Rotation.from_quat(q)
                d, _, p2 = rrtEnv.getClosestPoints(point, x, q)
                normal = x - p2

                obstacleTimeStep.append(Obstacle(j, normal, x, p2))
            obstacleSets.append(TimeStepObstacleConstraints(i+1, surfacePoints.shape[1], obstacleTimeStep))
        return obstacleSets        
        
    originalPath = path
    obstacles = getObstacles(path, translations, rrtEnv)
    rrtOpt.setup_optimization(path, obstacles, xi, xf) 
    dt = 0.5
    prev_u = np.zeros((6, len(path)))
    for i in range (1):
        print(f"Iteration {i}")
        obstacles = getObstacles(path, translations, rrtEnv)
        if obstacles is None:
           exit(1) 
        sol = rrtOpt.optimize(path, obstacles, dt, prev_u)        
        path, prev_u, dt  = rrtOpt.getSolution(sol)

    plt = rrtOpt.visualize_trajectory(originalPath, path, rrtEnv.voxel_mesh)
    def format_vec(v, width=6, prec=3):
        return "[" + " ".join(f"{x:{width}.{prec}f}" for x in v) + "]"
    def compute_plane_distance(collision_point, prevEnvPoint, normal):
        num = np.dot(collision_point - prevEnvPoint, normal)
        denom = np.linalg.norm(normal)
        return num / denom if denom > 1e-12 else 0.0  # avoid div by 0

    print(f"dt = {dt}")
    np.set_printoptions(
        linewidth=200,           # Increase max characters per line
        suppress=True,           # Suppress scientific notation
        precision=3,            # Set precision for floats
    )
    plt.show()


    


if __name__ == "__main__":
    main()
