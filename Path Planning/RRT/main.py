import os
import sys
import numpy as np 
import open3d as o3d

# Add the executable directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from RRT import RRTPlanner3D as RRTPlanner
from RRT import CollisionEnvironment3D as RRTEnv
from RRT import State as state
from RRTOptimization import RRTPathOptimization as RRTOpt, Robot

def main():
    point_cloud_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.pcd")
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    rrtEnv = RRTEnv(pcd)
    planner = RRTPlanner(rrtEnv)
    planner.run()

    path = planner.get_path()

    if path is None: 
        exit()

    A = np.load(os.path.join(os.path.dirname(__file__), "A_matrix.npy"))
    J = np.load(os.path.join(os.path.dirname(__file__), "J_matrix.npy"))
    m = np.load(os.path.join(os.path.dirname(__file__), "mass.npy"))
    ellipsoid_radius = np.load(os.path.join(os.path.dirname(__file__), "ellipsoidal_radius.npy"))
    robot = Robot(J, A, m, ellipsoid_radius)


    original_path = path
    path_optimizer = RRTOpt (robot)
    path_optimizer.setup_optimization(path)
    print("🚀 Optimizing path...")

    # Compute normals 
    normals = np.array([rrtEnv.get_normal_plane(p) for p in path]).T

    closest_points = np.array([rrtEnv.get_closest_points(p)[2] for p in path]).T
    xf = np.hstack([path[-1].x, path[-1].v, path[-1].q, path[-1].w])
    dt = 5
    for i in range(200):
        print(f"Iteration {i}")
        try:
            sol = path_optimizer.step(normals, closest_points, xf, path, dt)
        except RuntimeError as e:
            x = path_optimizer.opti.debug.value(path_optimizer.x)
            u = path_optimizer.opti.debug.value(path_optimizer.u)
            dt = path_optimizer.opti.debug.value(path_optimizer.dt)

            
            path = []
            path = [state(x[0:3, i], x[3:6, i], x[6:10, i], x[10:13, i]) for i in range(x.shape[1])]

            #path_optimizer.visualize_trajectory(original_path, path, rrtEnv.voxel_mesh)
            normals = np.array([rrtEnv.get_normal_plane(p) for p in path]).T
            closest_points = np.array([rrtEnv.get_closest_points(p)[2] for p in path]).T
            xf = np.hstack([path[-1].x, path[-1].v, path[-1].q, path[-1].w])
            continue
            
    #print(f"Delta time in the trajectory is {dt}")
    path_optimizer.visualize_trajectory(original_path, path, rrtEnv.voxel_mesh)

    

if __name__ == "__main__":
    main()
