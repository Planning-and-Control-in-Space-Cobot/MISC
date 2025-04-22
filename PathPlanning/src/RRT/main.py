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
from RRTOptimization import RRTPathOptimization as RRTOpt, Robot, Obstacle

def get_obstacle_list(path, rrtEnv):
    obstacles = []
    for i in range(len(path)):
        normal = rrtEnv.get_normal_plane(path[i])
        closest_point = rrtEnv.get_closest_points(path[i])[2]
        obstacles.append(Obstacle(normal, closest_point, i))
    
    return obstacles


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
    path_optimizer = RRTOpt(robot)

    original_path = path
    print("🚀 Optimizing path...")

    # Compute normals 


    xf = np.hstack([path[-1].x, path[-1].v, path[-1].q, path[-1].w])
    xi = np.hstack([path[0].x, path[0].v, path[0].q, path[0].w])
    dt = 5
    for i in range(100):
        print(f"Iteration {i}")
        obstacles = get_obstacle_list(path, rrtEnv)

        try: 
            sol = path_optimizer.optimize(
                path,
                obstacles, 
                xi, 
                xf,
                dt
            )
        except RuntimeError as e:
            x = path_optimizer.opti.debug.value(path_optimizer.x)
            u = path_optimizer.opti.debug.value(path_optimizer.u)
            dt = path_optimizer.opti.debug.value(path_optimizer.dt)

            path = [state(x[0:3, i], x[3:6, i], x[6:10, i], x[10:13, i]) for i in range(x.shape[1])]

            obstacles = get_obstacle_list(path, rrtEnv)
            for obstacle in obstacles:
                if np.linalg.norm(obstacle.normal) == 0:
                    print("Normal is zero")
                    i = 101
                    break
            
    path_optimizer.visualize_trajectory(original_path, path, rrtEnv.voxel_mesh)

if __name__ == "__main__":
    main()
