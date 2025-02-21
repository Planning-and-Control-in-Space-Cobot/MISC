from matplotlib.colors import TwoSlopeNorm
import sdf2d 

from shapely.geometry import Point, Polygon, LineString
import numpy as np

from Map import Map
from sdf2d import SDF
from rrt import PathPlanning
from path_optimizer import TrajectoryOptimizer

import matplotlib.pyplot as plt

def main():
    # Define environment limits
    x_size = 200
    y_size = 200

    start = (10, 10)
    goal = (195, 195)

    # Define obstacles using shapely
    obstacles = [
        Point(50, 50).buffer(30),
        Point(150, 150).buffer(35),
        Polygon([(0, 100), (80, 100), (80, 110), (0, 110)]),
        Polygon([(100, 0), (100, 100), (120, 100), (120, 0)])
    ]
    
    map = Map(x_size, y_size,start, goal, obstacles=obstacles)
    #map.plot_map()

    sdf = SDF(map)
    #sdf.plot_sdf()

    planner = PathPlanning(map)
    final_path, explored_paths = planner.rrt()

    np.save("initial_path.npy", final_path)
    map.plot_trajectory(final_path)


    # Trajectory Optimization
    print("Creating Trajectory Optimizer")
    optimizer = TrajectoryOptimizer(initial_path=final_path, SDF=sdf, map=map)

    print("Optimizing Trajectory")
    sol, x, u, t = optimizer.optimize()
    print(t)

   # if sol is not None:
   #     map.plot_trajectory(x.T)
   #     for val in x.T:
   #         print(val)


   # np.save("x_matrix", x.T)

    np.save("sdf.npy", sdf.sdf)
    np.save("map.npy", map.map)

    # Plot the SDF and Gaussian-filtered SDF side by side
    plt.figure(figsize=(16, 6))
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=80)
    cmap = plt.cm.bwr_r

    # Plot the original SDF
    plt.title("Signed Distance Field (SDF)")
    plt.imshow(sdf.sdf, origin="lower", cmap=cmap, norm=norm)
    x_, y_ = zip(*final_path)
    plt.plot(x_, y_, label="Initial Trajectory", color="green") # Plot the initial trajectory
    if x is not None:
        plt.plot(x[0], x[1], label="Optimized trajectory", color="black", marker="o", markersize=2) # Plot the optimized trajectory
        print("x:" , x[0])
        print("y:" , x[1])
    plt.colorbar(label="Distance")
    plt.legend()

    plt.xticks(np.linspace(0, 200, 20))
    plt.yticks(np.linspace(0, 200, 20))
    plt.title("Path to trajectory optimization in a 2D environment with obstacles")
    plt.xlabel("Map Coordinate X - m")
    plt.ylabel("Map Coordinate Y - m")
    plt.show()







if __name__ == '__main__':
    main()