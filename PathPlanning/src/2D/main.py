"""
MIT License

Copyright (c) 2025 André Rebelo Teixeira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import Point, Polygon
import matplotlib.colors as mcolors


from Map import Map
from sdf2d import SDF
from rrt import PathPlanning
from path_optimizer import TrajectoryOptimizer, PathLinearizer, ProblemType


# Function to generate a random point within bounds
def random_point(x_min=0, x_max=200, y_min=0, y_max=200):
    return random.randint(x_min, x_max), random.randint(y_min, y_max)


# Function to generate random obstacles
def random_obstacles(num_obstacles=5):
    obstacles = []
    for _ in range(num_obstacles):
        shape_type = random.choice(["circle", "rectangle"])

        if shape_type == "circle":
            center = random_point()
            radius = random.randint(10, 40)  # Random radius
            obstacles.append(Point(center).buffer(radius))
        else:  # Rectangle
            x1, y1 = random_point()
            width = random.randint(10, 50)
            height = random.randint(10, 50)
            x2, y2 = x1 + width, y1 + height
            obstacles.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    return obstacles


def run_experiment(experiment_id, base_dir):
    # Define environment limits
    x_size = 200
    y_size = 200

    # Generate random start and goal positions ensuring they are not in obstacles
    obstacles = random_obstacles(num_obstacles=random.randint(3, 6))

    while True:
        start = random_point()
        if not any(obs.contains(Point(start)) for obs in obstacles):
            break

    while True:
        goal = random_point()
        if not any(obs.contains(Point(goal)) for obs in obstacles) and goal != start:
            break

    map = Map(x_size, y_size, start, goal, obstacles=obstacles)
    sdf = SDF(map)
    sdf.set_max_sdf_value(20)
    planner = PathPlanning(map)
    final_path, explored_paths = planner.rrt()

    problem_type = {
        "SDF_ONLY": ProblemType.SDF_ONLY,
        "ACTUALTION_ONLY": ProblemType.ACTUATION_ONLY,
        "SQUARE_TIME": ProblemType.SQUARE_TIME,
        "LINEAR_TIME": ProblemType.LINEAR_TIME,
        "FULL": ProblemType.FULL,
    }

    # Create folder for this experiment
    exp_dir = os.path.join(base_dir, str(experiment_id))
    os.makedirs(exp_dir, exist_ok=True)

    if final_path is not None:
        counter = 0
        for pt in problem_type:
            print(problem_type[pt])
            counter += 1
            plt.subplot(2, 3, counter)

            # Plot the SDF
            norm_sdf = TwoSlopeNorm(vmin=0, vcenter=0.01, vmax=sdf.sdf.max())
            cmap_sdf = plt.cm.bwr_r
            sdf_plot = plt.imshow(sdf.sdf, origin="lower", cmap=cmap_sdf, norm=norm_sdf)

            # Add first colorbar for SDF
            cbar_sdf = plt.colorbar(sdf_plot, fraction=0.05, pad=0.04)
            cbar_sdf.set_label("Distance to Obstacle (m)")

            # Plot the RRT PATH
            x_, y_ = zip(*final_path)
            plt.plot(x_, y_, label="Initial Trajectory", color="green")
            plt.plot(start[0], start[1], "ro", label="start")
            plt.plot(goal[0], goal[1], "bo", label="goal")

            # Optimize and plot the trajectory
            optimizer = TrajectoryOptimizer(
                initial_path=final_path,
                SDF=sdf,
                map=map,
                Problem_Type=problem_type[pt],
            )
            plt.title(f"Optimization Problem Type: {pt} ")
            plt.xlabel(f"Map Coordinate X - m")
            plt.ylabel(f"Map Coordinate y - m")
            sol, x, u, t, v = optimizer.optimize()
            if sol is not None:
                v = np.linalg.norm(v, axis=0)
                plt.title(
                    f"Optimization Problem Type: {pt} - Delta Time - {round(t, 3)}"
                )
                norm_vel = mcolors.Normalize(vmin=min(v), vmax=max(v))
                cmap_vel = plt.cm.YlGn
                if np.any(np.isinf(x[0])) or np.any(np.isnan(x[0])):
                    print("Inf or Nan")
                if np.any(np.isinf(x[1])) or np.any(np.isnan(x[1])):
                    print("Inf or Nan")

                plt.scatter(
                    x[0],
                    x[1],
                    c=v,
                    cmap=cmap_vel,
                    norm=norm_vel,
                    marker="o",
                    s=50,
                )

            plt.legend()
    plt_name = f"tp_{experiment_id}.png"
    plt.savefig(os.path.join(exp_dir, plt_name))
    plt.show()


def main():
    # Check if "test" mode is enabled
    test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "test"

    if test_mode:
        print("Running in test mode: Generating 100 experiments.")
        base_dir = "plots"
        os.makedirs(base_dir, exist_ok=True)

        for i in range(1, 101):
            print(f"Running experiment {i}...")
            run_experiment(i, base_dir)

        print("All experiments completed. Check the 'plots' directory for results.")

    else:
        print("Running in normal mode.")
        run_experiment("single_run", "plots")
        print("Results saved in 'plots/single_run'.")


if __name__ == "__main__":
    main()
