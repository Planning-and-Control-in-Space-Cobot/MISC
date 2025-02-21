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
from path_optimizer import TrajectoryOptimizer, PathLinearizer


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
    planner = PathPlanning(map)
    final_path, explored_paths = planner.rrt()

    if final_path is not None:
        optimizer = TrajectoryOptimizer(initial_path=final_path, SDF=sdf, map=map)
        sol, x, u, t, v = optimizer.optimize()
        if sol is not None: 
            linearizer = PathLinearizer(sol, np.load("A_matrix.npy")[:2, :], 5, t)
            state = np.vstack((x, v))
            A_k_list, B_k_list = linearizer.linearize_trajectory(state, u)
            x_lin, u_lin = linearizer.compute_linearized_state_control(state, u)

            print("Linearized trajectory: ", x_lin)
            print("Non linearized trajectory: ", x)

            print("Linearized control: ", u_lin)
            print("Non linearized control: ", u)

            v = np.linalg.norm(v, axis=0)

    # Create folder for this experiment
    exp_dir = os.path.join(base_dir, str(experiment_id))
    os.makedirs(exp_dir, exist_ok=True)

    # Save data
    np.save(os.path.join(exp_dir, "initial_path.npy"), final_path)
    np.save(os.path.join(exp_dir, "sdf.npy"), sdf.sdf)
    np.save(os.path.join(exp_dir, "map.npy"), map.map)

    # Save plot instead of showing it
    fig, ax = plt.subplots(figsize=(16, 6))

    # Define colormap for SDF
    norm_sdf = TwoSlopeNorm(vmin=0, vcenter=0.01, vmax=sdf.sdf.max())
    cmap_sdf = plt.cm.bwr_r
    sdf_plot = ax.imshow(sdf.sdf, origin="lower", cmap=cmap_sdf, norm=norm_sdf)

    # Plot initial trajectory
    if final_path is not None:
        x_, y_ = zip(*final_path)
        ax.plot(x_, y_, label="Initial Trajectory", color="green")

        # If optimized path exists, plot it with velocity colors
        if x_lin is not None and u_lin is not None:
            ax.plot(x_lin[0], x_lin[1], label="Linearized Trajectory", color="blue")
            


        if x is not None and v is not None:
            norm_vel = mcolors.Normalize(vmin=min(v), vmax=max(v))
            cmap_vel = plt.cm.YlGn
            
            # Scatter plot for velocity
            sc = ax.scatter(x[0], x[1], c=v, cmap=cmap_vel, norm=norm_vel, marker="o", s=50, edgecolors="k")

    # Plot start and goal points
    ax.plot(start[0], start[1], "ro", label="Start")
    ax.plot(goal[0], goal[1], "bo", label="Goal")

    # Add first colorbar for SDF
    cbar_sdf = fig.colorbar(sdf_plot, ax=ax, fraction=0.05, pad=0.04)
    cbar_sdf.set_label("Distance to Obstacle (m)")

    # Add second colorbar for velocity if available
    if x is not None and v is not None:
        cbar_vel = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.12)
        cbar_vel.set_label("Velocity (m/s)")

    # Final plot settings
    ax.set_title(f"Experiment {experiment_id}: Path Optimization")
    ax.set_xlabel("Map Coordinate X - m")
    ax.set_ylabel("Map Coordinate Y - m")
    ax.legend()

    # Save plot
    plt_name = f"trajectory_plot_{experiment_id}.png"
    plt.savefig(os.path.join(exp_dir, plt_name))
    plt.show()
    plt.close()



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
