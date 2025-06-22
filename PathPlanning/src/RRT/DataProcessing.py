"""Module for processing and visualizing optimization data."""

# Import required packages for plotting, numerical operations and I/O
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import os
import sys

# Append parent directory to the system path for custom imports
scriptDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(scriptDir))

def main():
    """Read optimization data and visualize the optimization process.

    This function reads optimization data from a pickle file, processes it,
    and generates two figures showing cost variation, time steps, time per
    iteration, cumulative time, number of obstacles, maximum distances to
    obstacles, and collision status.
    """
    # Load the optimization data from a pickle file
    with open(os.path.join(scriptDir, "optimizedPath.pkl"), "rb") as f:
        data = pickle.load(f)

    print("Data loaded successfully.")

    # Initialize lists to store data for each iteration
    PathToOptimize = []
    OptimizedPath = []
    InitialU = []
    OptimizedU = []
    Dt = []
    Obstacles = []
    MaxDistances = []
    Cost = []
    OptimizationTime = []
    AnuCollision = []

    # Separate data for each iteration and store in corresponding lists
    for d in data:
        (optimizationPath, _optimizationPath, prev_u, _prev_u,
         dt, _dt, obstacles, maxDistances, cost,
         optimizationTime, anyCollision) = d

        PathToOptimize.append(optimizationPath)
        OptimizedPath.append(_optimizationPath)
        InitialU.append(prev_u)
        OptimizedU.append(_prev_u)
        Dt.append(dt)
        Obstacles.append(obstacles)
        MaxDistances.append(maxDistances)
        Cost.append(cost)
        OptimizationTime.append(optimizationTime)
        AnuCollision.append(anyCollision)

    # === Figure 1: Cost, time step, time per step, cumulative time ===
    fig1 = plt.figure(figsize=(12, 8))
    gs1 = gridspec.GridSpec(3, 2)

    ax_left = fig1.add_subplot(gs1[:, 0])
    ax_top_right = fig1.add_subplot(gs1[0, 1])
    ax_middle_right = fig1.add_subplot(gs1[1, 1])
    ax_bottom_right = fig1.add_subplot(gs1[2, 1])

    # Plot cost variation
    ax_left.set_title("Variation of the cost")
    ax_left.plot(Cost, marker='o', linestyle='-', color='b')
    ax_left.set_xlabel("Optimization Iteration")
    ax_left.set_ylabel("Cost")
    ax_left.grid(True)

    # Plot time step variation
    ax_top_right.set_title("Variation of the time step")
    ax_top_right.plot(Dt, marker='o', linestyle='-', color='g')
    ax_top_right.set_xlabel("Optimization Iteration")
    ax_top_right.set_ylabel("Time Step (s)")
    ax_top_right.grid(True)

    # Plot time needed per optimization iteration
    meanTime = np.mean(OptimizationTime)
    ax_middle_right.set_title("Time needed per optimization")
    ax_middle_right.plot(OptimizationTime, marker='o',
                         linestyle='-', color='r',
                         label='Time per Iteration')
    ax_middle_right.axhline(meanTime, linestyle='--',
                            color='orange', label='Mean Time')
    ax_middle_right.set_xlabel("Optimization Iteration")
    ax_middle_right.set_ylabel("Time (s)")
    ax_middle_right.legend()
    ax_middle_right.grid(True)

    # Plot cumulative optimization time
    cumSum = np.cumsum(OptimizationTime)
    ax_bottom_right.set_title("Cumulative optimization time")
    ax_bottom_right.plot(cumSum, marker='o', linestyle='-',
                         color='purple')
    ax_bottom_right.set_xlabel("Optimization Iteration")
    ax_bottom_right.set_ylabel("Cumulative Time (s)")
    ax_bottom_right.grid(True)

    fig1.suptitle("Optimization Process Analysis", fontsize=16)
    fig1.set_tight_layout(True)


    # Save figure 1 to disk
    directory = "/home/andret/MEEC/Thesis/images/robotOnly"
    fig1.savefig(os.path.join(directory,
                 "CostVariationWithIterationNoMaxDT.png"),
                 bbox_inches="tight", dpi=300)


    #== Figure 2: Time per iteration, num obstacles, max distance, collision ==
    fig2 = plt.figure(figsize=(12, 8))
    gs2 = gridspec.GridSpec(3, 2)

    ax_left = fig2.add_subplot(gs2[:, 0])
    ax_top_right = fig2.add_subplot(gs2[0, 1])
    ax_middle_right = fig2.add_subplot(gs2[1, 1])
    ax_bottom_right = fig2.add_subplot(gs2[2, 1])

    # Plot time per iteration
    meanTime = np.mean(OptimizationTime)
    ax_left.set_title("Time needed per optimization")
    ax_left.plot(OptimizationTime, marker='o', linestyle='-',
                 color='r', label='Time per Iteration')
    ax_left.axhline(meanTime, linestyle='--', color='orange',
                    label='Mean Time')
    ax_left.set_xlabel("Optimization Iteration")
    ax_left.set_ylabel("Time (s)")
    ax_left.legend()
    ax_left.grid(True)

    # Plot number of obstacles per iteration
    ax_top_right.set_title("Number of obstacles")
    num_obstacles = [len(obs) for obs in Obstacles]
    ax_top_right.plot(num_obstacles, marker='o', linestyle='-',
                      color='g')
    ax_top_right.set_xlabel("Optimization Iteration")
    ax_top_right.set_ylabel("Number of Obstacles")
    ax_top_right.grid(True)

    # Plot maximum distance to obstacles
    distances = np.mean(np.array(MaxDistances), axis=1)
    ax_middle_right.set_title("Max distance to obstacles")
    ax_middle_right.plot(distances, marker='o', linestyle='-',
                         color='b')
    ax_middle_right.set_xlabel("Optimization Iteration")
    ax_middle_right.set_ylabel("Max Distance (m)")
    ax_middle_right.grid(True)

    # Plot any collision flag per iteration
    any_collision = [1 if collision else 0 for collision in AnuCollision]
    ax_bottom_right.set_title("Any collision during optimization")
    ax_bottom_right.plot(any_collision, marker='o', linestyle='-',
                         color='purple')
    ax_bottom_right.set_xlabel("Optimization Iteration")
    ax_bottom_right.set_ylabel("Any Collision (1=Yes, 0=No)")
    ax_bottom_right.grid(True)

    fig2.suptitle("Optimization Process Analysis", fontsize=16)
    fig2.set_tight_layout(True)

    # Save figure 2 to disk
    fig2.savefig(os.path.join(directory,
                 "ObstacleVariationWithIterationNoMaxDT.png"),
                 bbox_inches="tight", dpi=300)

    plt.show()
    plt.close(fig2)


# Entry point
if __name__ == "__main__":
    main()
