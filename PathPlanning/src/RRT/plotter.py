import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

from RRTOptimization.OptimizationState import OptimizationState

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

def fit_and_score(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    r2 = r2_score(y, y_pred)
    rmse = sqrt(mean_squared_error(y, y_pred))
    return coeffs, r2, rmse, y_pred

def main():
    jsonFile = "rrt_results.json"

    if not os.path.exists(jsonFile):
        print(f"Error: {jsonFile} does not exist.")
        return

    with open(jsonFile, 'r') as f:
        data = json.load(f)

    # Organize runs by map
    maps = list(dict.fromkeys(run["pcd"] for run in data))
    runsPerMap = {map_name: [] for map_name in maps}
    for run in data:
        if run.get("timeTaken", 0) > 0:
            runsPerMap[run["pcd"]].append(run)

    # Store stats for summary plots
    mean_times = []
    std_times = []
    map_labels = []
    time_distributions = []
    timesPerMap = {map_name: [] for map_name in maps}

    # Individual trisurf plots
    for map_name in maps:
        runs = runsPerMap[map_name]
        if not runs:
            continue

        goalBias = np.array([run["goalBias"] for run in runs])
        stepSize = np.array([run["stepSize"] for run in runs])
        timeTaken = np.array([run["timeTaken"] for run in runs])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trisurf = ax.plot_trisurf(goalBias, stepSize, timeTaken,
                                  cmap='viridis', linewidth=0.2,
                                  antialiased=True, shade=True,
                                  alpha=0.9)
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(timeTaken)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Time Taken')

        ax.set_title(f"Map: {os.path.basename(map_name)}")
        ax.set_xlabel("Goal Bias")
        ax.set_ylabel("Step Size")
        ax.set_zlabel("Time Taken")

        map_labels.append(os.path.basename(map_name))
        mean_times.append(np.mean(timeTaken))
        std_times.append(np.std(timeTaken))
        time_distributions.append(timeTaken)
        timesPerMap[map_name].append(timeTaken)

    # Box and violin plots of time taken
    boxPlotData = [
        [run["timeTaken"] for run in runsPerMap[map_name]]
        for map_name in maps
    ]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].violinplot(boxPlotData, showmeans=False, showmedians=True)
    axs[0].set_title('Violin plot')
    axs[0].yaxis.grid(True)
    axs[0].set_xticks([x + 1 for x in range(len(map_labels))])
    axs[0].set_xticklabels(map_labels, rotation=45)
    axs[0].set_xlabel('Map')
    axs[0].set_ylabel('Time Taken')

    axs[1].boxplot(boxPlotData, whis=[0, 100])
    axs[1].set_title('Box plot')
    axs[1].yaxis.grid(True)
    axs[1].set_xticks([x + 1 for x in range(len(map_labels))])
    axs[1].set_xticklabels(map_labels, rotation=45)
    axs[1].set_xlabel('Map')
    axs[1].set_ylabel('Time Taken')

    plt.tight_layout()

    plt.figure()
    plt.violinplot(boxPlotData, showmeans=False, showmedians=True)
    plt.grid(True)
    plt.xticks([x + 1 for x in range(len(map_labels))], ["Simple Map", "Moderate Map", "Complex Map"], rotation=45)
    plt.xlabel('Map', fontsize=14)
    plt.ylabel('Time - s', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(rotation=0)

    for run in data:
        if not run.get("success", True):
            print(f"Run failed: {run['pcd']} with goal bias {run['goalBias']} and step size {run['stepSize']}")

    # Paired violin plot for path length
    violin_data = []
    violin_labels = []

    for m in maps:
        original = [r["pathLength"] for r in runsPerMap[m] if r.get("pathLength") is not None]
        pruned = [r["prunedPathLength"] for r in runsPerMap[m] if r.get("prunedPathLength") is not None]

        if len(original) == 0 and len(pruned) == 0:
            continue

        violin_data.append(original)
        violin_labels.append(f"{os.path.basename(m)}\nOriginal")

        violin_data.append(pruned)
        violin_labels.append(f"{os.path.basename(m)}\nPruned")

    plt.figure(figsize=(14, 6))
    plt.violinplot(violin_data, showmeans=False, showmedians=True, showextrema=False)
    plt.xticks(ticks=range(1, len(violin_labels) + 1), labels=violin_labels, rotation=45)
    plt.ylabel("Path Size (Number of Nodes)")
    plt.title("Distribution of Original vs Pruned Path Sizes per Map")
    plt.grid(axis="y")
    plt.tight_layout()

    # NEW: Violin plot — RRT Time vs Prune Time per map (side-by-side)
    pair_data = []
    pair_labels = []
    for m in maps:
        rrt_times = [r["timeTaken"] for r in runsPerMap[m] if r.get("timeTaken") is not None]
        prune_times = [r["pruneTime"] for r in runsPerMap[m] if r.get("pruneTime") is not None]
        if len(rrt_times) == 0 and len(prune_times) == 0:
            continue
        pair_data.append(rrt_times)
        pair_labels.append(f"{os.path.basename(m)}\nRRT")
        pair_data.append(prune_times)
        pair_labels.append(f"{os.path.basename(m)}\nPrune")

    plt.figure(figsize=(14, 6))
    plt.violinplot(pair_data, showmeans=False, showmedians=True, showextrema=False)
    plt.xticks(ticks=range(1, len(pair_labels) + 1), labels=pair_labels, rotation=0)
    plt.ylabel("Time (s)")
    plt.title("RRT Time vs Prune Time — Distribution per Map")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # NEW: Aggregated violin — all maps combined: RRT vs Prune
    all_rrt = [r["timeTaken"] for r in data if r.get("timeTaken") is not None and r["timeTaken"] > 0]
    all_prune = [r["pruneTime"] for r in data if r.get("pruneTime") is not None]

    plt.figure(figsize=(6, 6))
    plt.violinplot([all_rrt, all_prune], showmeans=False, showmedians=True, showextrema=False)
    plt.xticks([1, 2], ["RRT Time", "Prune Time"], rotation=0)
    plt.ylabel("Time (s)")
    plt.title("RRT Time vs Prune Time — All Maps")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Scatter: prune time vs pruned path size
    plt.figure(figsize=(8, 6))
    for m in maps:
        prune_time = [r["pruneTime"] for r in runsPerMap[m] if r.get("pruneTime") is not None]
        pruned_size = [r["prunedPathLength"] for r in runsPerMap[m] if r.get("prunedPathLength") is not None]
        if len(prune_time) == 0 or len(pruned_size) == 0:
            continue
        plt.scatter(pruned_size, prune_time, label=os.path.basename(m), alpha=0.7)

    plt.xlabel("Pruned Path Size (Number of Nodes)")
    plt.ylabel("Prune Time (s)")
    plt.title("Prune Time vs Pruned Path Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Regression comparison
    print("\nFull Fit Comparison per Map:")
    print("=" * 100)
    print(f"{'Map':<20} | {'Model':<10} | {'R':>6} | {'R²':>6} | {'RMSE':>10}")
    print("-" * 100)

    plt.figure(figsize=(10, 6))

    for m in maps:
        x = np.array([r["pathLength"] for r in runsPerMap[m] if r.get("pathLength") is not None])
        y = np.array([r["pruneTime"] for r in runsPerMap[m] if r.get("pruneTime") is not None])
        map_name = os.path.basename(m)

        if len(x) < 3 or len(y) < 3:
            continue

        lin_coeffs, r2_lin, rmse_lin, y_lin = fit_and_score(x, y, 1)
        r_lin = np.corrcoef(x, y)[0, 1]

        quad_coeffs, r2_quad, rmse_quad, y_quad = fit_and_score(x, y, 2)
        cubic_coeffs, r2_cubic, rmse_cubic, y_cubic = fit_and_score(x, y, 3)

        print(f"{map_name:<20} | {'Linear':<10} | {r_lin:6.4f} | {r2_lin:6.4f} | {rmse_lin:10.4f}")
        print(f"{map_name:<20} | {'Quadratic':<10} | {'-':>6} | {r2_quad:6.4f} | {rmse_quad:10.4f}")
        print(f"{map_name:<20} | {'Cubic':<10} | {'-':>6} | {r2_cubic:6.4f} | {rmse_cubic:10.4f}")
        print()

        # Plot best fit
        best_model, best_y_fit = max(
            [('Linear', r2_lin, y_lin), ('Quadratic', r2_quad, y_quad), ('Cubic', r2_cubic, y_cubic)],
            key=lambda x: x[1]
        )[0:3:2]
        coeffs = {'Linear': lin_coeffs, 'Quadratic': quad_coeffs, 'Cubic': cubic_coeffs}[best_model]
        x_vals = np.linspace(min(x), max(x), 100)
        y_plot = np.polyval(coeffs, x_vals)

        plt.scatter(x, y, label=f"{map_name} data", alpha=0.6)
        plt.plot(x_vals, y_plot, linestyle='--', label=f"{map_name} ({best_model})")

    plt.title("Prune Time vs Path Length with Best Fit per Map")
    plt.xlabel("Original Path Length")
    plt.ylabel("Prune Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # NEW: One combined violin plot — RRT vs Prune Time per map (publication style)
    figure = plt.figure(figsize=(8, 12))
    violin_data = []
    violin_positions = []
    offset = 0.15  # small horizontal offset so they don't overlap

    for i, m in enumerate(maps, start=1):
        rrt_times = [r["timeTaken"] for r in runsPerMap[m] if r.get("timeTaken") is not None]
        prune_times = [r["pruneTime"] for r in runsPerMap[m] if r.get("pruneTime") is not None]

        violin_data.append(rrt_times)
        violin_positions.append(i - offset)

        violin_data.append(prune_times)
        violin_positions.append(i + offset)

    parts = plt.violinplot(
        violin_data,
        positions=violin_positions,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # Color RRT and Prune differently
    for idx, pc in enumerate(parts['bodies']):
        if idx % 2 == 0:
            pc.set_facecolor('#1f77b4')  # blue for RRT
        else:
            pc.set_facecolor('#ff7f0e')  # orange for Prune
        pc.set_alpha(0.7)

    # Set x-ticks at map positions
    plt.xticks(
        ticks=range(1, len(maps) + 1),
        labels=["Simple Map", "Moderate Map", "Complex Map"],
        fontsize=12,
        rotation=0
    )
    plt.yticks(fontsize=12)
    plt.xlabel('Glass Maze map', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    plt.legend(
        [Patch(facecolor='#1f77b4', alpha=0.7, label='RRT Time'),
         Patch(facecolor='#ff7f0e', alpha=0.7, label='Prune Time')],
        ['RRT Time', 'Prune Time'],
        loc='upper right'
    )
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # === NEW FIGURE: Prune ratio distributions per map ===
    ratio_data = []
    map_ticks = []
    for m in maps:
        ratios = []
        for r in runsPerMap[m]:
            orig = r.get("pathLength")
            prun = r.get("prunedPathLength")
            if orig is None or prun is None or orig <= 0:
                continue
            ratio = prun / orig
            ratios.append(ratio)
        if ratios:
            ratio_data.append(ratios)
            map_ticks.append(os.path.basename(m))

    fig, ax = plt.subplots(figsize=(8, 6))

    vp1 = ax.violinplot(ratio_data, showmeans=False, showmedians=True, showextrema=False)

    # Set labels
    ax.set_xticks(range(1, len(map_ticks) + 1))
    ax.set_xticklabels(['Simple Map', 'Moderate Map', 'Complex Map'], fontsize=12)
    ax.set_ylabel('Pruned / Original (ratio)', fontsize=14)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='y', labelsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # load (left intentionally as in your original script)

if __name__ == "__main__":
    main()
    plt.savefig("voxelization_times.png")
