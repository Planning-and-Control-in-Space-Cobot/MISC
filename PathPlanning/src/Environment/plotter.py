import os
import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import numpy as np
import open3d as o3d

from Environment import EnvironmentHandler


# --- Helper function for multiprocessing ---
def voxelize_once(_):
    pcd_path = os.path.join(os.path.dirname(__file__), 'complexMap.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)
    env = EnvironmentHandler(pcd)
    return env.timeTaken


def _format_ticks(ax):
    ax.tick_params(labelsize=12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=6))
        axis.set_major_formatter(FormatStrFormatter('%.2f'))


def _apply_common_axes(ax, bounds):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    # Force visually equal lengths in all 3 dimensions
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-135)
    ax.set_proj_type('ortho')


def main():
    pcd_path = os.path.join(os.path.dirname(__file__), 'complexMap.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    print(f"points shape: {points.shape}")

    # --- Parallel voxelization (optional) ---
    print(f"Starting parallel voxelization with {cpu_count()} workers...")
    # with Pool(processes=cpu_count()) as pool:
    #     timeTaken = pool.map(voxelize_once, range(1000))

    timeTaken = np.load("voxelization_times.npy")

    # --- Violin Plot ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.violinplot(timeTaken, showmeans=False, showmedians=False, showextrema=True)
    ax.grid(True)
    ax.set_xlabel('Glass Maze map', fontsize=14)
    ax.set_ylabel('Time (seconds)', fontsize=14)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- Load one environment for visualization ---
    env = EnvironmentHandler(pcd)
    vertices = env.vertices
    quads = env.quads

    # Common bounds from raw point cloud
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    common_bounds = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

    # --- Raw Point Cloud ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5)
    ax.set_xlabel('X - m', fontsize=14)
    ax.set_ylabel('Y - m', fontsize=14)
    ax.set_zlabel('Z - m', fontsize=14)
    _format_ticks(ax)
    _apply_common_axes(ax, common_bounds)
    ax.legend(['Point Cloud'])
    plt.tight_layout()

    # --- Voxelized Mesh ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X - m", fontsize=14)
    ax.set_ylabel("Y - m", fontsize=14)
    ax.set_zlabel("Z - m", fontsize=14)

    # Build & add mesh
    faces = [[vertices[i] for i in quad] for quad in quads]
    face_color = (0.6, 0.8, 0.9, 0.1)  # RGBA
    mesh = Poly3DCollection(
        faces, facecolor=face_color, edgecolor='black',
        linewidths=0.05, alpha=0.1
    )
    ax.add_collection3d(mesh)

    _format_ticks(ax)
    _apply_common_axes(ax, common_bounds)
    ax.legend(['Voxelized Environment'])

    plt.show()


if __name__ == "__main__":
    main()
