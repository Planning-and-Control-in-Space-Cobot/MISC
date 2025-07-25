import os
import sys
import argparse
import tempfile
import time
import pickle
from typing import List, Tuple

import numpy as np
import trimesh
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt

from MapCreation.ObstacleMotion import SineMotion, NoAttitudeMotion
from MapCreation.Obstacle import Obstacle  # Updated import for Obstacle class

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def strToBool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tmesh_to_o3d(tmesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        temp_path = tmp.name
        tmesh.export(temp_path)
    o3d_mesh = o3d.io.read_triangle_mesh(temp_path)
    os.remove(temp_path)
    return o3d_mesh

def sample_tmesh(tmesh: trimesh.Trimesh, number_of_points=200000) -> o3d.geometry.PointCloud:
    o3d_mesh = tmesh_to_o3d(tmesh)
    return o3d_mesh.sample_points_uniformly(number_of_points=number_of_points)

def main():
    parser = argparse.ArgumentParser(description="Create a 3D mesh and visualize it.")
    parser.add_argument('--output', type=str, default='environment.pkl', help='Output file name for the mesh')
    parser.add_argument('--visualize', action='store_true', help='Visualize the mesh using PyVista')
    parser.add_argument('--glassMaze', type=strToBool, default=False, help='Create a glass maze structure')
    parser.add_argument('--pcd-size', type=int, default=500000, help='Number of points to sample from the mesh')
    args = parser.parse_args()

    outputFile = args.output

    if args.glassMaze:
        cube1 = trimesh.creation.box(extents=(1, 0.1, 5))
        cube2 = trimesh.creation.box(extents=(1, 0.1, 5))
        cube1.apply_translation([-0.6, 0, 0])
        cube2.apply_translation([0.6, 0, 0])
        cube3 = trimesh.creation.box(extents=(0.1, 1, 5))
        cube4 = trimesh.creation.box(extents=(0.1, 5, 5))
        cube3.apply_translation([1, 1.9, 0])
        cube4.apply_translation([1, -1.5, 0])
        cube5 = trimesh.creation.box(extents=(2.5, 5, 0.1))
        cube5.apply_translation([0, 1.1, 2.5])
        cube6 = trimesh.creation.box(extents=(0.1, 5, 1))
        cube7 = trimesh.creation.box(extents=(0.1, 5, 1))
        cube6.apply_translation([1, 1.7, 2.7])
        cube7.apply_translation([1, 1.7, 4.1])
        dynMesh = cube7
        finalMesh = cube1 + cube2 + cube3 + cube4 + cube5 + cube6
    else:
        cube1 = trimesh.creation.box(extents=(0.5, 10, 10))
        cube2 = trimesh.creation.box(extents=(0.75, 0.6, 0.3))
        cube2.apply_translation([0.0, 3, 3])
        cube8 = trimesh.creation.box(extents=(2., 1.00, 1.00))
        cube8.apply_translation([1.0, 3, 3])
        cube9 = trimesh.creation.box(extents=(2.00, 0.6, 0.3))
        cube9.apply_translation([1.0, 3, 3])
        cube8 = cube8.difference(cube9)
        cube1 = cube1.difference(cube2)
        cube4 = trimesh.creation.box(extents=(1.5, 10, 0.25))
        cube4.apply_translation([-0.50, 0, 5])
        cube5 = trimesh.creation.box(extents=(1.5, 9.5, 0.25))
        cube5.apply_translation([-0.50, -0.5, 0])
        cube6 = trimesh.creation.box(extents=(1.5, 0.25, 10))
        cube6.apply_translation([-0.50, 5, 0])
        cube7 = trimesh.creation.box(extents=(1.5, 10, 0.25))
        cube7.apply_translation([-0.50, 0, -5])
        dynMesh = cube7
        finalMesh = cube1 + cube4 + cube5 + cube6 + cube8

    # Static point cloud
    pcd = sample_tmesh(finalMesh, number_of_points=args.pcd_size)

    # Dynamic obstacle
    dyn_pcd = sample_tmesh(dynMesh, number_of_points=50000)
    dyn_motion = SineMotion()
    dyn_attitude = NoAttitudeMotion()
    dyn_obstacle = Obstacle(motion=dyn_motion, attitude=dyn_attitude, pcd=dyn_pcd, mesh=dynMesh)

    with open(outputFile, "wb") as f:
        pickle.dump({
            "staticPcd": np.asarray(pcd.points),
            "dynamicObstacles": [dyn_obstacle.to_dict()]
        }, f)

    print(f"Saved environment to {outputFile}")

    if args.visualize:
        pv_ = pv.Plotter()
        static_cloud = pv.PolyData(np.asarray(pcd.points))
        pv_.add_mesh(static_cloud, color='blue', point_size=2, render_points_as_spheres=True)

        dyn_cloud = pv.PolyData(np.asarray(dyn_obstacle.getPcd(0).points))
        pv_.add_mesh(dyn_cloud, color='red', point_size=3, render_points_as_spheres=True)

        pv_.add_axes()
        pv_.show_grid()

        n_frames = 100
        duration = 5.0
        delay = duration / n_frames

        pv_.open_gif("dynamic_obstacle.gif")
        for frame in range(n_frames):
            t = frame * delay
            dyn_cloud.points = np.asarray(dyn_obstacle.getPcd(t).points)
            pv_.write_frame()
            time.sleep(delay)

        pv_.close()

if __name__ == "__main__":
    main()
