import open3d as o3d
import pyvista as pv 
import trimesh as tm
import numpy as np

import sys 
import os

def main():
    def create_box(center, size):
        box = o3d.geometry.TriangleMesh.create_box(*size)
        box.translate(np.array(center) - np.array(size) / 2)
        box.compute_vertex_normals()
        return box

    boxes = [
        create_box(center=[-4.7, 4, 0], size=[1, 5, 5]),
        create_box(center=[0.4, 4, 0], size=[8, 5, 5])
    ]

    combined_pcd = o3d.geometry.PointCloud()
    for box in boxes:
        pcd = box.sample_points_poisson_disk(number_of_points=50000)
        combined_pcd += pcd

    # Visualize boxes and point cloud
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(boxes + [combined_pcd, axis])

    # Save to PCD
    o3d.io.write_point_cloud("environment.pcd", combined_pcd)

if __name__ == "__main__":
    main()