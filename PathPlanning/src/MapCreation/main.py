import trimesh
import open3d as o3d
import pyvista as pv
import numpy as np
import scipy.spatial.transform as trf
import matplotlib.pyplot as plt
import os
import argparse
import tempfile

from Environment import EnvironmentHandler
import coal

def o3d_to_pv(o3d_mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    if not isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
        raise TypeError("Input must be an open3d.geometry.TriangleMesh object")
    if not o3d_mesh.has_triangles():
        raise ValueError("Input mesh must have triangles")

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    faces_flat = np.hstack([[3, *tri] for tri in faces])
    return pv.PolyData(vertices, faces_flat)

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

def sample_tmmesh(tmesh: trimesh.Trimesh, number_of_points=200000):
    o3d_mesh = tmesh_to_o3d(tmesh)
    return o3d_mesh.sample_points_poisson_disk(number_of_points=number_of_points)

def auto_segment_planes(pcd, distance_threshold=0.01, min_inliers=100, max_iterations=100):
    planes, models, remaining = [], [], pcd
    count = 0
    while len(remaining.points) > min_inliers and count < max_iterations:
        model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers) < min_inliers:
            break
        plane = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        planes.append(plane)
        models.append(model)
        count += 1
    return planes, models, remaining

def plot_planes_matplotlib(planes, leftover=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, pl in enumerate(planes):
        pts = np.asarray(pl.points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, label=f"Plane {i+1}")
    if leftover is not None and len(leftover.points) > 0:
        pts = np.asarray(leftover.points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color="gray", label="Leftover")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def create_oobb_mesh_from_planes(planes, color='white'):
    box_meshes = []
    for i, pl in enumerate(planes):
        oobb = pl.get_oriented_bounding_box()
        corners = np.asarray(oobb.get_box_points())
        faces = [
            [3, 0, 1, 2], [3, 0, 2, 3],
            [3, 4, 5, 6], [3, 4, 6, 7],
            [3, 0, 1, 5], [3, 0, 5, 4],
            [3, 1, 2, 6], [3, 1, 6, 5],
            [3, 2, 3, 7], [3, 2, 7, 6],
            [3, 3, 0, 4], [3, 3, 4, 7]
        ]
        faces_flat = np.hstack(faces)
        pv_box = pv.PolyData(corners, faces_flat)
        pv_box["plane_id"] = np.full(len(pv_box.points), i)
        box_meshes.append(pv_box)
    return pv.MultiBlock(box_meshes)

def main():
    parser = argparse.ArgumentParser(description="Create a 3D mesh and visualize it.")
    parser.add_argument('--output', type=str, default='mesh.pcd', help='Output file name for the mesh')
    parser.add_argument('--visualize', action='store_true', help='Visualize the mesh using PyVista')
    parser.add_argument('--glassMaze', type=strToBool, default=False, help='Create a glass maze structure')
    parser.add_argument('--pcd-size', type=int, default=50000, help='Number of points to sample from the mesh')
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
        finalMesh = cube1 + cube2 + cube3 + cube4 + cube5 + cube6 + cube7
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
        finalMesh = cube1 + cube4 + cube5 + cube6 + cube7 + cube8

    pcd = sample_tmmesh(finalMesh, number_of_points=args.pcd_size)
    env = EnvironmentHandler(pcd)

    print("Segmenting planes...")
    import time
    timeStart = time.time()
    planes, models, remaining = auto_segment_planes(pcd)
    print(f"Plane segmentation took {time.time() - timeStart:.4f} seconds.")
    print(f"Found {len(planes)} planes in the point cloud.")
    
    # Plot plane segmentation
    plot_planes_matplotlib(planes, leftover=remaining)

    # Generate mesh from OOBBs of planes
    print("Creating OOBB meshes...")
    oobb_meshes = create_oobb_mesh_from_planes(planes)

    # Visualize in PyVista
    if args.visualize:
        pv_ = pv.Plotter()
        for box in oobb_meshes:
            pv_.add_mesh(box, show_edges=True, opacity=0.5)
        pv_.add_mesh(pv.PolyData(np.asarray(pcd.points)), color='blue', point_size=2, render_points_as_spheres=True)
        pv_.add_axes()
        pv_.show_grid()
        pv_.show()

    print(f"Point cloud saved to {outputFile}")
    o3d.io.write_point_cloud(outputFile, pcd)

if __name__ == "__main__":
    main()
