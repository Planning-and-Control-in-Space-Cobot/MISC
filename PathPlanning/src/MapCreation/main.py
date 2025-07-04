"""This script created a 3D point cloud mesh from a custom defined 3D object.

This mesh is also semgmented into multiple planes using RANSAC and visualized 
using both matplotlib and PyVista. The 3D point cloud is then saved to be used 
as the environment in a custom path planning with collision avoidance algorithm.
"""
import os
import argparse
import tempfile
import time
from typing import List, Tuple

import numpy as np
import trimesh
import open3d as o3d
import pyvista as pv
import matplotlib.pyplot as plt
import pickle as pkl

from Environment import EnvironmentHandler

def o3d_to_pv(o3d_mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    """Convert and open3d TriangleMesh to a PyVista PolyData object.

    In order to be able to show an open3d Triangle Mesh into a pyvista plot, we 
    need to convert it into a triangle mesh that is compatible with PyVista.

    Args:
        o3d_mesh (o3d.geometry.TriangleMesh): The open3d Triangle
            Mesh object to convert.
    
    Returns:
        pv.PolyData: The converted PyVista PolyData object.

    Raises:
        TypeError: If the input is not an open3d.geometry.TriangleMesh object.
        ValueError: If the input mesh does not have triangles.
    """
    if not isinstance(o3d_mesh, o3d.geometry.TriangleMesh):
        raise TypeError("Input must be an open3d.geometry.TriangleMesh object")
    if not o3d_mesh.has_triangles():
        raise ValueError("Input mesh must have triangles")

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    faces_flat = np.hstack([[3, *tri] for tri in faces])
    return pv.PolyData(vertices, faces_flat)

def strToBool(value: str) -> bool:
    """Convert a string representation of truth to a boolean value.

    This function is used to parse command line arguments that expect a boolean
    value. It accepts various string representations of truth and falsehood.

    Args:
        value (str): The string value to convert to boolean.

    Returns:
        bool: The converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the string value is not a valid boolean
            representation.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tmesh_to_o3d(tmesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert a trimesh Trimesh object to an open3d TriangleMesh object.

    The easier way to convert between these two libraries is to export the 
    mesh into a temporary file that both libraries can read and write, so beware
    a temporary file will be created and deleted in the process.

    Args:
        tmesh (trimesh.Trimesh): The trimesh Trimesh object to
            convert.
    
    Returns:
        o3d.geometry.TriangleMesh: The converted open3d TriangleMesh object.
    """
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        temp_path = tmp.name
        tmesh.export(temp_path)
    o3d_mesh = o3d.io.read_triangle_mesh(temp_path)
    os.remove(temp_path)
    return o3d_mesh

def sample_tmesh(
        tmesh: trimesh.Trimesh,
        number_of_points=200000) -> o3d.geometry.PointCloud:
    """Samples multiple point from the surface of a trimesh object.
    
    This function create an example point cloud with a specified number of 
    point that will then be saved as a point cloud file. It receives a trimesh
    object, converts it to an open3d TriangleMesh object and them takes x number
    of samples from the surface of the mesh using Poisson disk sampling.

    Args:
        tmesh (trimesh.Trimesh): The trimesh object to sample.
        number_of_points (int): The number of points to sample from the mesh.
    
    Returns:
        o3d.geometry.PointCloud: The sampled point cloud as an open3d PointCloud
        object.
    """
    o3d_mesh = tmesh_to_o3d(tmesh)
    return o3d_mesh.sample_points_poisson_disk(
        number_of_points=number_of_points
    )

def auto_segment_planes(
    pcd,
        distance_threshold=0.01,
        min_inliers=100,
        max_iterations=100
    ) -> Tuple[
        List[o3d.geometry.PointCloud],
        List[List[float]],
        o3d.geometry.PointCloud
    ]:
    """Automatically segments planes from a point cloud using RANSAC.

    This function uses the RANSAC algorithm to segment multiple planes from a
    point cloud. It iteratively finds planes until the number of inliers is less
    than the specified minimum or the maximum number of iterations is reached.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud to segment.
        distance_threshold (float): The maximum distance from a point to the 
            plane to be considered an inlier.
        min_inliers (int): The minimum number of inliers required to consider a
            plane valid.
        max_iterations (int): The maximum number of iterations to run the RANSAC
            algorithm.

    Returns:
        tuple: A tuple containing:
            - planes (list): A list of open3d.geometry.PointCloud objects, each
                representing a segmented plane.
            - models (list): A list of plane model parameters for each segmented
                plane.
            - remaining (o3d.geometry.PointCloud): The remaining points in the
                point cloud after plane segmentation.
    """
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

def plot_planes_matplotlib(planes, leftover=None) -> None:
    """Plots segmented planes in 3D using matplotlib.

    This function visualizes the segmented planes in a 3D scatter plot using
    matplotlib. Each plane is represented by its points, and any leftover points
    are shown in gray.

    Args:
        planes (list): A list of open3d.geometry.PointCloud objects, each
            representing a segmented plane.
        leftover (open3d.geometry.PointCloud, optional): The remaining points in
            the point cloud after plane segmentation. If provided, these points
            will be plotted in gray. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, pl in enumerate(planes):
        pts = np.asarray(pl.points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, label=f"Plane {i+1}")
    if leftover is not None and len(leftover.points) > 0:
        pts = np.asarray(leftover.points)
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=1,
            color="gray",
            label="Leftover"
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()

def create_oobb_mesh_from_planes(planes, color='white') -> pv.MultiBlock:
    """Creates a PyVista mesh from the oriented bounding boxes of planes.

    This function generates a mesh for each plane's oriented bounding box (OOBB)
    and combines them into a PyVista MultiBlock object. Each OOBB is represented
    as a polydata object with its corners and faces defined.

    Args:
        planes (list): A list of open3d.geometry.PointCloud objects, each
            representing a segmented plane.
        color (str, optional): The color to apply to the OOBB meshes. 
            Defaults to 'white'.

    Returns:
        pv.MultiBlock: A PyVista MultiBlock object containing the OOBB meshes 
            for each plane.
    """
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
    """Main function to create a 3D mesh, segment planes, and visualize."""
    parser = argparse.ArgumentParser(
        description="Create a 3D mesh and visualize it."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mesh.pcd',
        help='Output file name for the mesh'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the mesh using PyVista'
    )
    parser.add_argument(
        '--glassMaze',
        type=strToBool,
        default=False,
        help='Create a glass maze structure'
    )
    parser.add_argument(
        '--pcd-size',
        type=int,
        default=50000,
        help='Number of points to sample from the mesh'
    )
    parser.add_argument(
        '--measure-times', 
        type=strToBool,
        default=False,
        help='Make a careful measurement of the time taken in each step while' \
        'changing the number of sampled points in the point cloud.'
    )
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

    values = list(range(5000, args.pcd_size + 1, 5000))

    times = {
        "number_of_points": values,
        "pcdGenerationTime" : [],
        "coallisionMeshGenerationTimes" : []
    }
    if args.measure_times:
        print(f"Measuring times for point clouds with {values} points")

        # Prepare the plotter and open the GIF just once
        #pv_ = pv.Plotter(off_screen=True)
        #pv_.open_gif("orbit.gif")
        #pv_.add_axes()
        #pv_.show_grid()

        #center = np.array([0, 4, 3])  # Focal point of camera
        #radius = 15
        #elevation = 3
        #n_frames = 60

        for v in values:
            print(f"Sampling {v} points from the mesh...")
            timeStart = time.time()
            pcd = sample_tmesh(finalMesh, number_of_points=v)
            times["pcdGenerationTime"].append(time.time() - timeStart)

            timeStart = time.time()
            env = EnvironmentHandler(pcd)
            times["coallisionMeshGenerationTimes"].append(time.time() - timeStart)

            # Remove all actors from the previous round

            """
            pv_.clear_actors()

            # Add new point cloud
            cloud_actor = pv_.add_mesh(
                pv.PolyData(np.asarray(pcd.points)),
                color='blue',
                point_size=5,
                render_points_as_spheres=True,
                name="point_cloud"
            )

            # Add new voxel mesh
            voxel_actor = pv_.add_mesh(
                env.voxel_mesh,
                color='red',
                show_edges=True,
                opacity=0.5,
                name="voxel_mesh"
            )

            # Add text showing number of points
            label_actor = pv_.add_text(
                f"{v} points",
                position="upper_left",
                font_size=14,
                name="label"
            )

            for i in range(n_frames):
                theta = 2 * np.pi * i / n_frames
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                z = center[2]

                pv_.camera_position = [
                    (x, y, z),  # Camera on circle
                    center,     # Look at center
                    (0, 0, 1)   # Z-up
                ]

                pv_.render()
                pv_.write_frame()

        # Finalize and show (optional)
        pv_.close() """
        plt.subplot(1, 2, 1)
        plt.plot(
            times["number_of_points"],
            times["pcdGenerationTime"],
            marker='o',
            linestyle='-',
            color='blue',
            label='PCD Generation Time'
        )
        plt.xlabel("Number of Points")
        plt.ylabel("Time (s)")
        plt.title("Point Cloud Generation Time")
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(
            times["number_of_points"],
            times["coallisionMeshGenerationTimes"],
            marker='o',
            linestyle='-',
            color='red',
            label='Collision Mesh Generation Time'
        )
        plt.xlabel("Number of Points")
        plt.ylabel("Time (s)")
        plt.title("Collision Mesh Generation Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        with open("times.pkl", "wb") as f:
            pkl.dump(times, f)





            
    pcd = sample_tmesh(finalMesh, number_of_points=args.pcd_size)
    EnvironmentHandler(pcd)

    print("Segmenting planes...")
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
        pv_.add_mesh(
            pv.PolyData(np.asarray(pcd.points)),
            color='blue',
            point_size=2, 
            render_points_as_spheres=True)
        pv_.add_axes()
        pv_.show_grid()
        pv_.show()

    print(f"Point cloud saved to {outputFile}")
    o3d.io.write_point_cloud(outputFile, pcd)

if __name__ == "__main__":
    main()
