import trimesh
import open3d as o3d
import pyvista as pv
import numpy as np
import scipy.spatial.transform as trf

import os
import argparse

from Environment import EnvironmentHandler


def o3d_to_pv(o3d_mesh : o3d.geometry.TriangleMesh) -> pv.PolyData:
    """ Convert and open3d mesh into a pyvista mesh.

    Converts an open3d triangular mesh, into a mesh that can be used inside
    a pyvista plot.


    Parameters
       o3d_mesh (o3d.geometry.TriangleMesh): The open3d mesh to convert.
       
    Returns
       pv.PolyData: The converted mesh in pyvista format.

    Raises
         TypeError: If the input is not an open3d.geometry.TriangleMesh object.
         ValueError: If the input mesh does not have triangles.
    """
    if not type(o3d_mesh) is o3d.geometry.TriangleMesh:
        raise TypeError("Input must be an open3d.geometry.TriangleMesh object")
    
    if not o3d_mesh.has_triangles():
        raise ValueError("Input mesh must have triangles")
    
    # Extract vertices and faces
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # PyVista expects faces as a flat array: [3, i0, i1, i2, 3, i3, i4, i5, ...]
    faces_flat = np.hstack([[3, *tri] for tri in faces])

    print(f"Vertices shape: {vertices.shape}, Faces shape: {faces.shape}")
    # Create PolyData
    pv_mesh = pv.PolyData(vertices, faces_flat)
    return pv_mesh


def strToBool(value : str) -> bool:
    """Convert a string to a boolean value.

    This function converts a string representation of a boolean value
    to an actual boolean. It accepts various common representations
    such as 'yes', 'no', 'true', 'false', 't', 'f', 'y', 'n', '1', and '0'.

    Parameters
        value (str): The string to convert to a boolean.
    
    Returns:
        bool: The converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the string does not represent a valid 
        boolean value.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tmesh_to_o3d(tmesh : trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """ Convert a trimesh object to an open3d mesh object.

    Parameters
        tmesh (trimesh.Trimesh): The trimesh object to convert.
            
    Returns
        o3d.geometry.TriangleMesh: The converted open3d mesh object.    
    """
    tmesh.export("temp_mesh.obj")
    o3d_mesh = o3d.io.read_triangle_mesh("temp_mesh.obj")
    os.remove("temp_mesh.obj")
    return o3d_mesh

def sample_tmmesh(tmesh : trimesh.Trimesh, number_of_points=100000):
    """
    Sample points from a trimesh object.

    Parameters
        tmesh (trimesh.Trimesh): The trimesh object to sample points from.
        number_of_points (int): The number of points to sample from the mesh.
        
    Returns:
        o3d.geometry.PointCloud: A point cloud containing the sampled points.    
    """
    o3d_mesh = tmesh_to_o3d(tmesh)
    point_cloud = o3d_mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    return point_cloud

def main():
    parser = argparse.ArgumentParser(description="Create a 3D mesh and visualize it.")
    parser.add_argument('--output', type=str, default='mesh.pcd', help='Output file name for the mesh')
    parser.add_argument('--visualize', action='store_true', help='Visualize the mesh using PyVista')
    parser.add_argument('--glassMaze', type=strToBool, default=False, help='Create a glass maze structure')
    parser.add_argument('--pcd-size', type=int, default=10000, help='Number of points to sample from the mesh for point cloud generation')

    args = parser.parse_args()
    outputFile = args.output

    if args.glassMaze:
        # Generate the environment mesh of the glass maze first shown in the 
        # space cobot demo video (some changes were made to the original mesh,
        # in order to make the first trajectory the only one that is feasible)
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
        pcd = sample_tmmesh(finalMesh, number_of_points=args.pcd_size)

    else:
        # Create a cube
        cube1 = trimesh.creation.box(extents=(0.5, 10, 10))

        # Create a smaller cube to subtract
        cube2 = trimesh.creation.box(extents=(0.75, 0.6, 0.3))
        cube2.apply_translation([0.0, 3, 3])

        cube8 = trimesh.creation.box(extents=(2., 1.00, 1.00))
        cube8.apply_translation([1.0, 3, 3])

        cube9= trimesh.creation.box(extents=(2.00, 0.6, 0.3))
        cube9.apply_translation([1.0, 3, 3])

        cube8 = cube8.difference(cube9)


        cube1 = cube1.difference(cube2)

        cube3 = trimesh.creation.box(extents=(0.5, 10, 10))
        cube3.apply_translation([-0.85, 0, 0])
        # Perform boolean difference

        cube4 = trimesh.creation.box(extents=(1.5, 10, 0.25))
        cube4.apply_translation([-0.50, 0, 5])

        cube5 = trimesh.creation.box(extents=(1.5, 9.5, 0.25))
        cube5.apply_translation([-0.50, -0.5, 0])
            

        cube6 = trimesh.creation.box(extents=(1.5, 0.25, 10))
        cube6.apply_translation([-0.50, 5, 0])

        cube7 = trimesh.creation.box(extents=(1.5, 10, 0.25))
        cube7.apply_translation([-0.50, 0, -5])

        finalMesh = cube1 +  cube4 + cube5 + cube6 + cube7 + cube8 #+ cube3
        pcd = sample_tmmesh(finalMesh, number_of_points=args.pcd_size)

    env = EnvironmentHandler(pcd)
    if args.visualize:
        pv_ = pv.Plotter()
        #pv_.add_mesh(o3d_to_pv(tmesh_to_o3d(finalMesh)), show_edges=True, color='white', line_width=0.5, point_size=5, show_scalar_bar=False)
        pv_.add_mesh(env.voxel_mesh, show_edges=True, color='green', line_width=0.5, point_size=5, show_scalar_bar=False)
        pv_.add_mesh(pv.PolyData(np.asarray(pcd.points)), color='blue', point_size=2, render_points_as_spheres=True)
        pv_.add_axes()
        pv_.show_grid()


    if args.glassMaze:
        _spaceCobot = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))

        spaceCobot = _spaceCobot.copy()
        transform = np.eye(4)
        rot = trf.Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        pos = np.array([0.5, 2.0, 1.0])
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = pos
        spaceCobot.transform(transform)
        pv_.add_mesh(spaceCobot, color='red', show_edges=True)

        spaceCobot = _spaceCobot.copy()
        rot = trf.Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        pos = np.array([0.5, 5.0, 1.0])
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = pos
        spaceCobot.transform(transform)
        pv_.add_mesh(spaceCobot, color='green', show_edges=True)

        spaceCobot = _spaceCobot.copy()
        rot = trf.Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        pos = np.array([0.5, 5.0, 6.0])
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = pos
        spaceCobot.transform(transform)
        pv_.add_mesh(spaceCobot, color='green', show_edges=True)

    else: 
        _spaceCobot = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))

        spaceCobot = _spaceCobot.copy()
        transform = np.eye(4)
        rot = trf.Rotation.from_euler('xyz', [0, 75, 0], degrees=True)
        pos = np.array([-0.375, 0.0, -3.0])
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = pos

        spaceCobot.transform(transform)
        pv_.add_mesh(spaceCobot, color='red', show_edges=True)

        spaceCobot = _spaceCobot.copy()
        rot = trf.Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        pos = np.array([-0.375, 0.0, 3.0])
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = np.array([0.0, 3.0, 3.0]) 
        spaceCobot.transform(transform)
        pv_.add_mesh(spaceCobot, color='green', show_edges=True)
    
    if args.visualize:
        pv_.show()



if __name__ == "__main__":
    main()