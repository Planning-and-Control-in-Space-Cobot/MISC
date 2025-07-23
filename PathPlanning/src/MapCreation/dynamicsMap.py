import numpy as np
import pyvista as pv
import open3d as o3d
import trimesh
import time
import tempfile
import os

def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert a Trimesh to an Open3D TriangleMesh using a temp file."""
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        mesh.export(f.name)
        o3d_mesh = o3d.io.read_triangle_mesh(f.name)
    os.remove(f.name)
    return o3d_mesh

def sample_cube(center, size=(1, 1, 1), n_points=10000):
    cube = trimesh.creation.box(extents=size)
    cube.apply_translation(center)
    o3d_mesh = trimesh_to_o3d(cube)
    return np.asarray(o3d_mesh.sample_points_poisson_disk(n_points).points)

# Generate point clouds
static_points = sample_cube(center=(0, 0, 0))
moving_points_initial = sample_cube(center=(2, 0, 0))

# Combine into one polydata
all_points = np.vstack([static_points, moving_points_initial])
point_cloud = pv.PolyData(all_points)

# Plotter setup
plotter = pv.Plotter()
actor = plotter.add_mesh(point_cloud, color='blue', point_size=3, render_points_as_spheres=True)
plotter.show(auto_close=False, interactive_update=True)

# Separate moving points
n_static = static_points.shape[0]
n_moving = moving_points_initial.shape[0]

# Animation loop
for i in range(300):
    offset_y = 2.0 * np.sin(i * 0.1)
    new_moving = moving_points_initial.copy()
    new_moving[:, 1] += offset_y
    updated = np.vstack([static_points, new_moving])
    point_cloud.points = updated
    plotter.update()
    time.sleep(0.02)

plotter.close()
