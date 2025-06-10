import open3d as o3d
import numpy as np
import pyvista as pv
import time

# Load point cloud
pcd = o3d.io.read_point_cloud("environment.pcd")
print(f"Loaded point cloud with {np.asarray(pcd.points).shape[0]} points.")

# Estimate normals
start_time = time.time()
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.3, max_nn=300))
pcd.normalize_normals()
elapsed_time = time.time() - start_time
print(f"Normal estimation completed in {elapsed_time:.3f} seconds.")

# Convert to NumPy arrays
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# PyVista visualization
point_cloud = pv.PolyData(points)
point_cloud["Normals"] = normals

# Fit plane patches for selected points (optional)
sample_indices = np.random.choice(len(points), size=10, replace=False)
planes = []
for idx in sample_indices:
    p = points[idx]
    n = normals[idx]
    plane = pv.Plane(center=p, direction=n, i_size=0.3, j_size=0.3)
    planes.append(plane)

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, color="white", point_size=4, render_points_as_spheres=True)
plotter.add_arrows(points[sample_indices], normals[sample_indices], mag=0.3, color="red")
for plane in planes:
    plotter.add_mesh(plane, color="yellow", opacity=0.3)

plotter.add_axes()
plotter.show()

