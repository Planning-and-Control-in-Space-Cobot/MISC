import os
import sys 
import random 
import numpy as np
import casadi as ca

import trimesh as tm
import pymesh

import pyvista as pv
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

class Map:
    def __init__(self, size_x, size_y, size_z, obstacles):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.obstacles = obstacles
    
    def plot(self):
        """Creates and returns a PyVista plotter with the full map dimensions plotted."""
        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.show_axes()
        plotter.show_grid()
        plotter.add_axes(interactive=True, line_width=3, labels_off=False)

        # Create a bounding box to represent the full map dimensions
        bounding_box = pv.Box(bounds=(0, self.size_x, 0, self.size_y, 0, self.size_z))
        plotter.add_mesh(bounding_box, color='gray', opacity=0.1, style='wireframe')

        for mesh, color in self.obstacles:
            faces = np.hstack([[len(face)] + list(face) for face in mesh.faces])
            poly_data = pv.PolyData(mesh.vertices, faces)
            plotter.add_mesh(poly_data, color=color, opacity=0.5, show_edges=True)
        
        return plotter

    def get_closest_distances(self, point):
        """
        Compute the distance from a 3D point to the closest face/vertex of each obstacle.

        :param point: (x, y, z) coordinates of the query point.
        :return: List of tuples containing (distance, closest_point, obstacle_color).
        """
        distances = []
        for mesh, color in self.obstacles:
            query = tm.proximity.ProximityQuery(mesh)
            closest_point, distance, _ = query.on_surface([point])
            distances.append((distance[0], closest_point[0], color))

        return distances

class SDF:
    def __init__(self, definition_x, definition_y, definition_z, map_instance):
        self.definition_x = definition_x
        self.definition_y = definition_y
        self.definition_z = definition_z
        self.map = map_instance
        self.sdf_grid = None
        self.sdf_values = None
        self.sdf_array = None


    def compute_sdf(self):
        """Compute the signed distance field (SDF) for the given map using Trimesh, processing one obstacle at a time."""
        # Generate grid points
        x_vals = np.linspace(0, self.map.size_x, self.definition_x)
        y_vals = np.linspace(0, self.map.size_y, self.definition_y)
        z_vals = np.linspace(0, self.map.size_z, self.definition_z)

        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Store the SDF grid in the correct shape
        self.sdf_grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Initialize a 3D array for SDF values
        self.sdf_values = np.full((self.definition_z, self.definition_y, self.definition_x), np.inf)

        # Process each obstacle separately
        for mesh, _ in self.map.obstacles:
            sdf_temp = -tm.proximity.signed_distance(mesh, self.sdf_grid)
            sdf_temp[sdf_temp < 0] = 0  # Set negative values to zero

            # Reshape to (z, y, x)
            sdf_temp = sdf_temp.reshape(self.definition_z, self.definition_y, self.definition_x, order="F")

            self.sdf_values = np.minimum(self.sdf_values, sdf_temp)


                                
    def get_sdf(self):
        """Returns the SDF values as a numpy array with integer indexed positions."""
        self.compute_sdf()
        self.sdf_array = self.sdf_values.reshape((self.definition_x, self.definition_y, self.definition_z), order="F")
        return self.sdf_array
    
    def interpolation(self):
        """Creates a CasADi interpolant for the SDF."""
        if self.sdf_array is None:
            self.get_sdf()
        
        sdf_rows, sdf_cols, sdf_pages = self.sdf_array.shape
        # Knots for the interpolation
        d_knots = [
            np.linspace(0, sdf_pages, self.definition_x),  # X-axis
            np.linspace(0, sdf_cols, self.definition_y),   # Y-axis
            np.linspace(0, sdf_rows, self.definition_z)    # Z-axis
        ]
        # Flatten SDF matrix for interpolation
        d_flat = self.sdf_array.ravel(order="F")

        return ca.interpolant(
            "SDF",
            "bspline", 
            d_knots, 
            d_flat, 
            {"algorithm": "smooth_linear"}
        )
   
    def plot(self):
        """Creates and returns a PyVista plotter with the SDF plotted."""
        self.compute_sdf()
        plotter = self.map.plot()
        
        # Define the grid explicitly using RectilinearGrid
        x = np.linspace(0, self.map.size_x, self.definition_x)
        y = np.linspace(0, self.map.size_y, self.definition_y)
        z = np.linspace(0, self.map.size_z, self.definition_z)
        grid = pv.RectilinearGrid(x, y, z)
        grid.point_data["SDF Distance"] = self.sdf_values.ravel(order="F")
        
        # Plot interpolated SDF as a volume with transparency
        opacity_levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plotter.add_volume(grid, scalars="SDF Distance", cmap="bwr", opacity=opacity_levels, opacity_unit_distance=4.0)
        plotter.add_scalar_bar(title="SDF Distance")
        return plotter

def main():
    # Define obstacles (box and sphere)
    box = tm.creation.box(extents=[2, 2, 2], transform=tm.transformations.translation_matrix([3, 3, 3]))
    sphere = tm.creation.icosphere(radius=1.5, subdivisions=3)
    sphere.apply_translation([7, 7, 7])
    obstacles = [(box, 'red'), (sphere, 'blue')]

    # Create the map instance
    map_instance = Map(size_x=10, size_y=10, size_z=10, obstacles=obstacles)
    map_plotter = map_instance.plot()
    #map_plotter.show()
    

    y = np.linspace(-1, 1, 100) 
    # Create and plot the SDF
    sdf_instance = SDF(definition_x=10, definition_y=10, definition_z=10, map_instance=map_instance)
    sdf_plotter = sdf_instance.plot()
    sdf_plotter.show()
    
    # Example usage of get_sdf and interpolation
    sdf_array = sdf_instance.get_sdf()
    print("SDF Shape:", sdf_array.shape)
    sdf_interpolant = sdf_instance.interpolation()
    print("CasADi SDF Interpolation created.", sdf_interpolant((2, 2, 2)), " sdf query ", sdf_instance.get_sdf()[2, 2, 2])
    print("Casadi interplatin at (3, 3, 3) ", sdf_interpolant((3, 3, 3)))
    print(box.vertices)
    print("Box contains (2, 2, 2) ", box.contains([[2, 2, 2]]), " now (3,3, 3) ", box.contains([[3, 3, 3]]))
    print("Box distance (3, 3, 3)", -tm.proximity.signed_distance(box, [[3, 3, 3]]))
    print("Casadi Interplolation at (1, 2, 2) ", sdf_interpolant((1, 2, 2)), "(2, 2, 2), ", sdf_interpolant((2, 2, 2)), "(3, 3, 3)", sdf_interpolant((3, 3, 3)))

if __name__ == "__main__":
    main()
