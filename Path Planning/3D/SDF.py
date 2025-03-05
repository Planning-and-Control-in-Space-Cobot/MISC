import numpy as np
import trimesh as tm
import pyvista as pv
import casadi as ca


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

        self.x_vals = x_vals
        self.y_vals = y_vals
        self.z_vals = z_vals

        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Store the SDF grid in the correct shape
        self.sdf_grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Initialize a 3D array for SDF values
        self.sdf_values = np.full(
            (self.definition_z, self.definition_y, self.definition_x), np.inf
        )

        # Process each obstacle separately
        for mesh, _ in self.map.obstacles:
            sdf_temp = -tm.proximity.signed_distance(mesh, self.sdf_grid)
            sdf_temp[sdf_temp < 0] = 0  # Set negative values to zero

            # Reshape to (z, y, x)
            sdf_temp = sdf_temp.reshape(
                self.definition_z,
                self.definition_y,
                self.definition_x,
                order="F",
            )

            self.sdf_values = np.minimum(self.sdf_values, sdf_temp)

    def get_sdf(self):
        """Returns the SDF as a numpy array with integer-indexed positions."""
        self.compute_sdf()
        self.sdf_array = self.sdf_values.reshape(
            (self.definition_x, self.definition_y, self.definition_z), order="F"
        )
        return self.sdf_array

    def get_sdf_with_coords(self, i, j, k):
        """Returns the SDF value and real-world coordinates of a given index (i, j, k)."""
        if (
            0 <= i < self.definition_x
            and 0 <= j < self.definition_y
            and 0 <= k < self.definition_z
        ):
            sdf_value = self.sdf_values[i, j, k]
            coord_x = i * (self.x_vals[1] - self.x_vals[0])
            coord_y = j * (self.y_vals[1] - self.y_vals[0])
            coord_z = k * (self.z_vals[1] - self.z_vals[0])
            return sdf_value, (coord_x, coord_y, coord_z)
        else:
            raise IndexError("SDF index out of bounds")

    def interpolation(self):
        """Creates a CasADi interpolant for the SDF."""
        if self.sdf_array is None:
            self.get_sdf()

        sdf_rows, sdf_cols, sdf_pages = self.sdf_array.shape
        # Knots for the interpolation
        d_knots = [
            np.linspace(0, sdf_pages, self.definition_x),  # X-axis
            np.linspace(0, sdf_cols, self.definition_y),  # Y-axis
            np.linspace(0, sdf_rows, self.definition_z),  # Z-axis
        ]
        # Flatten SDF matrix for interpolation
        d_flat = self.sdf_array.ravel(order="F")

        return ca.interpolant(
            "SDF",
            "bspline",
            d_knots,
            d_flat,
            # {"algorithm": "smooth_linear"}
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
        plotter.add_volume(
            grid,
            scalars="SDF Distance",
            cmap="bwr",
            opacity=opacity_levels,
            opacity_unit_distance=4.0,
        )
        plotter.add_scalar_bar(title="SDF Distance")
        return plotter
