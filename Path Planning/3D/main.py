import os
import sys
import random
import numpy as np
import casadi as ca
import spatial_casadi as sc

import trimesh as tm
import pymesh

import pyvista as pv
from scipy.spatial import cKDTree, KDTree
from scipy.interpolate import griddata

from path_optimizer import TrajectoryOptimizer

from long_horizon_mpc import LongHorizonMPC


class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent


class RRT:
    def __init__(
        self, start, goal, map_instance, step_size=0.5, max_iterations=1000
    ):
        self.start = Node(start)
        self.goal = Node(goal)
        self.map = map_instance
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tree = [self.start]

    def get_nearest_node(self, sample):
        """Finds the nearest node in the tree to a sampled point."""
        positions = np.array([node.position for node in self.tree])
        tree = KDTree(positions)
        _, idx = tree.query(sample)
        return self.tree[idx]

    def is_collision_free(self, point):
        """Checks if a point is collision-free by checking with obstacles."""
        for mesh, _ in self.map.obstacles:
            if mesh.contains([point]):
                return False
        return True

    def steer(self, from_node, to_point):
        """Steers from a node toward a given point within the step size."""
        direction = to_point - from_node.position
        direction = direction / np.linalg.norm(direction) * self.step_size
        new_position = from_node.position + direction
        return Node(new_position, parent=from_node)

    def reached_goal(self, node):
        """Checks if a node is close enough to the goal."""
        return (
            np.linalg.norm(node.position - self.goal.position) < self.step_size
        )

    def plan_path(self):
        """Executes the RRT algorithm."""
        for _ in range(self.max_iterations):
            # Sample a random point or bias towards goal
            if random.random() < 0.2:  # 20% chance to sample goal directly
                sample = self.goal.position
            else:
                sample = np.array(
                    [
                        random.uniform(0, self.map.size_x),
                        random.uniform(0, self.map.size_y),
                        random.uniform(0, self.map.size_z),
                    ]
                )

            # Get nearest node and steer towards sample
            nearest_node = self.get_nearest_node(sample)
            new_node = self.steer(nearest_node, sample)

            # Add node if no collision
            if self.is_collision_free(new_node.position):
                self.tree.append(new_node)

                # Check if goal is reached
                if self.reached_goal(new_node):
                    return self.extract_path(new_node)

        return None  # No path found

    def extract_path(self, node):
        """Extracts the final path by tracing back from the goal."""
        path = []
        while node is not None:
            path.append(node.position)
            node = node.parent
        return path[::-1]  # Reverse to get start-to-goal order


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
        bounding_box = pv.Box(
            bounds=(0, self.size_x, 0, self.size_y, 0, self.size_z)
        )
        plotter.add_mesh(
            bounding_box, color="gray", opacity=0.1, style="wireframe"
        )

        for mesh, color in self.obstacles:
            faces = np.hstack([[len(face)] + list(face) for face in mesh.faces])
            poly_data = pv.PolyData(mesh.vertices, faces)
            plotter.add_mesh(
                poly_data, color=color, opacity=0.5, show_edges=True
            )

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


def main():
    np.set_printoptions(linewidth=np.inf)

    # Define obstacles (box and sphere)
    box = tm.creation.box(
        extents=[2, 2, 2],
        transform=tm.transformations.translation_matrix([3, 3, 3]),
    )
    sphere = tm.creation.icosphere(radius=1.5, subdivisions=3)
    sphere.apply_translation([7, 7, 7])
    obstacles = [(box, "red"), (sphere, "blue")]

    # Create the map instance
    map_instance = Map(size_x=10, size_y=10, size_z=10, obstacles=obstacles)

    # Instantiate the RRT planner
    rrt_planner = RRT(
        start=np.array([0, 0, 0]),
        goal=np.array([10, 10, 10]),
        map_instance=map_instance,
    )

    sdf = SDF(
        definition_x=10,
        definition_y=10,
        definition_z=10,
        map_instance=map_instance,
    )
    sdf.compute_sdf()
    sdf.plot().show()

    # Compute the path
    path = rrt_planner.plan_path()

    if path is None:
        print("No valid path found.")
        return

    # Convert path to numpy array for visualization
    path_array = np.array(path)

    # Plot the map
    plotter = map_instance.plot()

    # Plot the path
    plotter.add_mesh(pv.PolyData(path_array), color="green", line_width=4)
    plotter.show()

    print("Computed path:", path)
    sdf_interpolant = sdf.interpolation()
    try:
        sdf_value, (coords) = sdf.get_sdf_with_coords(2, 5, 2)
        print(
            "CasADi SDF Interpolation created.",
            sdf_value,
            " sdf query ",
            sdf_interpolant(coords),
            " coords ",
            coords,
        )
    except IndexError:
        print("Index out of bounds")

    print(
        "Box distance (3, 3, 3)",
        -tm.proximity.signed_distance(box, [[3, 3, 3]]),
    )

    LHMPC = LongHorizonMPC(sdf_interpolant, map_instance)
    LHMPC.setup_problem(
        N=40,
        dt=0.1,
        A_=np.load("A_matrix.npy"),
        J_=np.load("J_matrix.npy"),
        m_=np.load("mass.npy"),
        sdf_interpoland=sdf_interpolant,
    )

    sol = LHMPC.solve_problem(
        start_pos=np.array([0, 0, 0]),
        start_vel=np.array([0, 0, 0]),
        start_quat=np.array([0, 0, 0, 1]),
        start_ang_vel=np.array([0, 0, 0]),
        goal_pos=np.array([10, 10, 10]),
        goal_vel=np.array([0, 0, 0]),
        goal_quat=np.array([0, 0, 0, 1]),
        goal_ang_vel=np.array([0, 0, 0]),
    )

    if sol is not None:
        print(sol.value(LHMPC.p))
        print("\n")
        print(sol.value(LHMPC.v))

    # to = TrajectoryOptimizer(sdf_interpolant, map)
    # to.setup_problem(len(path), np.load("A_matrix.npy"), np.load("J_matrix.npy"), np.load("mass.npy"), initial_path=path)
    # sol = to.solve_problem(
    #    np.array([0, 0, 0]),
    #    np.array([0, 0, 0]),
    #    np.array([0, 0, 0, 1]),
    #    np.array([0, 0, 0]),
    #    np.array([10, 10, 10]),
    #    np.array([0, 0, 0]),
    #    np.array([0, 0, 0, 1]),
    #    np.array([0, 0, 0])
    # )

    # if sol is not None:
    #    print(sol.value(to.p))
    #    plotter = to.plot_trajectory(map_instance)
    #    plotter.show()

    # if  sol is not None:
    #    p = sol.value(to.p)

    #    for x, y, z in p.T:
    #        print(f"x: {x}, y: {y}, z: {z} - sdf: {sdf_interpolant([x, y, z])}")

    # trajectory_optimizer = TrajectoryOptimizer(start=(0, 0, 0), goal=(10, 10, 10), map_instance=map_instance, sdf_object=sdf, initial_path=path)

    # trajectory_optimizer.create_optimization_problem(N=20)

    # sol, x, v, t, q, w = trajectory_optimizer.optimize_path(
    #    start_pos=np.array([0, 0, 0]),
    #    start_vel=np.array([0, 0, 0]),
    #    start_q=np.array([1, 0, 0, 0]),
    #    start_w=np.array([0, 0, 0]),
    #    end_pos=np.array([1, 0, 0]),
    #    end_vel=np.array([0, 0, 0]),
    #    end_q=np.array([1, 0, 0, 0]),
    #    end_w=np.array([0, 0, 0])
    # )


if __name__ == "__main__":
    main()
