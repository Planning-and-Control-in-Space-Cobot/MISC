import pyvista as pv
import numpy as np
import trimesh as tm


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
        plotter.add_mesh(bounding_box, color="gray", opacity=0.1, style="wireframe")

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
