import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, x, y, z, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent
        self.cost = 0  # For RRT*


class PathPlanning:
    def __init__(self, map_obj, step_size=1, max_iterations=10000, goal_sample_rate=0.2):
        self.map_obj = map_obj
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate

    def _get_random_node(self):
        if random.random() < self.goal_sample_rate:
            return Node(self.map_obj.goal[0], self.map_obj.goal[1], self.map_obj.goal[2])
        x = random.uniform(self.map_obj.start[0], self.map_obj.goal[0])
        y = random.uniform(self.map_obj.start[1], self.map_obj.goal[1])
        z = random.uniform(self.map_obj.start[2], self.map_obj.goal[2])
        return Node(x, y, z)

    def _get_nearest_node(self, node_list, random_node):
        distances = [
            np.sqrt(
                (node.x - random_node.x) ** 2 +
                (node.y - random_node.y) ** 2 +
                (node.z - random_node.z) ** 2
            )
            for node in node_list
        ]
        nearest_index = np.argmin(distances)
        return node_list[nearest_index]

    def _is_collision_free(self, node1, node2):
        resolution = 10  # Number of steps to interpolate
        x_vals = np.linspace(node1.x, node2.x, resolution)
        y_vals = np.linspace(node1.y, node2.y, resolution)
        z_vals = np.linspace(node1.z, node2.z, resolution)

        for obstacle in self.map_obj.obstacles:
            for x, y, z in zip(x_vals, y_vals, z_vals):
                if obstacle.contains(x, y, z):
                    return False
        return True

    def _get_path(self, end_node):
        path = []
        node = end_node
        while node is not None:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]

    def rrt(self):
        node_list = [Node(self.map_obj.start[0], self.map_obj.start[1], self.map_obj.start[2])]
        for _ in range(self.max_iterations):
            random_node = self._get_random_node()
            nearest_node = self._get_nearest_node(node_list, random_node)
            theta_xy = np.arctan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
            theta_z = np.arctan2(random_node.z - nearest_node.z, np.sqrt((random_node.x - nearest_node.x) ** 2 + (random_node.y - nearest_node.y) ** 2))

            new_x = nearest_node.x + self.step_size * np.cos(theta_xy) * np.cos(theta_z)
            new_y = nearest_node.y + self.step_size * np.sin(theta_xy) * np.cos(theta_z)
            new_z = nearest_node.z + self.step_size * np.sin(theta_z)

            new_node = Node(new_x, new_y, new_z, nearest_node)

            if self._is_collision_free(nearest_node, new_node):
                node_list.append(new_node)
                if np.sqrt(
                    (new_node.x - self.map_obj.goal[0]) ** 2 +
                    (new_node.y - self.map_obj.goal[1]) ** 2 +
                    (new_node.z - self.map_obj.goal[2]) ** 2
                ) <= self.step_size:
                    return self._get_path(new_node)
        return None

    def rrt_star(self, radius=10):
        node_list = [Node(self.map_obj.start[0], self.map_obj.start[1], self.map_obj.start[2])]
        for _ in range(self.max_iterations):
            random_node = self._get_random_node()
            nearest_node = self._get_nearest_node(node_list, random_node)
            theta_xy = np.arctan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
            theta_z = np.arctan2(random_node.z - nearest_node.z, np.sqrt((random_node.x - nearest_node.x) ** 2 + (random_node.y - nearest_node.y) ** 2))

            new_x = nearest_node.x + self.step_size * np.cos(theta_xy) * np.cos(theta_z)
            new_y = nearest_node.y + self.step_size * np.sin(theta_xy) * np.cos(theta_z)
            new_z = nearest_node.z + self.step_size * np.sin(theta_z)

            new_node = Node(new_x, new_y, new_z, nearest_node)

            if self._is_collision_free(nearest_node, new_node):
                new_node.cost = nearest_node.cost + np.sqrt(
                    (new_node.x - nearest_node.x) ** 2 +
                    (new_node.y - nearest_node.y) ** 2 +
                    (new_node.z - nearest_node.z) ** 2
                )
                node_list.append(new_node)

                for node in node_list:
                    if np.sqrt(
                        (node.x - new_node.x) ** 2 +
                        (node.y - new_node.y) ** 2 +
                        (node.z - new_node.z) ** 2
                    ) < radius and self._is_collision_free(node, new_node):
                        new_cost = new_node.cost + np.sqrt(
                            (node.x - new_node.x) ** 2 +
                            (node.y - new_node.y) ** 2 +
                            (node.z - new_node.z) ** 2
                        )
                        if new_cost < node.cost:
                            node.parent = new_node
                            node.cost = new_cost

                if np.sqrt(
                    (new_node.x - self.map_obj.goal[0]) ** 2 +
                    (new_node.y - self.map_obj.goal[1]) ** 2 +
                    (new_node.z - self.map_obj.goal[2]) ** 2
                ) <= self.step_size:
                    return self._get_path(new_node)
        return None

class Obstacle:
    def contains(self, x, y, z):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def intersects(self, p1, p2):
        raise NotImplementedError("This method should be overridden by subclasses.")

class Sphere(Obstacle):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def contains(self, x, y, z):
        return np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2) <= self.radius

    def intersects(self, p1, p2):
        resolution = 10  # Number of interpolation steps
        x_vals = np.linspace(p1[0], p2[0], resolution)
        y_vals = np.linspace(p1[1], p2[1], resolution)
        z_vals = np.linspace(p1[2], p2[2], resolution)
        for x, y, z in zip(x_vals, y_vals, z_vals):
            if self.contains(x, y, z):
                return True
        return False

class Box(Obstacle):
    def __init__(self, min_corner, max_corner):
        self.min_corner = min_corner
        self.max_corner = max_corner

    def contains(self, x, y, z):
        return (
            self.min_corner[0] <= x <= self.max_corner[0] and
            self.min_corner[1] <= y <= self.max_corner[1] and
            self.min_corner[2] <= z <= self.max_corner[2]
        )

    def intersects(self, p1, p2):
        resolution = 10  # Number of interpolation steps
        x_vals = np.linspace(p1[0], p2[0], resolution)
        y_vals = np.linspace(p1[1], p2[1], resolution)
        z_vals = np.linspace(p1[2], p2[2], resolution)
        for x, y, z in zip(x_vals, y_vals, z_vals):
            if self.contains(x, y, z):
                return True
        return False

class SDF:
    def __init__(self, map_obj):
        self.map_obj = map_obj

    def compute_environment_map(self):
        grid_size = 50  # Resolution of the 3D map
        x = np.linspace(self.map_obj.x_limits[0], self.map_obj.x_limits[1], grid_size)
        y = np.linspace(self.map_obj.y_limits[0], self.map_obj.y_limits[1], grid_size)
        z = np.linspace(self.map_obj.z_limits[0], self.map_obj.z_limits[1], grid_size)
        env_map = np.ones((grid_size, grid_size, grid_size))

        for obstacle in self.map_obj.obstacles:
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        if obstacle.contains(x[i], y[j], z[k]):
                            env_map[i, j, k] = 0
        return env_map

    def visualize(self, env_map, paths=None):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        grid_size = env_map.shape[0]
        x, y, z = np.indices((grid_size, grid_size, grid_size))
        filled = env_map == 0

        ax.voxels(filled, edgecolor='k', alpha=0.5)

        if paths:
            for path, color, name in paths:
                path_np = np.array(path)
                ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2], '-o', color=color, label=f'{name} Path')
            ax.legend()

        ax.set_title("3D Environment Map with Paths")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

class Map3D:
    def __init__(self, obstacles, start, goal, x_limits, y_limits, z_limits):
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits

# Example usage in 3D
if __name__ == "__main__":
    x_limits = [0, 40]
    y_limits = [0, 40]
    z_limits = [0, 40]

    obstacles = [
        Sphere(center=(20, 20, 20), radius=10),
        Box(min_corner=(5, 5, 5), max_corner=(15, 15, 15))
    ]

    start = (0, 0, 0)
    goal = (40, 40, 40)

    map_obj = Map3D(obstacles, start, goal, x_limits, y_limits, z_limits)
    planner = PathPlanning(map_obj)

    path_rrt = planner.rrt()
    path_rrt_star = planner.rrt_star()

    sdf = SDF(map_obj)
    environment_map = sdf.compute_environment_map()

    paths = []
    if path_rrt:
        for vertice in path_rrt:
            for v in vertice: 
                print(v, end = " ")
            print()
        paths.append((path_rrt, 'red', 'RRT'))
    #if path_rrt_star:
    #    print("RRT* Path:", path_rrt_star)
    #    paths.append((path_rrt_star, 'blue', 'RRT*'))

    # Print Box corners
    for obstacle in obstacles:
        if isinstance(obstacle, Box):
            print("Box Corners:", obstacle.min_corner, "to", obstacle.max_corner)

    sdf.visualize(environment_map, paths)
