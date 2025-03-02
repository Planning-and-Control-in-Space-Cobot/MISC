import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon

import matplotlib.pyplot as plt


# Node class
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = 0.0  # Initialize cost


# Path Planning class with RRT and RRT*
class PathPlanning:
    def __init__(
        self,
        map_obj,
        step_size=3,
        max_iterations=10000,
        goal_sample_rate=0.5,
        radius=5,
    ):
        self.map_obj = map_obj
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.radius = radius

    def _get_random_node(self):
        if random.random() < self.goal_sample_rate:
            return Node(self.map_obj.goal[0], self.map_obj.goal[1])
        x = random.uniform(0, self.map_obj.size_x)
        y = random.uniform(0, self.map_obj.size_y)
        return Node(x, y)

    def _get_nearest_node(self, node_list, random_node):
        distances = [
            np.hypot(node.x - random_node.x, node.y - random_node.y)
            for node in node_list
        ]
        nearest_index = np.argmin(distances)
        return node_list[nearest_index]

    def _is_collision_free(self, node1, node2):
        line = LineString([(node1.x, node1.y), (node2.x, node2.y)])
        for obstacle in self.map_obj.obstacles:
            if line.intersects(obstacle):
                return False
        return True

    def _get_path(self, end_node):
        path = []
        node = end_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

    def _get_near_nodes(self, node_list, new_node):
        near_nodes = []
        for node in node_list:
            if (
                np.hypot(node.x - new_node.x, node.y - new_node.y)
                <= self.radius
            ):
                near_nodes.append(node)
        return near_nodes

    def rrt(self):
        node_list = [Node(self.map_obj.start[0], self.map_obj.start[1])]
        explored_paths = []

        for _ in range(self.max_iterations):
            random_node = self._get_random_node()
            nearest_node = self._get_nearest_node(node_list, random_node)
            theta = np.arctan2(
                random_node.y - nearest_node.y, random_node.x - nearest_node.x
            )
            new_x = nearest_node.x + self.step_size * np.cos(theta)
            new_y = nearest_node.y + self.step_size * np.sin(theta)
            new_node = Node(new_x, new_y, nearest_node)

            if self._is_collision_free(nearest_node, new_node):
                node_list.append(new_node)
                explored_paths.append(
                    ((nearest_node.x, nearest_node.y), (new_node.x, new_node.y))
                )

                if (
                    np.hypot(
                        new_node.x - self.map_obj.goal[0],
                        new_node.y - self.map_obj.goal[1],
                    )
                    <= self.step_size
                ):
                    final_path = self._get_path(new_node)
                    return final_path, explored_paths

        return None, explored_paths

    def rrt_star(self):
        node_list = [Node(self.map_obj.start[0], self.map_obj.start[1])]
        explored_paths = []

        for _ in range(self.max_iterations):
            random_node = self._get_random_node()
            nearest_node = self._get_nearest_node(node_list, random_node)
            theta = np.arctan2(
                random_node.y - nearest_node.y, random_node.x - nearest_node.x
            )
            new_x = nearest_node.x + self.step_size * np.cos(theta)
            new_y = nearest_node.y + self.step_size * np.sin(theta)
            new_node = Node(new_x, new_y, nearest_node)

            if self._is_collision_free(nearest_node, new_node):
                near_nodes = self._get_near_nodes(node_list, new_node)
                min_cost = nearest_node.cost + np.hypot(
                    new_node.x - nearest_node.x, new_node.y - nearest_node.y
                )
                best_node = nearest_node

                for near_node in near_nodes:
                    cost = near_node.cost + np.hypot(
                        new_node.x - near_node.x, new_node.y - near_node.y
                    )
                    if cost < min_cost and self._is_collision_free(
                        near_node, new_node
                    ):
                        min_cost = cost
                        best_node = near_node

                new_node.parent = best_node
                new_node.cost = min_cost
                node_list.append(new_node)
                explored_paths.append(
                    ((best_node.x, best_node.y), (new_node.x, new_node.y))
                )

                for near_node in near_nodes:
                    cost = new_node.cost + np.hypot(
                        new_node.x - near_node.x, new_node.y - near_node.y
                    )
                    if cost < near_node.cost and self._is_collision_free(
                        new_node, near_node
                    ):
                        near_node.parent = new_node
                        near_node.cost = cost

                if (
                    np.hypot(
                        new_node.x - self.map_obj.goal[0],
                        new_node.y - self.map_obj.goal[1],
                    )
                    <= self.step_size
                ):
                    final_path = self._get_path(new_node)
                    return final_path, explored_paths

        return None, explored_paths

    def plot_map_with_paths(self, explored_paths, final_path):
        map_image = self.map_obj.get_map_image()

        plt.figure(figsize=(8, 6))
        plt.title("Map with RRT* Paths")
        plt.imshow(map_image, origin="lower", cmap="gray")

        # Plot explored paths in red
        for path in explored_paths:
            plt.plot(
                [path[0][0], path[1][0]],
                [path[0][1], path[1][1]],
                "r-",
                linewidth=0.5,
            )

        # Plot final path with 'X' markers
        x_coords = [x for x, _ in final_path]
        y_coords = [y for _, y in final_path]
        plt.plot(x_coords, y_coords, "g-", linewidth=2)
        plt.scatter(x_coords, y_coords, marker="X", color="green", s=50)

        plt.scatter(
            self.map_obj.start[0],
            self.map_obj.start[1],
            color="b",
            marker="o",
            label="Start",
        )
        plt.scatter(
            self.map_obj.goal[0],
            self.map_obj.goal[1],
            color="g",
            marker="o",
            label="Goal",
        )

        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.show()
