import numpy as np
from scipy.spatial import cKDTree, KDTree
import random


class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent


class RRT:
    def __init__(self, start, goal, map_instance, step_size=0.5, max_iterations=1000):
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
        return np.linalg.norm(node.position - self.goal.position) < self.step_size

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
