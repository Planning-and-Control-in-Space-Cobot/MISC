import shapely.geometry as gm
import numpy as np

import matplotlib.pyplot as plt


class Map:
    """
    A class to represent a 2D map with obstacles.

    Attributes:
        size_x (int): The width of the map.
        size_y (int): The height of the map.
        start (tuple): The starting position of the robot as a tuple (x, y).
        goal (tuple): The goal position of the robot as a tuple (x, y).
        obstacles (list): A list of shapely.geometry objects representing the obstacles in the map.
        map (np.array): A NumPy array representing the map with obstacles.
    """

    def __init__(
        self,
        size_x=200,
        size_y=200,
        start=(0, 0),
        goal=(180, 150),
        obstacles=[],
    ):
        """
        Initialize the map with the given size, start, goal, and obstacles.

        Parameters:
            size_x (int): The width of the map.
            size_y (int): The height of the map.
            start (tuple): The starting position of the robot as a tuple (x, y).
            goal (tuple): The goal position of the robot as a tuple (x, y).
            obstacles (list): A list of shapely.geometry objects representing the obstacles in the map.

        Returns:
            Object: A new Map object.
        """
        self.size_x = size_x
        self.size_y = size_y
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.map = np.ones((self.size_x, self.size_y))
        self.construct_map()

    def reset_map(self):
        """
        Reset the map to its original state.

        Parameters:
            None

        Returns:
            None
        """

        self.map = np.ones((self.size_x, self.size_y))
        self.construct_map()

    def construct_map(self):
        """
        Construct the map with obstacles.

        Parameters:
            None

        Returns:
            None
        """
        for x in range(self.size_x):
            for y in range(self.size_y):
                point = gm.Point(x, y)
                for obstacle in self.obstacles:
                    if obstacle.contains(point):
                        self.change_map_value(x, y, 0.0)
                        break

    def get_map_image(self):
        return self.map

    def change_map_value(self, x, y, value):
        self.map[y, x] = value

    def contains(self, x, y):
        return 0 <= x < self.size_x and 0 <= y < self.size_y

    def plot_map(self):
        map_image = self.get_map_image().T
        plt.figure(figsize=(8, 6))
        plt.title("Map")
        plt.imshow(map_image, origin="lower", cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def plot_trajectory(self, trajectory=None):
        if trajectory is None:
            print("No trajectory provided.")
            return

        # Ensure the trajectory is a NumPy array for consistent handling
        if isinstance(trajectory, list):
            trajectory = np.array(trajectory)

        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                "Trajectory must be a list of tuples or a NumPy array with shape Nx2."
            )

        # Plot the map
        map_image = self.get_map_image()
        plt.figure(figsize=(8, 6))
        plt.title("Map with Trajectory")
        plt.imshow(map_image, origin="lower", cmap="gray")
        plt.xticks(np.linspace(0, self.size_x, 10))
        plt.yticks(np.linspace(0, self.size_y, 10))

        plt.xlabel("Map Coordinate X")
        plt.ylabel("Map Coordinate Y")

        # Extract x and y coordinates from the trajectory
        x_coords, y_coords = trajectory[:, 0], trajectory[:, 1]

        # Plot the trajectory
        plt.plot(x_coords, y_coords, color="red", linewidth=2, label="Trajectory")
        plt.scatter(
            self.start[0], self.start[1], color="green", label="Start", zorder=5
        )
        plt.scatter(self.goal[0], self.goal[1], color="blue", label="Goal", zorder=5)

        plt.legend()
