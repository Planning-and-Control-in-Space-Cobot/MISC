import numpy as np
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree


import casadi as ca

# Node class
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

# Map class using shapely geometries
class Map:
    def __init__(self, obstacles, start, goal, size=(200, 200)):
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.size = size

# Path Planning class with RRT using shapely
class PathPlanning:
    def __init__(self, map_obj, step_size=1, max_iterations=10000, goal_sample_rate=0.1):
        self.map_obj = map_obj
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate

    def _get_random_node(self):
        if random.random() < self.goal_sample_rate:
            return Node(self.map_obj.goal[0], self.map_obj.goal[1])
        x = random.uniform(0, 200)
        y = random.uniform(0, 200)
        return Node(x, y)

    def _get_nearest_node(self, node_list, random_node):
        distances = [np.hypot(node.x - random_node.x, node.y - random_node.y) for node in node_list]
        nearest_index = np.argmin(distances)
        return node_list[nearest_index]

    def _is_collision_free(self, node1, node2):
        # Check if a line segment intersects any obstacle
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

    def rrt(self):
        node_list = [Node(self.map_obj.start[0], self.map_obj.start[1])]
        explored_paths = []  # To store all explored paths

        for _ in range(self.max_iterations):
            random_node = self._get_random_node()
            nearest_node = self._get_nearest_node(node_list, random_node)
            theta = np.arctan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
            new_x = nearest_node.x + self.step_size * np.cos(theta)
            new_y = nearest_node.y + self.step_size * np.sin(theta)
            new_node = Node(new_x, new_y, nearest_node)

            if self._is_collision_free(nearest_node, new_node):
                node_list.append(new_node)
                explored_paths.append(((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)))

                if np.hypot(new_node.x - self.map_obj.goal[0], new_node.y - self.map_obj.goal[1]) <= self.step_size:
                    final_path = self._get_path(new_node)
                    return final_path, explored_paths

        return None, explored_paths

class TrajectoryOptimizer():
    def __init__(self, 
                 initial_path = None, 
                 SDF = None, 
                 origin = (0, 0), 
                 goal = (0, 0)): 
        self.initial_path = initial_path
        self.SDF = SDF
        self.origin = origin
        self.goal = goal

    def optimize(self):
        opti = ca.Opti() 

        # Variables
        x = opti.variable(2, len(self.initial_path))
        u = opti.variable(6, len(self.initial_path))

        A_ = np.load("A_matrix.npy")
        A_ = A_[0:2, :] # Only consider x and y components
        
        A = opti.parameter(2, 6)
        opti.set_value(A, A_)
        
        m = opti.parameter(1) # mass in kg
        opti.set_value(m, 5)

        opti.subject_to(ca.norm_2(x[:, 0] - self.initial_path[0] + 1e-5) <= 5)
        opti.subject_to(ca.norm_2(x[:, -1] - self.goal + 1e-5) <= 5)
        for i in range(1, len(self.initial_path)):
            opti.subject_to(x[:, i] == x[:, i-1] + (A @ u[:, i]) / m)
        
        for i in range(len(self.initial_path)):
            opti.subject_to(opti.bounded(-3, u[:, i], 3))
            


        cost = 0
        for i in range(len(self.initial_path)):
            cost += ca.sumsqr(u[:, i])

        opti.minimize(cost)

        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000}

        opti.solver("ipopt", p_opts, s_opts)

        # Number of variables and constraints
        num_vars = opti.nx
        num_constraints = opti.ng

        print("Number of variables:", num_vars)
        print("Number of constraints:", num_constraints)


        opti.set_initial(x, np.array(self.initial_path).T)

        try:
            sol = opti.solve()
        except:
            print("Optimization failed")


        ## Debug
        print(opti.debug.value(x))
        print(opti.debug.value(u))
        print(self.goal)
        print(self.origin)

# Visualization using matplotlib
class SDF:
    def __init__(self, map_obj):
        self.map_obj = map_obj

    def min_distance_to_obstacle(self):
        map = np.zeros(self.map_obj.size)
        # Create a spatial index for obstacles
        obstacle_tree = STRtree(self.map_obj.obstacles)

        # Create the grid points
        x_coords, y_coords = np.meshgrid(np.arange(self.map_obj.size[0]), np.arange(self.map_obj.size[1]), indexing='ij')

        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]

        # Compute minimum distances using the spatial index
        distances = np.array([
            min(point.distance(obstacle) for obstacle in obstacle_tree.query(point))
            for point in points
        ])

        # Reshape to grid size
        distances = distances.reshape(self.map_obj.size[0], self.map_obj.size[1])


    def visualize(self, explored_paths, final_path, env_map):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot paths and obstacles
        ax = axes[0]
        for obstacle in self.map_obj.obstacles:
            x, y = (obstacle.exterior.xy if isinstance(obstacle, Polygon) else obstacle.boundary.xy)
            ax.fill(x, y, color='gray')

        ax.plot(self.map_obj.start[0], self.map_obj.start[1], 'go', label='Start')
        ax.plot(self.map_obj.goal[0], self.map_obj.goal[1], 'ro', label='Goal')

        for segment in explored_paths:
            x_values, y_values = zip(*segment)
            ax.plot(x_values, y_values, color='red', alpha=0.5)

        if final_path:
            path_np = np.array(final_path)
            ax.plot(path_np[:, 0], path_np[:, 1], '-o', color='green', label='Final Path')

        ax.legend()
        ax.set_title("Path Planning Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        # Plot SDF
        ax = axes[1]
        ax.imshow(env_map, cmap='coolwarm', origin='lower')
        ax.set_title("SDF of the Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Define environment limits
    x_limits = [0, 200]
    y_limits = [0, 200]

    # Define obstacles using shapely
    obstacles = [
        Point(50, 50).buffer(40),
        Point(150, 125).buffer(35),
        Polygon([(0, 100), (90, 100), (90, 110), (0,110 )]),
        Polygon([(100, 20), (100, 100), (120, 100), (120, 20)])
    ]

    # Define start and goal points
    start = (0, 0)
    goal = (175, 175)

    # Create Map instance
    map_obj = Map(obstacles, start, goal)

    # Perform Path Planning
    #planner = PathPlanning(map_obj)
    #final_path, explored_paths = planner.rrt()

    # Compute SDF
    sdf = SDF(map_obj)
    #env_map = sdf.compute_environment_map()

    # Visualize results
#    sdf.visualize(explored_paths, final_path, env_map)
    map = sdf.min_distance_to_obstacle()
    plt.imshow(map)
    plt.show()

