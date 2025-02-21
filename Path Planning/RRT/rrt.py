import numpy as np
import matplotlib.pyplot as plt
import random
import time
from shapely.geometry import Polygon, Point, LineString

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = float('inf')

def distance(node1, node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def is_collision_free(node1, node2, obstacles):
    line = LineString([(node1.x, node1.y), (node2.x, node2.y)])
    for obstacle in obstacles:
        if line.intersects(obstacle):
            return False
    return True

def get_nearest_node(tree, rnd_node):
    return min(tree, key=lambda node: distance(node, rnd_node))

def get_nearby_nodes(tree, new_node, radius):
    return [node for node in tree if distance(node, new_node) <= radius]

def reconstruct_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]

def rrt(start, goal, obstacles, x_limits, y_limits, max_iter=1000, step_size=5.0, goal_radius=5.0):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    tree = [start_node]

    for _ in range(max_iter):
        rnd_x = random.uniform(x_limits[0], x_limits[1])
        rnd_y = random.uniform(y_limits[0], y_limits[1])
        rnd_node = Node(rnd_x, rnd_y)

        nearest_node = get_nearest_node(tree, rnd_node)
        theta = np.arctan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)

        new_x = nearest_node.x + step_size * np.cos(theta)
        new_y = nearest_node.y + step_size * np.sin(theta)
        new_node = Node(new_x, new_y)

        if not is_collision_free(nearest_node, new_node, obstacles):
            continue

        new_node.parent = nearest_node
        tree.append(new_node)

        if distance(new_node, goal_node) <= goal_radius:
            goal_node.parent = new_node
            return reconstruct_path(goal_node)

    return None

def rrt_star(start, goal, obstacles, x_limits, y_limits, max_iter=1000, step_size=5.0, goal_radius=5.0, search_radius=10.0):
    start_node = Node(start[0], start[1])
    start_node.cost = 0
    goal_node = Node(goal[0], goal[1])

    tree = [start_node]
    gamma_rrt = search_radius
    for i in range(max_iter):
        rnd_x = random.uniform(x_limits[0], x_limits[1])
        rnd_y = random.uniform(y_limits[0], y_limits[1])
        rnd_node = Node(rnd_x, rnd_y)

        nearest_node = get_nearest_node(tree, rnd_node)
        theta = np.arctan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)

        new_x = nearest_node.x + step_size * np.cos(theta)
        new_y = nearest_node.y + step_size * np.sin(theta)
        new_node = Node(new_x, new_y)

        if not is_collision_free(nearest_node, new_node, obstacles):
            continue

        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + distance(nearest_node, new_node)

        radius = min(gamma_rrt * (np.log(len(tree)) / len(tree))**0.5, step_size * 2)
        nearby_nodes = get_nearby_nodes(tree, new_node, radius)

        for nearby_node in nearby_nodes:
            if is_collision_free(nearby_node, new_node, obstacles):
                new_cost = nearby_node.cost + distance(nearby_node, new_node)
                if new_cost < new_node.cost:
                    new_node.parent = nearby_node
                    new_node.cost = new_cost

        tree.append(new_node)

        for nearby_node in nearby_nodes:
            if is_collision_free(new_node, nearby_node, obstacles):
                new_cost = new_node.cost + distance(new_node, nearby_node)
                if new_cost < nearby_node.cost:
                    nearby_node.parent = new_node
                    nearby_node.cost = new_cost

        if distance(new_node, goal_node) <= goal_radius:
            goal_node.parent = new_node
            goal_node.cost = new_node.cost + distance(new_node, goal_node)
            tree.append(goal_node)
            return reconstruct_path(goal_node)

    return None

def compute_signed_distance_field(x_limits, y_limits, obstacles, resolution=1.0):
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    x_range = np.arange(x_min, x_max + resolution, resolution)
    y_range = np.arange(y_min, y_max + resolution, resolution)
    sdf = np.zeros((len(y_range), len(x_range)))

    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            point = Point(x, y)
            min_distance = float('inf')
            inside = False
            for obstacle in obstacles:
                distance = obstacle.exterior.distance(point)
                if obstacle.contains(point):
                    inside = True
                min_distance = min(min_distance, distance)
            sdf[i, j] = -min_distance if inside else min_distance

    return sdf, x_range, y_range

# Define the environment
x_limits = [0, 100]
y_limits = [0, 100]
obstacles = [
    Polygon([(20, 20), (40, 20), (40, 40), (20, 40)]),
    Polygon([(60, 60), (80, 60), (80, 80), (60, 80)]), 
    Polygon([(45, 5), (45, 95), (55, 95), (55, 5)])
]

# Run Simple RRT
start = (10, 10)
goal = (90, 90)

start_time_rrt = time.time()
path_rrt = rrt(start, goal, obstacles, x_limits, y_limits)
end_time_rrt = time.time()

# Run RRT*
start_time_rrt_star = time.time()
path_rrt_star = rrt_star(start, goal, obstacles, x_limits, y_limits)
end_time_rrt_star = time.time()

print(f"Simple RRT Time: {end_time_rrt - start_time_rrt:.2f} seconds")
print(f"RRT* Time: {end_time_rrt_star - start_time_rrt_star:.2f} seconds")

# Visualization of RRT and RRT*
fig, ax = plt.subplots()
ax.set_xlim(x_limits)
ax.set_ylim(y_limits)

for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    ax.fill(x, y, color='gray')

if path_rrt:
    path_rrt = np.array(path_rrt)
    ax.plot(path_rrt[:, 0], path_rrt[:, 1], '-o', label="RRT Path")
    ax.scatter(start[0], start[1], c='green', label="Start")
    ax.scatter(goal[0], goal[1], c='red', label="Goal")
else:
    print("RRT Path not found.")

if path_rrt_star:
    path_rrt_star = np.array(path_rrt_star)
    ax.plot(path_rrt_star[:, 0], path_rrt_star[:, 1], '-x', label="RRT* Path")
else:
    print("RRT* Path not found.")

ax.legend()
plt.show()

# Compute and Visualize SDF
resolution = 1.0
sdf, x_range, y_range = compute_signed_distance_field(x_limits, y_limits, obstacles, resolution)

fig, ax = plt.subplots()
c = ax.imshow(sdf, extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]), origin='lower', cmap='coolwarm')
ax.set_title("Signed Distance Field")
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig.colorbar(c, ax=ax, label="Signed Distance")
plt.show()
