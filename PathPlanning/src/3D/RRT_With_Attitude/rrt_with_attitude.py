import numpy as np
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class Node:
    def __init__(self, state, parent=None):
        self.state = state  # [x, y, z, qx, qy, qz, qw]
        self.parent = parent
        self.cost = 0.0


class RRTStarSE3:
    def __init__(self, start, goal, bounds, max_iter=300, step_size=1.0, goal_sample_rate=0.1, radius=2.0):
        self.start = Node(start)
        self.goal = Node(goal)
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.radius = radius
        self.nodes = [self.start]

    def distance(self, state1, state2):
        pos1, quat1 = np.array(state1[:3]), np.array(state1[3:])
        pos2, quat2 = np.array(state2[:3]), np.array(state2[3:])
        d_pos = np.linalg.norm(pos1 - pos2)
        dot = np.abs(np.dot(quat1, quat2))
        d_quat = 2 * np.arccos(np.clip(dot, -1.0, 1.0))  # angular distance
        return d_pos + d_quat

    def sample(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal.state
        pos = [np.random.uniform(low, high) for (low, high) in self.bounds]
        quat = R.random().as_quat()  # [x, y, z, w]
        return pos + list(quat)

    def nearest(self, sampled_state):
        dists = [self.distance(n.state, sampled_state) for n in self.nodes]
        return self.nodes[np.argmin(dists)]

    def steer(self, from_state, to_state):
        pos1, pos2 = np.array(from_state[:3]), np.array(to_state[:3])
        quat1, quat2 = np.array(from_state[3:]), np.array(to_state[3:])
        dir_vec = pos2 - pos1
        dist = np.linalg.norm(dir_vec)
        if dist == 0:
            return from_state

        alpha = min(self.step_size / dist, 1.0)
        new_pos = pos1 + alpha * dir_vec

        rot1 = R.from_quat(quat1)
        rot2 = R.from_quat(quat2)
        slerp = Slerp([0, 1], R.concatenate([rot1, rot2]))
        new_quat = slerp([alpha])[0].as_quat()

        return list(new_pos) + list(new_quat)

    def collision_free(self, state1, state2):
        return True  # Replace with collision checking if needed

    def get_nearby_nodes(self, new_node):
        n = len(self.nodes)
        radius = min(self.radius * np.sqrt(np.log(n + 1) / (n + 1)), self.step_size * 5)
        return [node for node in self.nodes if self.distance(node.state, new_node.state) < radius]

    def plan(self):
        for _ in range(self.max_iter):
            sampled_state = self.sample()
            nearest_node = self.nearest(sampled_state)
            new_state = self.steer(nearest_node.state, sampled_state)
            if not self.collision_free(nearest_node.state, new_state):
                continue
            new_node = Node(new_state, parent=nearest_node)
            new_node.cost = nearest_node.cost + self.distance(nearest_node.state, new_state)

            neighbors = self.get_nearby_nodes(new_node)
            for neighbor in neighbors:
                potential_cost = neighbor.cost + self.distance(neighbor.state, new_node.state)
                if self.collision_free(neighbor.state, new_node.state) and potential_cost < new_node.cost:
                    new_node.parent = neighbor
                    new_node.cost = potential_cost

            self.nodes.append(new_node)

            for neighbor in neighbors:
                potential_cost = new_node.cost + self.distance(new_node.state, neighbor.state)
                if self.collision_free(new_node.state, neighbor.state) and potential_cost < neighbor.cost:
                    neighbor.parent = new_node
                    neighbor.cost = potential_cost

            if self.distance(new_node.state, self.goal.state) < self.step_size:
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + self.distance(new_node.state, self.goal.state)
                self.nodes.append(self.goal)
                break

        return self.reconstruct_path()

    def reconstruct_path(self):
        path = []
        node = self.goal if self.goal in self.nodes else min(self.nodes, key=lambda n: self.distance(n.state, self.goal.state))
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]


def visualize(rrt_star, path, start, goal, bounds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_box(ax, box, color='black', alpha=0.3):
        min_pt = np.array(box['min'])
        max_pt = np.array(box['max'])
        dx, dy, dz = max_pt - min_pt
        ax.bar3d(min_pt[0], min_pt[1], min_pt[2], dx, dy, dz, color=color, alpha=alpha)


    # Plot tree edges
    for node in rrt_star.nodes:
        if node.parent:
            p1 = node.state[:3]
            p2 = node.parent.state[:3]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'gray', linewidth=0.5)

    # Plot path with orientation
    for state in path:
        pos = np.array(state[:3])
        quat = np.array(state[3:])
        rot = R.from_quat(quat)
        direction = rot.apply([1, 0, 0])  # local x-axis
        ax.quiver(pos[0], pos[1], pos[2],
                  direction[0], direction[1], direction[2],
                  length=0.5, color='red')

    for obs in rrt_star.obstacles:
        if obs['type'] == 'box':
            draw_box(ax, obs)

    # Draw final path line
    path_positions = np.array([state[:3] for state in path])
    ax.plot(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2], 'r', linewidth=2, label='Path')

    # Plot start and goal
    ax.scatter(*start[:3], c='green', s=50, label='Start')
    ax.scatter(*goal[:3], c='blue', s=50, label='Goal')

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("RRT* in SE(3) with Orientation")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    start = [0, 0, 0, 0, 0, 0, 1]       # [x, y, z, qx, qy, qz, qw]
    goal = [5, 5, 5, 0, 0, 0, 1]
    bounds = [(-10, 10), (-10, 10), (-10, 10)]

    planner = RRTStarSE3(start, goal, bounds, max_iter=300, step_size=1.0)
    path = planner.plan()
    visualize(planner, path, start, goal, bounds)

