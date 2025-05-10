import open3d as o3d
import pyvista as pv
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R, Slerp

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from Environment import EnvironmentHandler as CollisionEnvironment3D


class RRTState:
    def __init__(self, x: np.ndarray, q: np.ndarray, i: int = 0):
        self._x = x
        self._q = q
        self._i = i

    @property
    def x(self):
        return self._x

    @property
    def q(self):
        return self._q

    @property
    def i(self):
        return self._i

    @x.setter
    def x(self, x: np.ndarray):
        self._x = x

    @q.setter
    def q(self, q: np.ndarray):
        self._q = q

    @i.setter
    def i(self, i: int):
        self._i = i

    def distance_to(self, other):
        pos_dist = np.linalg.norm(self.x - other.x)
        quat_dist = 1 - np.abs(np.dot(self.q, other.q))
        return pos_dist + quat_dist


class RRTPlanner3D:
    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent

    def __init__(
        self,
        env: CollisionEnvironment3D,
        step_size=0.3,
        goal_tolerance=0.3,
        goal_sample_rate=0.4,
        max_iters=10000,
    ):
        self.env = env
        self.robot = env.buildEllipsoid()

        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.path = []

    def _interpolate(self, s1, s2, alpha):
        x = (1 - alpha) * s1.x + alpha * s2.x
        slerp = Slerp([0, 1], R.from_quat([s1.q, s2.q]))
        q = slerp([alpha])[0].as_quat()
        return RRTState(x=x, q=q)

    def _is_valid(self, state):
        return self.env.numCollisions(self.robot, state.x, state.q) == 0

    def _motion_valid(self, from_state, to_state):
        dist = from_state.distance_to(to_state)
        steps = int(np.ceil(dist / (self.step_size / 2)))
        for i in range(steps + 1):
            alpha = i / steps
            if not self._is_valid(self._interpolate(from_state, to_state, alpha)):
                return False
        return True

    def _sample(self, goal):
        if np.random.rand() < self.goal_sample_rate:
            return goal
        pos = np.random.uniform([0, -5, 0], [10, 20, 2])
        quat = R.random().as_quat()
        return RRTState(pos, quat)

    def _reconstruct(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def plan(self, start, goal):
        tree = [self.Node(start)]
        for _ in range(self.max_iters):
            rand = self._sample(goal)
            nearest = min(tree, key=lambda n: n.state.distance_to(rand))
            new = self._interpolate(
                nearest.state,
                rand,
                min(self.step_size / nearest.state.distance_to(rand), 1.0),
            )
            if self._motion_valid(nearest.state, new):
                node = self.Node(new, nearest)
                tree.append(node)
                if new.distance_to(goal) < self.goal_tolerance:
                    return self._reconstruct(node)
        return []

    def run(self):
        start = RRTState(np.array([0, -3, 1.0]), np.array([0, 0, 0, 1]))
        goal = RRTState(np.array([20.0, 15.0, 1.0]), np.array([0, 0, 0, 1]))

        print("🚀 Planning...")
        self.path = self.plan(start, goal)

        if self.path:
            print(f"✅ Found path with {len(self.path)} states.")
        else:
            print("❌ No path found.")

    def get_path(self):
        return self.path

    def visualize_path(self):
        plotter = pv.Plotter()
        plotter.add_mesh(self.env.voxel_mesh, color="red", opacity=0.4)

        if self.path:
            for s in self.path:
                mesh = pv.ParametricEllipsoid(0.24, 0.24, 0.1)
                T = np.eye(4)
                T[:3, :3] = R.from_quat(s.q).as_matrix()
                T[:3, 3] = s.x
                mesh.transform(T, inplace=True)
                plotter.add_mesh(mesh, color="lime", opacity=1.0)

        plotter.add_axes_at_origin()
        plotter.add_axes()

        plotter.show()


def main():
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pcd_path = os.path.join(script_dir, "map.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)

    env = CollisionEnvironment3D(pcd)
    planner = RRTPlanner3D(env)
    planner.run()
    planner.visualize_path()


if __name__ == "__main__":
    main()
