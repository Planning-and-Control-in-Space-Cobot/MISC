import open3d as o3d
import pyvista as pv
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R, Slerp
import scipy.spatial.transform as trf

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from Environment import EnvironmentHandler as CollisionEnvironment3D

from typing import List, Tuple

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
        robot : fcl.CollisionObject, 
        payload : fcl.CollisionObject = None, 
        translation : np.ndarray = np.array([0, 0, 0]), 
        originalPayloadAttitude : trf.Rotation  = trf.Rotation.from_quat([0, 0, 0, 1]),
        step_size=0.3,
        goal_tolerance=0.3,
        goal_sample_rate=0.4,
        max_iters=10000,
        numTriesSampling = 100,
        posMin : List[int]= [0, -5, 0], 
        posMax : List[int]= [10, 20, 2]
    ):
        '''
        Initialized the RRT planner

        Parameters:
            env (CollisionEnvironment3D): The environment containing the obstacles in order to check for collisions.
            robot (fcl.CollisionObject): Collision object representing the robot
            payload (fcl.CollisionObject, optional): Collision object representing the payload attached to the robot. If value if None then no payload is considered for the trajectory. If this is provided then the translation vector must also be specified.
            translation (np.ndarray, optional): Translation vector for the robot to the payload. If the this is provided then the payload needs to be specified as well.
            originalPayloadAttitude (scipy.spatial.transform.Rotation, optional): The original attitude of the payload. Default is a quaternion representing no rotation.
            step_size (float, optional): Step size for the RRT algorithm. Default is 0.3.
            goal_tolerance (float, optional): Tolerance for reaching the goal. Default is 0.3.
            goal_sample_rate (float, optional): Probability of sampling the goal state. Default is 0.4.
            max_iters (int, optional): Maximum number of iterations for the RRT algorithm. Default is 10000.
            numTriesSampling (int, optional): Maximum number of tried when sampling a random position before sampling the goal node
            posMin (List[int], optional): Map minimum value for the random sampling
            posMax (List[int], optional)

        '''
        if payload is not None and translation == np.array([0, 0, 0]):
            raise ValueError("Translation vector must be specified to attach the payload.")

        if payload is None:
            self.considerPayload = False
        else:
            self.considerPayload = True
     
        self.payload = payload
        self.translation = translation
        self.originalPayloadAttitude = originalPayloadAttitude

        self.robot = robot
        self.env = env

        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.numTriesSampling = numTriesSampling
        self.posMin = posMin 
        self.posMax = posMax
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
        '''
        Generates a random state in the environment that is not in collision with the obstacle.
        
        Parameters:
            goal (RRTState): The goal state to sample from with a probability of goal_sample_rate. 
        
        Returns:
            RRTState: A random state in the environment that is not in collision with the obstacle.
        '''
        if np.random.rand() < self.goal_sample_rate:
            return goal
        counter = 0
        while True:
            pos = np.random.uniform(self.posMin, self.posMax)
            quat = R.random().as_quat()

            numCollisions = self.env.numCollisions(self.robot, pos, quat)
            if self.considerPayload:
                numCollisions += self.env.numCollisions(self.payload, pos + self.translation, quat)

            if numCollisions == 0:
                return RRTState(pos, quat)
            
            if counter >= self.numTriesSampling:
                return goal

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

    def visualize_path(self, path : List[RRTState] = None):
        '''
        Visualized the path in the environment using Pyvista.

        If the path is not provided we will simply be visualizing the environment.

        Parameters:
            path (List[RRTState], optional): The path to visualize. If None, only the environment will be visualized.
            
        Returns:
            None
        '''
        plotter = pv.Plotter()
        plotter.add_mesh(self.env.voxel_mesh, color="red", opacity=0.4)

        mesh = pv.ParametricEllipsoid(0.24, 0.24, 0.1)
        if path:
            for s in path:
                m = mesh.copy()
                T = np.eye(4)
                T[:3, :3] = R.from_quat(s.q).as_matrix()
                T[:3, 3] = s.x
                m.transform(T, inplace=True)
                plotter.add_mesh(m, color="blue", opacity=0.4)

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
