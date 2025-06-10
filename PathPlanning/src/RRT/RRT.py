import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
import fcl
from scipy.spatial.transform import Rotation as R, Slerp
import scipy.spatial.transform as trf
from typing import List, Optional

# Add Environment module path
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

    def distance_to(self, other: "RRTState"):
        pos_dist = np.linalg.norm(self.x - other.x)
        quat_dist = 1 - np.abs(np.dot(self.q, other.q))
        return pos_dist + quat_dist

_unset = object()

class RRTPath:
    def __init__(
        self,
        states: List[RRTState],
        robotMesh: Optional[pv.PolyData] = None,
        payloadMesh: Optional[pv.PolyData] = None,
        environmentMesh: Optional[pv.PolyData] = None,
        originalPayloadAttitude: trf.Rotation = trf.Rotation.from_quat([0, 0, 0, 1]),
        payloadTranslation: np.ndarray = np.array([0, 0, 0]),
        considerPayload: bool = False
    ):
        self.states = states
        self.robotMesh = robotMesh
        self.payloadMesh = payloadMesh
        self.environmentMesh = environmentMesh
        self.originalPayloadAttitude = originalPayloadAttitude
        self.payloadTranslation = payloadTranslation
        self.considerPayload = considerPayload

    def pathEmpty(self) -> bool:
        return len(self.states) == 0
    def getRobotMesh(self, state: RRTState) -> Optional[pv.PolyData]:
        if self.robotMesh is None:
            return None
        T = np.eye(4)
        T[:3, :3] = R.from_quat(state.q).as_matrix()
        T[:3, 3] = state.x
        m_ = self.robotMesh.copy()
        m_.transform(T, inplace=True)
        return m_

    def getPayloadMesh(self, state: RRTState) -> Optional[pv.PolyData]:
        if not self.considerPayload or self.payloadMesh is None:
            return None
        R = trf.Rotation.from_quat(state.q)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(state.q).as_matrix()
        T[:3, 3] = state.x + R.apply(self.payloadTranslation)
        m_ = self.payloadMesh.copy()
        m_.transform(T, inplace=True)
        return m_

    def visualizePath(self, env : CollisionEnvironment3D, payload ) -> pv.Plotter:
        pv_ = pv.Plotter()
        if self.environmentMesh is not None:
            pv_.add_mesh(self.environmentMesh, color="red", opacity=1)

        for state in self.states:
            print(f"State {state.i}: pos={state.x}, quat={state.q}")
            robot_mesh = self.getRobotMesh(state)
            if robot_mesh is not None:
                pv_.add_mesh(robot_mesh, color="blue", opacity=0.4)

            if self.considerPayload:
                payload_mesh = self.getPayloadMesh(state)
                print(f"num Collisions: {env.numCollisions(payload, state.x + trf.Rotation.from_quat(state.q).apply(self.payloadTranslation), state.q)}")
                if payload_mesh is not None:
                    pv_.add_mesh(payload_mesh, color="green", opacity=0.4)

        return pv_

class RRTPlanner3D:
    class Node:
        def __init__(self, state: RRTState, parent: Optional["RRTPlanner3D.Node"] = None):
            self.state = state
            self.parent = parent

    def __init__(
        self,
        env: CollisionEnvironment3D,
        robot: fcl.CollisionObject,
        robotMesh : pv.PolyData,
        payload: Optional[fcl.CollisionObject] = None,
        payloadMesh : Optional[pv.PolyData] = None,
        translation: np.ndarray = np.array([0, 0, 0]),
        originalPayloadAttitude: trf.Rotation = trf.Rotation.from_quat([0, 0, 0, 1]),
        step_size=0.3,
        goal_tolerance=0.3,
        goal_sample_rate=0.4,
        max_iters=30000,
        numTriesSampling=1000,
        posMin: List[float] = [0, -5, 0],
        posMax: List[float] = [10, 20, 2]
    ):
        self.env = env
        self.robot = robot
        self.robotMesh = robotMesh
        self.payload = payload
        self.payloadMesh = payloadMesh
        self.translation = translation
        self.originalPayloadAttitude = originalPayloadAttitude
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.numTriesSampling = numTriesSampling
        self.posMin = posMin
        self.posMax = posMax

        self.considerPayload = payload is not None
        self.path: List[RRTState] = []

    def _extend(self, tree: List["RRTPlanner3D.Node"], target: RRTState) -> Optional["RRTPlanner3D.Node"]:
        nearest = min(tree, key=lambda n: n.state.distance_to(target))
        dist = nearest.state.distance_to(target)
        alpha = min(self.step_size / dist, 1.0) if dist > 1e-6 else 1.0
        new_state = self._interpolate(nearest.state, target, alpha)

        if self._motion_valid(nearest.state, new_state, considerPayload=self.considerPayload):
            new_node = self.Node(new_state, nearest)
            tree.append(new_node)
            return new_node
        return None

    def _connect(self, tree: List["RRTPlanner3D.Node"], target: RRTState) -> Optional["RRTPlanner3D.Node"]:
        nearest = min(tree, key=lambda n: n.state.distance_to(target))
        dist = nearest.state.distance_to(target)
        alpha = min(self.step_size / dist, 1.0) if dist > 1e-6 else 1.0
        new_state = self._interpolate(nearest.state, target, alpha)

        if not self._motion_valid(nearest.state, new_state, considerPayload=self.considerPayload):
            return None

        node = self.Node(new_state, nearest)
        tree.append(node)

        # Keep connecting until goal is reached
        while new_state.distance_to(target) > self.goal_tolerance:
            nearest = node
            dist = nearest.state.distance_to(target)
            alpha = min(self.step_size / dist, 1.0)
            new_state = self._interpolate(nearest.state, target, alpha)
            if not self._motion_valid(nearest.state, new_state, considerPayload=self.considerPayload):
                return None
            node = self.Node(new_state, nearest)
            tree.append(node)
        return node
    def _interpolate(self, s1: RRTState, s2: RRTState, alpha: float) -> RRTState:
        '''
        Interpolate between two RRTState objects using linear interpolation for position and spherical linear interpolation (slerp) for quaternion.

        Parameters:
            s1 (RRTState): The initial state
            s2 (RRTState): The final state
            alpha (float): The interpolation factor, should be between 0 and 1
        
        Returns:
            RRTState: A new RRTState object that is the result of the interpolation

        '''
        x = (1 - alpha) * s1.x + alpha * s2.x
        slerp = Slerp([0, 1], R.from_quat([s1.q, s2.q]))
        q = slerp([alpha])[0].as_quat()
        return RRTState(x=x, q=q)

    def _is_valid(self,
                   state: RRTState,
                   env : Optional[CollisionEnvironment3D] = _unset,
                   robot : Optional[fcl.CollisionObject] = _unset,
                   payload : Optional[fcl.CollisionObject] = _unset,
                   translation: Optional[np.ndarray] = _unset, 
                   considerPayload : bool = False) -> bool:
        ''' 
        Check if a given state is valid, by computing the number of collisions between the robot and the environment.
        If a payload is provided then we also check for the collision between the payload and the environment.
    
        Parameters:
            state (RRTState): The state to check for validity
            env (Optional[CollisionEnvironment3D]): The environment in which the planning is being done, if not provided we use the one passed in the __init__ function.
            robot (Optional[fcl.CollisionObject]): The collision object representing the robot, if not provided we use the one passed in the __init__ function.
            payload (Optional[fcl.CollisionObject]): The collision object representing the payload, if not provided we use the one passed in the __init__ function.
            translation (Optional[np.ndarray]): The translation vector for the payload, if not provided we use the one passed in the __init__ function.
        considerPayload (bool): Whether to consider the payload in the collision check, if not provided we use the one passed in the __init__ function.
        
        Returns:
            bool: True if the state is valid (no collisions), False otherwise
        '''
        env = env if env is not _unset else self.env
        robot = robot if robot is not _unset else self.robot
        payload = payload if payload is not _unset else self.payload
        translation = translation if translation is not _unset else self.translation

        collisions = env.numCollisions(robot, state.x, state.q)
        R = trf.Rotation.from_quat(state.q)
        # payload and translation can only be None, if we dont receive a payload here and in the __init__ function, this means that no matter what the user want, we cannot compute the collisions between the payload and the environment
        if considerPayload and payload is not None and translation is not None:
            collisions += env.numCollisions(payload, state.x + R.apply(translation), state.q)
        return collisions == 0

    def _motion_valid(self,
                     from_state: RRTState, 
                     to_state: RRTState,
                        env: Optional[CollisionEnvironment3D] = _unset,
                        robot: Optional[fcl.CollisionObject] = _unset,
                        payload: Optional[fcl.CollisionObject] = _unset,
                        translation: Optional[np.ndarray] = _unset,
                        considerPayload: bool = False
                     ) -> bool:

        '''
        Check if the motion from one state to another is valid by interpolating between the two states and checking for collisions at each step.

        Parameters:
            from_state (RRTState): The starting state of the motion.
            to_state (RRTState): The ending state of the motion.
            env (Optional[CollisionEnvironment3D]): The environment in which the planning is being done, if not provided we use the one passed in the __init__ function.
            robot (Optional[fcl.CollisionObject]): The collision object representing the robot, if not provided we use the one passed in the __init__ function.
            payload (Optional[fcl.CollisionObject]): The collision object representing the payload, if not provided we use the one passed in the __init__ function.
            translation (Optional[np.ndarray]): The translation vector for the payload, if not provided we use the one passed in the __init__ function.
            considerPayload (bool): Whether to consider the payload in the collision check, if not provided we use the one passed in the __init__ function.

        Returns:
            bool: True if the motion is valid (no collisions at any interpolated step), False otherwise.
        '''
        env = env if env is not _unset else self.env
        robot = robot if robot is not _unset else self.robot
        payload = payload if payload is not _unset else self.payload
        translation = translation if translation is not _unset else self.translation
        considerPayload = considerPayload if considerPayload is not _unset else self.considerPayload


        dist = from_state.distance_to(to_state)
        if dist < 1e-6:
            return True
        steps = int(np.ceil(dist / (self.step_size / 2)))

        for i in range(steps + 1):
            alpha = i / steps
            if not self._is_valid(self._interpolate(from_state, to_state, alpha), env, robot, payload, translation, considerPayload):
                return False
        return True

    def _sample(self, 
                goal : RRTState, 
                env : Optional[CollisionEnvironment3D] = _unset,
                robot : Optional[fcl.CollisionObject] = _unset,
                payload: Optional[fcl.CollisionObject] = _unset,
                translation: Optional[np.ndarray] = _unset,
                posMin: Optional[List[float]] = _unset,
                posMax: Optional[List[float]] = _unset, 
                considerPayload: bool = False) -> RRTState:
        '''
        Sample a random state in the configuration space, biased towards the goal if specified.

        Parameters:
            goal (RRTState): The goal state to bias the sampling towards.
            env (Optional[CollisionEnvironment3D]): The environment in which the planning is being done, if not provided we use the one passed in the __init__ function.
            robot (Optional[fcl.CollisionObject]): The collision object representing the robot, if not provided we use the one passed in the __init__ function.
            payload (Optional[fcl.CollisionObject]): The collision object representing the payload, if not provided we use the one passed in the __init__ function.
            translation (Optional[np.ndarray]): The translation vector for the payload, if not provided we use the one passed in the __init__ function.
            posMin (Optional[List[float]]): Minimum position bounds for sampling, if not provided we use the one passed in the __init__ function.
            posMax (Optional[List[float]]): Maximum position bounds for sampling, if not provided we use the one passed in the __init__ function.
            considerPayload (bool): Whether to consider payload during sampling, if not provided we use the one passed in the __init__ function.
        
        Returns:
            RRTState: A randomly sampled state in the configuration space, biased towards the goal.
        '''
        if np.random.rand() < self.goal_sample_rate:
            return goal

        env = env if env is not _unset else self.env
        robot = robot if robot is not _unset else self.robot
        payload = payload if payload is not _unset else self.payload
        translation = translation if translation is not _unset else self.translation
        posMin = posMin if posMin is not _unset else self.posMin
        posMax = posMax if posMax is not _unset else self.posMax

        for _ in range(self.numTriesSampling):
            pos = np.random.uniform(posMin, posMax)
            quat = R.random().as_quat()
            state = RRTState(pos, quat)
            if self._is_valid(state, env, robot, payload, translation, considerPayload):
                print(f"Sampling Valid State: pos={state.x}, quat={state.q}")
                return state

        return goal

    def _reconstruct(self, node: Node) -> List[RRTState]:
        '''
        Reconstruct the path from the goal node back to the start node by traversing the parent pointers.

        Parameters:
            node (Node): The goal node from which to reconstruct the path.

        Returns:
            List[RRTState]: A list of RRTState objects representing the path from start to goal.
        '''
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def _createPath(self, path: List[RRTState]) -> RRTPath:
        '''
        Create a RRTPath object from a list of RRTState objects.
        This can be used in the plan method to return the correct object

        Parameters
            path (List[RRTState]): List of RRTState objects representing the path
        
        Returns
            RRTPath: A RRTPath object containing the path and other relevant information
        '''
        return RRTPath(
            states=path, 
            robotMesh = self.robotMesh, 
            payloadMesh = self.payloadMesh, 
            environmentMesh = self.env.getMesh() if self.env else None, 
            originalPayloadAttitude = self.originalPayloadAttitude,
            payloadTranslation=self.translation, 
            considerPayload=self.considerPayload
        )

    def plan(self,
            start: RRTState,
            goal: RRTState,
            env: CollisionEnvironment3D = _unset,
            robot: fcl.CollisionObject = _unset,
            payload: Optional[fcl.CollisionObject] = _unset,
            translation: np.ndarray = _unset,
            originalPayloadAttitude= _unset,
            posMin=_unset,
            posMax=_unset) -> RRTPath:

        env = env if env is not _unset else self.env
        robot = robot if robot is not _unset else self.robot
        payload = payload if payload is not _unset else self.payload
        translation = translation if translation is not _unset else self.translation
        originalPayloadAttitude = originalPayloadAttitude if originalPayloadAttitude is not _unset else self.originalPayloadAttitude
        posMin = posMin if posMin is not _unset else self.posMin
        posMax = posMax if posMax is not _unset else self.posMax

        self.tree_start = [self.Node(start)]
        self.tree_goal = [self.Node(goal)]

        for i in range(self.max_iters):
            print(f"Bi-RRT Iteration {i+1}/{self.max_iters}")
            rand = self._sample(goal, considerPayload=self.considerPayload)

            # Extend start tree
            node_start = self._extend(self.tree_start, rand)
            if node_start is None:
                continue

            # Try to connect goal tree to the new node
            node_goal = self._connect(self.tree_goal, node_start.state)
            if node_goal is not None:
                path_start = self._reconstruct(node_start)
                path_goal = self._reconstruct(node_goal)
                self.path = path_start + path_goal[::-1]
                break

            # Swap trees
            self.tree_start, self.tree_goal = self.tree_goal, self.tree_start

        return self._createPath(self.path)
        

    def get_path(self) -> List[RRTState]:
        return self.path

    def get_rrt_path(self, robot_mesh=None, payload_mesh=None, environment_mesh=None) -> RRTPath:
        return RRTPath(
            states=self.path,
            robotMesh=robot_mesh,
            payloadMesh=payload_mesh,
            environmentMesh=environment_mesh,
            originalPayloadAttitude=self.originalPayloadAttitude,
            payloadTranslation=self.translation,
            considerPayload=self.considerPayload,
        )
