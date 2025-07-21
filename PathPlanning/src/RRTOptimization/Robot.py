import casadi as ca
import spatial_casadi as sc

import trimesh as tm
import numpy as np

import coal

import pyvista as pv

import scipy.spatial.transform as trf
from abc import abstractmethod

from typing import TypeVar, List, Optional, Tuple 

_T = TypeVar("_T", bound="Model")
from typing_extensions import override

from RRTOptimization.Obstacle import Obstacle
from Environment import EnvironmentHandler
from RRTOptimization.OptimizationState import OptimizationState


class Model:
    def __init__(
        self,
        name: str,
        fcl_obj: coal.CollisionGeometry = None,
        mesh: pv.PolyData = None,
    ):
        """Base class for a model in the optimization problem, this class will be used to represent the robot dynamics, shape and collision geometry

        The init function of the model may be overloaded if necessary
        Parameters
            name (str): Name of the model
            collisionGeometry (fcl.CollisionGeometry): Collision geometry of the model, this will be used to compute the collision constraints in the optimization problem
            mesh (o2d.geometry.TriangleMesh): Mesh of the model, this will be used to visualize the model in the optimization problem
        """
        self.fcl_obj = fcl_obj
        self.mesh = mesh

    def getCollisionGeometry(self):
        """Function to get the collision geometry of the model,"""
        return self.fcl_obj

    def getMesh(self):
        """Function to get the mesh of the model, this will be used to visualize the model in the optimization problem"""
        return self.mesh

    @abstractmethod
    def getPVMesh(self):
        """Function to get the mesh of the model in a format that can be used by pyvista for visualization

        Returns:
            pv.PolyData: Mesh of the model in a format that can be used by pyvista for visualization
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    @abstractmethod
    def f(self, state, u, dt):
        """Function to compute the next state of the Model given the current state, control inputs and time step

        Parameters
            state (np.ndarray): current state of the model
            u (np.ndarray): control inputs
            dt (float): time step
        Returns
            np.ndarray: next state of the model
        """
        raise NotImplementedError("This method should be implemented in the subclass")


class face():
    """ Class to represent a single face of the robot.

    To more accurately represent the robot, and be able to get the obstacles 
    closer to the robot in all scenarios, we will represent the square robot as 
    a collection of 6 faces, each face, will then be 1 rectangular face 
    (only for distance checking, not for collision checking). This will allow
    more flexibility. 
    
    Observation, since fcl / coal only support 3D shapes, we will represent the
    face as a thin rectangular prism, allowing us to compare it to a normal face
    """
    def __init__(self : _T, 
                 normal : np.ndarray,
                 sideLength : np.ndarray,
                 centerOffset : np.ndarray
                 ):
        """Constructor for the face class
        Parameters
            normal (np.ndarray): normal vector of the face
            sideLenght (np.ndarray): side lenght of the face, should be a 3D 
                vector, with one of the dimensions being 0, since this is a 
                face and not a volume.
            centerOffset (np.ndarray): offse of the center of the face from the 
                center of the robot

        Returns:
            None
        """
        if np.shape(normal) != (3,) or np.shape(sideLength) != (3,):
            raise ValueError(
                "normal must be a 3D vector and sideLength must be a 3D vector"
            )

        if np.count_nonzero(sideLength) != 2:
            raise ValueError(
                "sideLength must be a 3D vector with two non-zero elements," \
                " since this is a face and not a volume"
            )

        self.normal = normal
        self.sideLength = sideLength
        self.sideLength[np.where(self.sideLength == 0)] = 1e-3  # Avoid division by zero    
        self.centerOffset = centerOffset

        self.faceObj = coal.Box(sideLength)

    def getFCLObject(self : _T) -> coal.CollisionGeometry:
        """Function to get the fcl object of the face, this will be used to compute the collision constraints in the optimization problem

        Returns:
            coal.CollisionGeometry: fcl object of the face
        """
        return self.faceObj 

    def getClosestObstacle(self : _T,
                            environment : EnvironmentHandler,
                            pos : np.ndarray, 
                            R : trf.Rotation,
                            iter : int) -> Optional[Obstacle]:
        """Function to get the closest obstacle to the face in the environment
        Parameters
            environment (EnvironmentHandler): Environment handler object that contains the environment information
            pos (np.ndarray): position of the face in the environment, this is the translation from
                the center of the face to the closest point in the face
            R (trf.Rotation): quaternion of the face in the environment, this is the
                rotation from the center of the face to the closest point in the face
            iter (int): iteration of the path that the obstacle return should be considered
        Returns:
            Optional[Obstacle]: Obstacle object representing the closest obstacle to the face in the environment,
                or None if the obstacle closest to this face is not alligned with the face normal, since this means this normal is not facing any relecant obstacle
                """
        if not isinstance(environment, EnvironmentHandler):
            raise TypeError("Environment must be an instance of EnvironmentHandler")

        minDistance, pt1, pt2, normal = environment.distance(
            self.faceObj, pos + R.apply(self.centerOffset), R
        )

        # Strange BUG in coal, distance return Nan for all values of normal
        normal = (pt1 - pt2) / np.linalg.norm(pt1 - pt2)
        if np.dot(normal, R.apply(self.normal)) > -0.2:
            # If the normal of the face is not facing the obstacle, we ignore it
            return None
        """
        pv_ = pv.Plotter()
        pv_.add_mesh(environment.voxel_mesh, color="lightgray", opacity=0.1)

        square = pv.Box(bounds=(
            -self.sideLength[0] / 2, self.sideLength[0] / 2,
            -self.sideLength[1] / 2, self.sideLength[1] / 2,
            -self.sideLength[2] / 2, self.sideLength[2] / 2
        ))

        transform = np.eye(4)
        transform[:3, :3] = R.as_matrix()
        transform[:3, 3] = pos + self.centerOffset
        square.transform(transform)

        pv_.add_mesh(square, color="red", opacity=0.5)

        pt1Mesh = pv.Sphere(radius=0.01, center=pt1)
        pt2Mesh = pv.Sphere(radius=0.01, center=pt2)
        pv_.add_mesh(pt1Mesh, color="blue", opacity=0.5)
        pv_.add_mesh(pt2Mesh, color="green", opacity=0.5)

        plane = pv.Plane(
            center=pos + self.centerOffset,
            direction=self.normal,
            i_size=self.sideLength[0],
            j_size=self.sideLength[1],
        )

        arrow   = pv.Arrow( 
            start=pt2, 
            direction=normal, 
            scale=0.1, 
            tip_length=0.1
        )

        pv_.add_mesh(plane, color="orange", opacity=0.5)
        pv_.add_mesh(arrow, color="purple", opacity=0.5)


        pv_.show()"""

        return Obstacle(pt2, normal, minDistance, iter, pt1)


class Robot(Model):
    """This class derives from the Model class and will represent the Space Cobot Robot in the optimization process.

    The Space Cobot Robot is a 6 DoF robot with an ellipsoid shaped body, nevertheless, for simplicity in the optimization problem, we will represent the robot with a rectangular body, since this is a convex hull an allows for easy half plane obstacle avoidance constraints, such as the ones we want to use xn >= a;

    The robot can be contained in a 0.45m x 0.45m x 0.12m rectangular box, as the one used.

    """

    @override
    def __init__(
        self,
        J: np.ndarray,
        A: np.ndarray,
        m: float,
        x : float = 0.45,
        y : float = 0.45,
        z : float = 0.12
    ):
        """Robot class to represent the robot in the optimization problem

        Parameters
            J (np.ndarray): inertia matrix of the robot
            A (np.ndarray): actuation matrix of the robot
            m (float): mass of the robot
            fcl_obj (fcl.CollisionGeometry) : collision geometry of the robot, this might be used to compute the collision constraints of the robot.
            mesh (pv.PolyData): mesh

        Returns:
            None
        """
        if np.shape(J) != (3, 3) or np.shape(A) != (6, 6) or m <= 0:
            raise ValueError(
                "J must be a 3x3 matrix, A must be a 6x6 matrix, ellipsoid_radius must be a 3x1 vector and m must be a positive scalar"
            )

        self.J = J
        self.A = A
        self.m = m

        self.fcl_obj = coal.Box(np.array([x, y, z]))
        self.mesh = pv.Box(bounds=(-x/2, x/2, -y/2, y/2, -z/2, z/2))

        self.x = x
        self.y = y
        self.z = z
        self.faces = self._createFaces()

        #self.drawRobotAndFaces()

    def _createFaces(self : _T) -> List[face]:
        """
        Create the 6 face objects for a cuboid robot centered at the origin.

        Returns:
            List[face]: list containing 6 face objects (top, bottom, front, back, left, right)
        """
        half_l = self.x / 2
        half_w = self.y / 2
        half_h = self.z / 2

        length = self.x
        width = self.y
        height = self.z

        return [
            # Top face (+Z)
            face(
                normal=np.array([0, 0, 1]),
                sideLength=np.array([length, width, 0]),
                centerOffset=np.array([0, 0, half_h]),
            ),
            # Bottom face (-Z)
            face(
                normal=np.array([0, 0, -1]),
                sideLength=np.array([length, width, 0]),
                centerOffset=np.array([0, 0, -half_h]),
            ),
            # Front face (+Y)
            face(
                normal=np.array([0, 1, 0]),
                sideLength=np.array([length, 0, height]),
                centerOffset=np.array([0, half_w, 0]),
            ),
            # Back face (-Y)
            face(
                normal=np.array([0, -1, 0]),
                sideLength=np.array([length, 0, height]),
                centerOffset=np.array([0, -half_w, 0]),
            ),
            # Right face (+X)
            face(
                normal=np.array([1, 0, 0]),
                sideLength=np.array([0, width, height]),
                centerOffset=np.array([half_l, 0, 0]),
            ),
            # Left face (-X)
            face(
                normal=np.array([-1, 0, 0]),
                sideLength=np.array([0, width, height]),
                centerOffset=np.array([-half_l, 0, 0]),
            ),
        ]

    def collisionFree(self,
                      path : List[OptimizationState],
                      environment: EnvironmentHandler,
                      ) -> bool:
        """Function to check if the robot is in a collision free state in the environment
        Parameters
            x (np.ndarray): position of the robot in the environment, this is the translation from the center of the robot to the closest point in the robot
            R (trf.Rotation): quaternion of the robot in the environment, this is the rotation from the center of the robot to the closest point in the robot
            environment (EnvironmentHandler): Environment handler object that contains the environment information
        
        Returns:
            bool: True if the robot is in a collision free state, False otherwise
        """
        if not isinstance(environment, EnvironmentHandler):
            raise TypeError("Environment must be an instance of EnvironmentHandler")

        for p in path:
            collision, _, _, _, _ = environment.collide(
                self.fcl_obj, p.x, trf.Rotation.from_quat(p.q)
            )
            if collision:
                return False
        
        return True

    def getObstacles(
        self,
        environment: EnvironmentHandler,
        path, 
    ) -> Tuple[List[Obstacle], List[float]]:
        """Function to compute the obstacles for a collision free path

        Parameters
            environment (EnvironmentHandler): Environment handler object 
            path (List[OptimizationState]) : List of states representing the 
                trajectory


        Returns:
            Tuple[List[Obstacle], List[float]]:
                - List of obstacles detected in the environment, if a collision 
                    was detected in the current path, it is the previous 
                    obstacles and the collisions planes
                - Maximum distance of the closest obstacles detected in the 
                environment by each face 
        """
        anyCollision = False
        obstacles, maxDistance = [], []
        for i, p in enumerate(path):
            collision, _, _, _, _ = environment.collide(
                self.fcl_obj, p.x, trf.Rotation.from_quat(p.q)
            )
            
            _minDistance = []
            _obstacles = []
            for f in self.faces:
                obs = f.getClosestObstacle(environment, p.x, trf.Rotation.from_quat(p.q), i)
                if obs is not None:
                    _minDistance.append(obs.minDistance)
                    newObstacle = True
                    for _obs in _obstacles:
                        if np.allclose(_obs.normal, obs.normal, 1e-1):
                            newObstacle = False 
                    if newObstacle:
                        _obstacles.append(obs)
                        obstacles.append(obs)
            if _minDistance == []:
                md, pt1, pt2, _ = environment.distance(
                    self.fcl_obj, p.x, trf.Rotation.from_quat(p.q)
                )
                _minDistance.append(md)
                obstacles.append(
                    Obstacle(pt2, (pt1 - pt2) / np.linalg.norm(pt1 - pt2), md, i, pt1)
                )
            
            maxDistance.append(max(_minDistance))

        return obstacles, maxDistance

    def getCollision(self,
                     environment: EnvironmentHandler,
                     path : List[OptimizationState]) -> List[Obstacle]:
        """Function to compute the collision of the robot with the environment.
        
        Parameters
            environment (EnvironmentHandler): Environment handler object that 
                contains the environment information
            path (List[OptimizationState]): List of states representing the 
                trajectory
        Returns:
            List [Obstacle] : List of all the detected collisions in the 
                environment.
        """
        if not isinstance(environment, EnvironmentHandler):
            raise TypeError("Environment must be an instance of EnvironmentHandler")
        
        collisions = []
        for p in path:
            x = p.x
            R = trf.Rotation.from_quat(p.q)
            collision, depth, pt1, pt2, normal = environment.collide(
                self.fcl_obj, x, R
            )
            if collision:
                collisions.append(
                    Obstacle(pt2,-normal, depth, p.i, pt1
                    )
                )
        return collisions



    def drawRobotAndFaces(self : _T):
        """Function to draw the robot and its faces in a pyvista plotter, this is used for debugging purposes"""
        pv_ = pv.Plotter()
        pv_.add_mesh(self.mesh, color="blue", opacity=0.5)

        for f in self.faces:
            square = pv.Box(
                bounds=(
                    -f.sideLength[0] / 2, f.sideLength[0] /
                    2,
                    -f.sideLength[1] / 2, f.sideLength[1] /
                    2,
                    -f.sideLength[2] / 2, f.sideLength[2] /
                    2,
                )
            )
            transform = np.eye(4)
            transform[:3, 3] = f.centerOffset
            square.transform(transform)

            arrow = pv.Arrow(
                start=f.centerOffset,
                direction=f.normal,
                scale=0.1,
                tip_length=0.1
            )
            pv_.add_mesh(arrow, color="purple", opacity=0.5)
            pv_.add_mesh(square, color="red", opacity=0.5)

        pv_.show()

    @override
    def getPVMesh(self):
        """Function to get the mesh of the robot in a format that can be used by pyvista for visualization

        Returns:
            pv.PolyData: Mesh of the robot in a format that can be used by pyvista for visualization
        """
        return self.mesh

    def getPVMesh(self, pos: np.ndarray, R: trf.Rotation) -> pv.PolyData:
        """Function to get the mesh of the robot in a format that can be used by pyvista for visualization, given the position and rotation of the robot

        Parameters
            pos (np.ndarray): position of the robot in the environment, this is the translation from the center of the robot to the closest point in the robot
            R (trf.Rotation): quaternion of the robot in the environment, this is the rotation from the center of the robot to the closest point in the robot

        Returns:
            pv.PolyData: Mesh of the robot in a format that can be used by pyvista for visualization
        """
        T = np.eye(4)
        T [:3, :3] = R.as_matrix()
        T[:3, 3] = pos
        m_ = self.mesh.copy()
        m_.transform(T)
        return m_

    def getVertices(self) -> np.ndarray:
        """Returns the vertices of the box containing the robot, when the robot is represented as a rectangular box, that is centered with the origin and axis aligned with the axes.

        Parameters
            None

        Returns:
            np.ndarray: 3x8 array with the vertices of the box containing the robot, each column is a vertex in 3D space
        """
        return np.array(
            [
                [-0.225, -0.225, -0.06],
                [0.225, -0.225, -0.06],
                [0.225, 0.225, -0.06],
                [-0.225, 0.225, -0.06],
                [-0.225, -0.225, 0.06],
                [0.225, -0.225, 0.06],
                [0.225, 0.225, 0.06],
                [-0.225, 0.225, 0.06],
            ]
        )

    def f(self, state: ca.MX, u: ca.MX, dt: ca.MX) -> ca.MX:
        """
        Computes the next state of the robot using RK4 integration.

        Parameters:
            state (ca.MX): Current state [13x1] (position, velocity, quaternion, angular velocity)
            u (ca.MX): Control input [6x1]
            dt (ca.MX): Time step

        Returns:
            ca.MX: Next state [13x1] after applying RK4 integration
        """
        def f_dot(x, u):
            # Unpack state
            p = x[0:3]       # position
            v = x[3:6]       # velocity
            q = x[6:10]      # quaternion (x, y, z, w)
            w = x[10:13]     # angular velocity

            # Compute force and moment
            F = self.A[0:3, :] @ u  # force in body frame
            M = self.A[3:6, :] @ u  # moment in body frame

            # Convert quaternion to rotation matrix
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]
            R = ca.vertcat(
                ca.horzcat(1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)),
                ca.horzcat(    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)),
                ca.horzcat(    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)),
            )

            # Quaternion kinematic matrix
            Q = ca.vertcat(
                ca.horzcat( qw, -qz,  qy),
                ca.horzcat( qz,  qw, -qx),
                ca.horzcat(-qy,  qx,  qw),
                ca.horzcat(-qx, -qy, -qz)
            )

            # Compute derivatives
            p_dot = v
            v_dot = (1 / self.m) * R.T @ F
            q_dot = 0.5 * Q @ w
            w_dot = np.linalg.inv(self.J) @ (M - ca.cross(w, self.J @ w))

            return ca.vertcat(p_dot, v_dot, q_dot, w_dot)

        # RK4 integration
        k1 = f_dot(state, u)
        k2 = f_dot(state + 0.5 * dt * k1, u)
        k3 = f_dot(state + 0.5 * dt * k2, u)
        k4 = f_dot(state + dt * k3, u)

        next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state
    
    def numericalF (self, state, u, dt):
        """Numerical approximation of the robot dynamics using finite differences.

        This function is used to compute the next state of the robot given the
        current state, control inputs, and time step using a numerical
        approximation method.

        Parameters:
            state (np.ndarray): Current state of the robot.
            u (np.ndarray): Control inputs for the robot.
            dt (float): Time step for the state update.

        Returns:
            np.ndarray: Next state of the robot.
        """
        def unflat(state, u):
            x = state[0:3]
            v = state[3:6]
            q = state[6:10]
            w = state[10:13]
            return x, v, q, w, self.A[0:3, :] @ u, self.A[3:6, :] @ u
    
        def flat(x, v, q, w):
            return np.concatenate((x, v, q, w))

        def quat_mul(q1, q2):
            q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
            q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]
            return np.array([
                q1w * q2x + q1z * q2y - q1y * q2z + q1x * q2w,
                -q1z * q2x + q1w * q2y + q1x * q2z + q1y * q2w,
                q1y * q2x - q1x * q2y + q1w * q2z + q1z * q2w,
                -q1x * q2x -q1y *q2y -q1z*q2z +q1w*q2w
            ])
        
        def quat_int(q, w, dt):
            w_norm = np.linalg.norm(w) + 1e-3
            q_ = np.concatenate((w / w_norm * np.sin(w_norm * dt / 2), 
                                 np.array([np.cos(w_norm * dt / 2)])))
            return quat_mul(q_, q)

        x, v, q, w, F, M = unflat(state, u)
        R = trf.Rotation.from_quat(q)
        x_next = x + v * dt
        v_next = v + dt * (1 / self.m) * R.as_matrix().T @ F
        q_next = quat_int(q, w, dt)
        w_next = w + dt * np.linalg.inv(self.J) @ (M - np.cross(w, self.J @ w))
        return flat(x_next, v_next, q_next, w_next)