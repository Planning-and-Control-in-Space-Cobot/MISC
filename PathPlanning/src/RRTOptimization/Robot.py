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

from Obstacle import Obstacle
from Environment import EnvironmentHandler


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
        print(
            f"iter {iter} Face normal: {R.apply(self.normal)}, Obstacle normal: {normal}, dot: {np.dot(normal, R.apply(self.normal))} = center offset {self.centerOffset}"
        )

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

    def getObstaclesSingle(
            self: _T,
            environment: EnvironmentHandler,
            cPos : np.ndarray,
            cR : trf.Rotation,
    ) -> List[Obstacle]:
        if not isinstance(environment, EnvironmentHandler):
            raise TypeError("Environment must be an instance of EnvironmentHandler")

        obstacles = []
        for v in self.getVertices():
            point = environment.buildSinglePoint()
            minDistance, pt1, pt2, _ = environment.distance(
                point, cR.as_matrix() @ v + cPos, cR
            )
            obstacles.append(
                Obstacle(pt2, (pt1 - pt2) / np.linalg.norm(pt1 - pt2), minDistance, -1, pt1)
            )

        filteredObstacles = []
        for obs in obstacles:
            newObstacle = True
            for fobs in filteredObstacles:
                if np.allclose(obs.normal, fobs.normal, 1e-1):
                    newObstacle = False
            if newObstacle:
                filteredObstacles.append(obs)
                
        return filteredObstacles

    def getObstacles(
        self,
        environment: EnvironmentHandler,
        path, 
        previousPath,
        previousObstacles, 
        prevMaxDistances
    ) -> Tuple[List[Obstacle], List[float], bool, List[Obstacle]]:
        """Function to compute the collision planes of the robot with the environment

        Parameters
            environment (EnvironmentHandler): Environment handler object that contains the environment information
            pos (np.ndarray): position of the robot in the environment, this is the translation from the center of the robot to the closest point in the robot
            R (trf.Rotation): quaternion of the robot in the environment, this is the rotation from the center of the robot to the closest point in the robot
            iteration (int) : iteration on the path that the obstacles returns should be considered
            count (int): number of points to sample on the surface of the robot mesh, default is 100

        Returns:
            Tuple[List[Obstacle], List[float], bool, List[Obstacle]]:
                - List of obstacles detected in the environment, if a collision was detected in the current path, it is the previous obstacles and the collisions planes
                - Maximum distance of the closest obstacles detected in the environment by each face 
                - Boolean indicating if any collision was detected
                - List of the collision planes if any exist

        """
        anyCollision = False
        obstacles, maxDistance = [], []
        for i, p in enumerate(path):
            collision, _, _, _, _ = environment.collide(
                self.fcl_obj, p.x, trf.Rotation.from_quat(p.q)
            )

            if collision:
                anyCollision = True
                break
            
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
                return None, None, None, None
            
            maxDistance.append(max(_minDistance))

        collisionObstacles = []
        if anyCollision:
            for i, p in enumerate(path):
                collision, depth, pt1, pt2, normal = environment.collide(
                    self.fcl_obj, p.x, trf.Rotation.from_quat(p.q)
                )

                if collision:
                    collisionObstacles.append(
                        Obstacle(pt2, -normal, depth, i, pt1)
                ) 
            previousObstacles.extend(collisionObstacles)
            return previousObstacles, prevMaxDistances, anyCollision,  collisionObstacles
        return obstacles, maxDistance, anyCollision, collisionObstacles

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

    @override
    def f(self, state, u, dt):
        '''
        Computes the next state of the robot given the current state.

        Taking into account the robot's dynamics, the current state, control 
        inputs, and the time step, this function computes the next state of the 
        robot. This function works with casadi variables and not numpy arrays, 
        since this is to be used inside an optimization problem and not for a
        simulation directly.

        Parameters:
            state (ca.MX): Current state of the robot, which includes position, 
                velocity, quaternion and angular velocity. 13x1 vector.
            u (ca.MX): Control inputs for the robot. 6x1 vector.
            dt (float): Time step for the state update.

        Returns:
            ca.MX: Next state of the robot, which includes position, velocity,
                quaternion and angular velocity. 13x1 vector.
        '''        
        def unflat(state, u):
            '''
            Unflattens the state and control inputs vectors

            Unflattens the state vector and the control input vector into the 
            respective components: position, velocity, quaternion, angular
            velocity, force, and moment.


            Parameters:
                state (ca.MX): Current state of the robot, which includes
                    position, velocity, quaternion and angular velocity. 13x1 vector.
                u (ca.MX): Control inputs for the robot. 6x1 vector.
            
            Returns:
                tuple: A tuple containing the position (x), velocity (v),
                quaternion (q), angular velocity (w), force (F), and moment (M)
                of the robot.
            '''
            x = state[0:3]
            v = state[3:6]
            q = state[6:10]
            w = state[10:13]
            return x, v, q, w, self.A[0:3, :] @ u, self.A[3:6, :] @ u

        def flat(x, v, q, w):
            '''
            Flattens the state variables into a single vector

            Flattens the position, velocity, quaternion, and angular velocity
            vectors into a single vector representing the state of the robot.

            Parameters:
                x (ca.MX): Position vector of the robot. 3x1 vector.
                v (ca.MX): Velocity vector of the robot. 3x1 vector.
                q (ca.MX): Quaternion representing the orientation of the robot. 4x1
                vector.
                w (ca.MX): Angular velocity vector of the robot. 3x1 vector
            
            Returns:
                ca.MX: A single vector containing the flattened state of the robot,
                which includes position, velocity, quaternion, and angular velocity.
            '''
            return ca.vertcat(x, v, q, w)

        def quat_mul(q1, q2):
            '''
            Multiplies two quaternions

            Multiplies two quaternions q1 and q2 using the quaternion 
            multiplication method described into 
            "Indirect Kalman Filter for 3D Attitude Estimation" 

            Parameters:
                q1 (ca.MX): First quaternion to be multiplied. 4x1 vector
                q2 (ca.MX): Second quaternion to be multiplied. 4x1 vector

            Returns:
                ca.MX: The resulting quaternion after multiplying q1 and q2.
            '''
            q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
            q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]
            q_ = ca.vertcat(
                ca.horzcat(q1w, q1z, -q1y, q1x),
                ca.horzcat(-q1z, q1w, q1x, q1y),
                ca.horzcat(q1y, -q1x, q1w, q1z),
                ca.horzcat(-q1x, -q1y, -q1z, q1w),
            )
            return q_ @ ca.vertcat(q2x, q2y, q2z, q2w)

        def quat_int(q, w, dt):
            '''
            Integration of a quaternion using angular velocity

            Integrates the quaternion q, considering the angular velocity and 
            time step dt using the method described in 
            "Indirect Kalman Filter for 3D Attitude Estimation"

            Parameters:
                q (ca.MX): Current quaternion representing the orientation of the robot. 4x
                vector.
                w (ca.MX): Angular velocity vector of the robot. 3x1 vector
                dt (float): Time step for the integration.
            
            Returns:
                ca.MX: The resulting quaternion after integrating the angular
                velocity over the time step dt.
            '''
            w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)
            q_ = ca.vertcat(
                w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2)
            )
            return quat_mul(q_, q)

        x, v, q, w, F, M = unflat(state, u)
        R = sc.Rotation.from_quat(q)
        x_next = x + v * dt
        v_next = v + dt * (1 / self.m) * R.as_matrix() @ F
        q_next = quat_int(q, w, dt)
        w_next = w + dt * ca.inv(self.J) @ (M - ca.cross(w, self.J @ w))
        return flat(x_next, v_next, q_next, w_next)

    def linearizedDynamics(self, x : ca.MX, u : ca.MX, dt : ca.MX):
        """Computes the linearized dynamics of the robot
        
        The dynamics of the system are non linear, and non convex, in order to
        work with a convex optimization problem, we need to linearize the 
        dynamics around a given point. This will then allow us to then represent 
        the system like this x(k+1)  = A(k) (xk - x0) + B(k) (uk - u0) + x0

        Parameters:
            x (ca.MX): Current state of the robot, which includes position, 
                velocity, quaternion and angular velocity. 13x1 vector.
            u (ca.MX): Control inputs for the robot. 6x1 vector.
            dt (ca.MX): Time step for the state update.
        """
        f = self.f(x, u, dt)
        A = ca.jacobian(f, x)
        B = ca.jacobian(f, u)
        return A, B, f
