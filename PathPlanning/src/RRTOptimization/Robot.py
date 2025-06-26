import casadi as ca
import spatial_casadi as sc

import trimesh as tm
import numpy as np

import coal

import pyvista as pv

import scipy.spatial.transform as trf
from abc import abstractmethod

from typing import TypeVar, List

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
        fcl_obj: coal.CollisionGeometry,
        mesh: tm.Trimesh,
    ):
        """Robot class to represent the robot in the optimization problem

        Parameters
            J (np.ndarray): inertia matrix of the robot
            A (np.ndarray): actuation matrix of the robot
            m (float): mass of the robot
            fcl_obj (fcl.CollisionGeometry) : collision geometry of the robot, this might be used to compute the collision constraints of the robot.
            mesh (tm.Trimesh): mesh

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
        self.fcl_obj = fcl_obj
        self.mesh = mesh

    def getObstacles(
        self,
        environment: EnvironmentHandler,
        path, 
        previousPath,
        previousObstacles, 
        prevMaxDistances
    ) -> List[Obstacle]:
        """Function to compute the collision planes of the robot with the environment

        Parameters
            environment (EnvironmentHandler): Environment handler object that contains the environment information
            pos (np.ndarray): position of the robot in the environment, this is the translation from the center of the robot to the closest point in the robot
            R (trf.Rotation): quaternion of the robot in the environment, this is the rotation from the center of the robot to the closest point in the robot
            iteration (int) : iteration on the path that the obstacles returns should be considered
            count (int): number of points to sample on the surface of the robot mesh, default is 100

        Returns:
            List[Obstacle]: list of Obstacle objects representing the collision planes of the robot with the environment

        """
        if not isinstance(environment, EnvironmentHandler):
            raise TypeError("Environment must be an instance of EnvironmentHandler")
        Obstacles = []
        maxDistances = []

        anyCollision = False    
        for i, p in enumerate(path):
            pos = p.x
            R = trf.Rotation.from_quat(p.q)
            collision, depth, nearestPointRobot, nearestPointObstacle, normal = environment.collide(self.fcl_obj, pos, R)

            if collision:
                anyCollision = True
                break
            obstacles = []
            maxDistance = -np.inf
            for v in self.getVertices():
                point = environment.buildSinglePoint()
                minDistance, pt1, pt2, _ = environment.distance(point, R.as_matrix() @ v + pos, R)
                obstacles.append(
                    Obstacle(pt2, (pt1 - pt2) / np.linalg.norm(pt1 - pt2), minDistance, i)
                )
                maxDistance = max(maxDistance, minDistance)

            filteredObstacles = []
            for obs in obstacles: 
                newObstacle = True
                for fobs in filteredObstacles:
                    if np.allclose(obs.normal, fobs.normal, 1e-1):
                        newObstacle = False
                if newObstacle:
                    filteredObstacles.append(obs)
            Obstacles.extend(filteredObstacles)
            maxDistances.append(maxDistance)
        
        if anyCollision:
            print("Collision detected in the path, computing collision planes")
            collisionObstacles = []
            for i, p in enumerate(path):
                pos = p.x
                R = trf.Rotation.from_quat(p.q)
                collision, depth, nearestPointRobot, nearestPointObstacle, normal = environment.collide(self.fcl_obj, pos, R)
                if collision:
                    t = trf.Rotation.from_quat(p.q).as_matrix().T @ (nearestPointRobot - pos)

                    point = environment.buildSinglePoint()
                    prevPath = previousPath[i]
                    prevP = prevPath.x

                    minDistance, pt1, pt2, _ = environment.distance(
                        point, prevP + trf.Rotation.from_quat(prevPath.q).as_matrix().T @ t
                    )


                    normal = (pt1 - nearestPointObstacle) / np.linalg.norm(pt1 - nearestPointObstacle)

                    print(f"Collision at {i} in {nearestPointObstacle} {normal} ")
                    previousObstacles.append(Obstacle(nearestPointObstacle, normal, minDistance, i))
                    collisionObstacles.append(
                        Obstacle(nearestPointObstacle, normal, minDistance, i)
                    )
            return previousObstacles, prevMaxDistances, True, collisionObstacles
        
        return Obstacles, maxDistances, False, []

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

        # Update the state using the Euler method
        x, v, q, w, F, M = unflat(state, u)
        R = sc.Rotation.from_quat(q)
        x_next = x + v * dt
        v_next = v + dt * (1 / self.m) * R.as_matrix() @ F
        q_next = quat_int(q, w, dt)
        q_next = q_next / ca.norm_2(q_next)  # Normalize quaternion
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