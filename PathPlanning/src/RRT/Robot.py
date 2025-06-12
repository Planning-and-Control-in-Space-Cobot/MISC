import casadi as ca 
import spatial_casadi as sc

import trimesh as tm 
import open3d as o3d
import numpy as np
import fcl

import scipy.spatial.transform as trf
from abc import ABC, abstractmethod

from typing import Any, Dict, Optional, Union, TypeVar, List
_T = TypeVar('_T', bound='Model')
from typing_extensions import override

from Obstacle import Obstacle
from Environment import EnvironmentHandler


class Model: 
    def __init__(self, name :  str, CollisionGeometry : fcl.CollisionGeometry = None, mesh : tm.Trimesh = None):
        '''
        Base class for a model in the optimization problem, this class will be used to represent the robot dynamics, shape and collision geometry

        The init function of the model may be overloaded if necessary
        Parameters
            name (str): Name of the model
            collisionGeometry (fcl.CollisionGeometry): Collision geometry of the model, this will be used to compute the collision constraints in the optimization problem
            mesh (o2d.geometry.TriangleMesh): Mesh of the model, this will be used to visualize the model in the optimization problem
        '''
        self.collisionGeometry = None


    def getCollisionGeometry(self):
        '''
        Function to get the collision geometry of the model, '''
    

    @abstractmethod    
    def f(self, state, u, dt):
        '''
        Function to compute the next state of the Model given the current state, control inputs and time step


        Parameters
            state (np.ndarray): current state of the model
            u (np.ndarray): control inputs
            dt (float): time step
        Returns
            np.ndarray: next state of the model
        '''
        raise NotImplementedError("This method should be implemented in the subclass")    

class Robot(Model):
    @override
    def __init__(
        self, J: np.ndarray, A: np.ndarray, m: float, fcl_obj : fcl.CollisionGeometry, mesh : tm.Trimesh
    ):
        """
        Robot class to represent the robot in the optimization problem

        Parameters
            J (np.ndarray): inertia matrix of the robot
            A (np.ndarray): actuation matrix of the robot
            m (float): mass of the robot
            fcl_obj (fcl.CollisionGeometry) : collision geometry of the robot, this might be used to compute the collision constraints of the robot.
            mesh (tm.Trimesh): mesh 

        Returns
            None
        """
        if (
            np.shape(J) != (3, 3)
            or np.shape(A) != (6, 6)
            or m <= 0
        ):
            raise ValueError(
                "J must be a 3x3 matrix, A must be a 6x6 matrix, ellipsoid_radius must be a 3x1 vector and m must be a positive scalar"
            )

        self.J = J
        self.A = A
        self.m = m
        self.fcl_obj = fcl_obj
        self.mesh = mesh
        
    def uniformSurfaceSample(self, count : int = 100) -> np.ndarray:
        '''
        Function to sample points uniformly on the surface of the robot mesh

        Parameters
            count (int): number of points to sample on the surface of the robot mesh, default is 100

        Returns
            np.ndarray: 3xN array of points sampled uniformly on the surface of the robot mesh. Robot is assumed to be centered at the origin, so the point is also the translation from the center of the robot to the point.
        '''
        if self.mesh is None:
            raise ValueError("Mesh is not defined for the robot")
        
        if not isinstance(self.mesh, tm.Trimesh):
            raise TypeError("Mesh must be a trimesh object")
        
        if count <= 0:
            raise ValueError("Count must be a positive integer")
            
        return tm.sample.sample_surface_even(self.mesh, count)[0].T
    

    def getObstacles(self, environment : EnvironmentHandler, pos : np.ndarray, R : trf.Rotation, iteration : int, useSampling=True, count=10) -> List[Obstacle]:
        '''
        Function to compute the collision planes of the robot with the environment

        If useSampling is True, we will sample multiple points on the surface of the robot mesh, and returns a list of the planes that are generated from these points.
        Take into consideration that we will remove planes that are equivalent across multiple points, meaning that the number of planes returns may be smaller than the number of sampled points.

        If useSampling is set to False, we will only compute the closest obstacle to the robot, this may not be representative for close collisions.

        Parameters
            environment (EnvironmentHandler): Environment handler object that contains the environment information
            pos (np.ndarray): position of the robot in the environment, this is the translation from the center of the robot to the closest point in the robot
            R (trf.Rotation): quaternion of the robot in the environment, this is the rotation from the center of the robot to the closest point in the robot
            iteration (int) : iteration on the path that the obstacles returns should be considered
            useSampling (bool): whether to use sampling or not, default is True
            count (int): number of points to sample on the surface of the robot mesh, default is 100
        
        Returns
            List[Obstacle]: list of Obstacle objects representing the collision planes of the robot with the environment

        '''
        if not isinstance(environment, EnvironmentHandler):
            raise TypeError("Environment must be an instance of EnvironmentHandler")
    
        if useSampling:
            points = self.uniformSurfaceSample(count)
            obj = environment.buildSinglePoint() # Collision object representation of the point
            obstacles = []
            for point in points.T:
                # ! STRANGE BEHAVIOR:
                # ! FOR A SPHERE THE pRobot IS ASSUMING THE ROBOT IS CENTERED AT THE ORIGIN, BUT THE MATH FROM FCL DOES NOT CONSIDER THE ROBOT AT THE ORIGIN, BUT RATHER AT THE CORRECT POSITION, SOME DIFFERENCE IN FRAMES IS HAPPENING, EVEN THO IT IS THE SAME METHOD!!!!

                distance, pRobot, pEnv = environment.getClosestPoints(obj, pos + R.apply(point), trf.Rotation.from_euler('xyz', [0, 0, 0]).as_quat())
                normal = pos + R.apply(point) + pRobot - pEnv
                normal /= np.linalg.norm(normal) # Normalize the normal vector
                print(f"Point: {point}, pRobot: {pRobot}, pEnv: {pEnv}, Normal: {normal}, Distance: {distance}")

                obstacles.append(Obstacle(pEnv, normal, distance, iteration))

            filteredObstacles = []
            np.set_printoptions(precision=3, suppress=True)
            for obs in obstacles:
                print(f"Norm: {obs.normal}, Distance: {obs.minDistance}")
                if not any(np.allclose(obs.normal, filteredObs.normal, atol=1e-1) for filteredObs in filteredObstacles):
                    filteredObstacles.append(obs) 
            print(f"Number of obstacles: {len(filteredObstacles)} vs len(obstacles): {len(obstacles)}")
            for obs in filteredObstacles:
                print(f"Obstacle: {obs.closestPointObstacle}, Normal: {obs.normal}, Distance: {obs.minDistance}, Iteration: {obs.iteration}")
            return filteredObstacles
        else:
            # Only compute the closest obstacles to the robot
            distance, pRobot, pEnv = environment.getClosestPoints(self.fcl_obj, pos, R.as_quat())
            normal = pRobot - pEnv 
            normal /= np.linalg.norm(normal)
            return [Obstacle(pEnv, normal, distance, iteration)] 

    @override
    def f(self, state, u, dt):
        def unflat(state, u):
            x = state[0:3]
            v = state[3:6]
            q = state[6:10]
            w = state[10:13]
            return x, v, q, w, self.A[0:3, :] @ u, self.A[3:6, :] @ u

        def flat(x, v, q, w):

            return ca.vertcat(x, v, q, w)

        def quat_mul(q1, q2):
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