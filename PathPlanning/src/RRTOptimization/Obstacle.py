import numpy as np
import fcl

class Obstacle:
    def __init__(
        self,
        closestPointObstacle : np.ndarray, 
        normal : np.ndarray, 
        distance : float,
        iteration : int,
        closestPointRobot : np.ndarray ,
        safetyMargin : float = 0.01):
        '''
            This class represents an obstacle in the optimization problem

            Parameters
                closestPointRobot (np.ndarray): closest point on the robot
                closestPointObstacle (np.ndarray): closest point on the obstacle
                translation (np.ndarray): translation from the center of the robot to the closest point in the robot
                iteration (int) :  the iteration in the optimization problem that this obstacle plane should be considered
                safetyMargin (float): safety margin for the obstacle
        '''

        self.closestPointObstacle = closestPointObstacle
        self.normal = normal / np.linalg.norm(normal)
        self.minDistance = distance
        self.iteration = iteration
        self.safetyMargin = safetyMargin
        self.closestPointRobot = closestPointRobot
    
    def generateSquare(self):
        """
        Generate 4 vertices of a square in 3D space lying in a plane.

        Parameters:
        - p0 (array-like): 3D point on the plane (shape: (3,))
        - n (array-like): 3D normal vector of the plane (shape: (3,))
        - d (float): Half side length of the square (i.e., square is 2d x 2d)

        Returns:
        - V (np.ndarray): 3x4 matrix with columns as the 4 square vertices in 3D
        """
        # Pick a helper vector that is not parallel to n
        helper = np.array([0, 1, 0]) if not np.allclose(self.normal, [0, 1, 0]) else np.array([1, 0, 0])

        # Generate two orthonormal vectors in the plane
        u = np.cross(self.normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u)
        v /= np.linalg.norm(v)

        minDistance_ = self.minDistance
        self.minDistance = max(self.minDistance, 2)

        # Compute square vertices
        V = np.array([
            self.closestPointObstacle + self.minDistance*u + self.minDistance*v,
            self.closestPointObstacle - self.minDistance*u + self.minDistance*v,
            self.closestPointObstacle - self.minDistance*u - self.minDistance*v,
            self.closestPointObstacle + self.minDistance*u - self.minDistance*v
        ]).T  # shape (3, 4)
        self.minDistance = minDistance_

        return V

    def generateCube(self, ):
        """
        Generate 8 vertices of a cube in 3D space that lies in the half-space
        away from the obstacle (in the direction opposite to the normal).

        Parameters:
        - size (float): edge length of the cube

        Returns:
        - V (np.ndarray): 3x8 array where each column is a 3D vertex
        """
        size = max(self.minDistance + self.safetyMargin, 5)
        # Find center of cube, offset in the direction *away* from the obstacle
        offset = -self.normal * (size / 2.0)
        center = self.closestPointObstacle + offset

        # Create orthonormal basis (u, v, n)
        helper = np.array([0, 1, 0]) if not np.allclose(self.normal, [0, 1, 0]) else np.array([1, 0, 0])
        u = np.cross(self.normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u)
        v /= np.linalg.norm(v)
        n = self.normal

        # Half size in each direction
        hs = size / 2.0

        # Build 8 corners of the cube in the local frame and transform
        directions = [
            +u + v + n,  -u + v + n,  -u - v + n,  +u - v + n,
            +u + v - n,  -u + v - n,  -u - v - n,  +u - v - n,
        ]

        V = np.stack([center + hs * d for d in directions], axis=1)  # 3x8

        return V