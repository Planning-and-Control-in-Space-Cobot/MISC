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
        safetyMargin : float = 0.001):
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
        self.closestPointRobot = closestPointRobot
        self.safetyMargin = safetyMargin