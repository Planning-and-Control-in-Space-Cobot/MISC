import numpy as np

class OptimizationState:
    """Class representing a single state in the optimization problem

    This class encapsulates the state of a system at a given time step in the 
    optimization problem. This class will be used as an interface for both the 
    global optimization problem, as well as the local optimization problem.
    """
    def __init__(
        self,
        x: np.ndarray,
        q: np.ndarray,
        v: np.ndarray = np.zeros((3)),
        w: np.ndarray = np.zeros((3)),
        u: np.ndarray = np.zeros((6, 1)),
        i: np.ndarray = 0,
    ):
        """Initialized the optimization state

        Args:
            x (np.ndarray): position in 3D space - (3,)
            v (np.ndarray): velocity in 3D space - (3,)
            q (np.ndarray): quaternion representing orientation - (4,)
            w (np.ndarray): angular velocity in 3D space - (3,)
            u (np.ndarray): control inputs - (6, 1)
            i (np.ndarray): index of the state in the problem - (1,) 
        """
        self.x = x  # position
        self.v = v  # velocity
        self.q = q  # quaternion
        self.q = self.q / np.linalg.norm(self.q)
        self.w = w  # angular velocity
        self.u = u  # control inputs
        self.i = i  # index of the state in the optimization problem

    def get_state(self) -> np.ndarray:
        """Return the state in a flatten numpy array

        Return:
            np.ndarray: flatten state - (13,)
        """
        return np.hstack([self.x, self.v, self.q, self.w])
