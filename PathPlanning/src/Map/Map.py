import os
import numpy as np 
import open3d as o3d
 
class Map:
    startState = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) 
    endState   = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    boundsMin  = np.array([0.0, 0.0, 0.0])
    boundsMax  = np.array([0.0, 0.0, 0.0])
    pointCloud = o3d.geometry.PointCloud()
     
    def __init__(self, 
                 pointCloudPath : str, 
                 startState : np.ndarray = None, 
                 endState   : np.ndarray = None, 
                 boundsMin  : np.ndarray = None,
                 boundsMax  : np.ndarray = None):
        """ Initializer for the Map class.

        Parameters:
            pointCloudPath - str:
                Full path string for the place where the point cloud is stored
            startState - np.ndarray (13,):
                Initial State for the robot in order: (pos, vel, quat, omega)
            endState - np.ndarray (13,):
                Final State for the robot in order: (pos, vel, quat, omega)
            boundsMin - np.ndarray (3,):
                Minimum bounds for the environment
            boundsMax - np.ndarray (3,):
                Maximum bounds for the environment
        """
        if startState is not None:
            if np.shape(startState) != (13,):
                raise ValueError("startState must be a 13-element array")
            startState[6:10] /= np.linalg.norm(startState[6:10])  # <- your style
            self.startState = startState

        if endState is not None:
            if np.shape(endState) != (13,):
                raise ValueError("endState must be a 13-element array")
            endState[6:10] /= np.linalg.norm(endState[6:10])      # <- your style
            self.endState = endState

        if boundsMin is not None:
            if np.shape(boundsMin) != (3,):
                raise ValueError("boundsMin must be a 3-element array")
            self.boundsMin = boundsMin

        if boundsMax is not None:
            if np.shape(boundsMax) != (3,):
                raise ValueError("boundsMax must be a 3-element array")
            self.boundsMax = boundsMax

        if not os.path.exists(pointCloudPath):
            raise FileNotFoundError(f"Point cloud file {pointCloudPath} does not exist.")
        self._pointCloudPath = os.path.abspath(pointCloudPath)
        self.pointCloud = o3d.io.read_point_cloud(self._pointCloudPath)        

    # -------------------- Getters --------------------
    def getStartState(self) -> np.ndarray:
        """Returns the start state of the robot"""
        return self.startState

    def getEndState(self) -> np.ndarray:
        """Returns the end state of the robot"""
        return self.endState   

    def getPointCloud(self) -> o3d.geometry.PointCloud:
        """Returns the point cloud of the environment"""
        return self.pointCloud

    def getBoundsMin(self) -> np.ndarray:
        """Returns the minimum bounds (3,)"""
        return self.boundsMin

    def getBoundsMax(self) -> np.ndarray:
        """Returns the maximum bounds (3,)"""
        return self.boundsMax

    def getPointCloudPath(self) -> str:
        """Returns the absolute path to the point cloud file"""
        return self._pointCloudPath

    # -------------------- Convenience --------------------
    def to_dict(self) -> dict:
        """Lightweight metadata representation (no Open3D objects)"""
        return {
            "point_cloud_path": self._pointCloudPath,
            "start_state": self.startState,
            "end_state": self.endState,
            "bounds_min": self.boundsMin,
            "bounds_max": self.boundsMax,
        }

    def __repr__(self) -> str:
        return (
            f"Map(pointCloudPath='{self._pointCloudPath}', "
            f"startState={self.startState}, "
            f"endState={self.endState}, "
            f"boundsMin={self.boundsMin}, boundsMax={self.boundsMax}, "
            f"numPoints={len(self.pointCloud.points)})"
        )
