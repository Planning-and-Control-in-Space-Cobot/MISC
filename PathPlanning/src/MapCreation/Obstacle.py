from typing import Tuple

import numpy as np 
import open3d as o3d
import trimesh as tm
from scipy.spatial.transform import Rotation as R

from MapCreation.ObstacleMotion import Motion, NoMotion, CircleMotion, LinearMotion, SineMotion
from MapCreation.ObstacleMotion import AttitudeMotion, NoAttitudeMotion, CircleAttitudeMotion, LinearAttitudeMotion, SineAttitudeMotion

class Obstacle: 
    """Representing a obstacle.
    
    The obstacle can be static or dynamic, that will be determined by the 
    obstacle motion and attitude motion 
    """
    def __init__(self, 
                 motion: Motion,
                 attitude: AttitudeMotion,
                 pcd : o3d.geometry.PointCloud, 
                 mesh :  tm.Trimesh):
        """
        Initialize the Obstacle instance.

        Parameters:
            motion (ObstacleMotion): The motion of the obstacle.
            attitude (AttitudeMotion): The attitude motion of the obstacle.
            pcd (o3d.PointCloud): The point cloud representing the obstacle.
            mesh (tm.Trimesh): The mesh representing the obstacle.
        
        Raises:
            AssertionError: If motion is not an instance of ObstacleMotion,
                            attitude is not an instance of AttitudeMotion,
                            pcd is not an instance of o3d.PointCloud,
                            or mesh is not an instance of tm.Trimesh.
        """
        assert isinstance(motion, Motion)
        assert isinstance(attitude, AttitudeMotion)
        assert isinstance(pcd, o3d.geometry.PointCloud)
        assert isinstance(mesh, tm.Trimesh)
        self.motion = motion
        self.attitude = attitude
        self.pcd = pcd
        self.mesh = mesh
        
    def __call__(self, t: float) -> Tuple[np.ndarray, R]:
        """Calculate position at time t.

        Args:
            t (float): Time in seconds.
        Returns:
            np.ndarray: Position in 3D space at time t.
        """
        return self.motion(t), self.attitude(t)
        
    def __repr__(self) -> str:
        """String representation of the DynamicObstacle instance.
        
        Returns:
            str: A string describing the DynamicObstacle instance.
        
        Example:
            DynamicObstacle(motion=CircleMotion(radius=1.0, speed=1.0, center=[0.0, 0.0, 0.0]), pcd=<PointCloud>)
        """
        return f"DynamicObstacle(motion={self.motion}, pcd={self.pcd})"
    
    def getPcd(self, t : float = 0) -> o3d.geometry.PointCloud:
        """Get the point cloud of the dynamic obstacle.
    
        Parameters:
            t (float): Time in seconds. Default is 0.

        Returns:
            o3d.PointCloud: The point cloud representing the obstacle.
        """
        if t < 0:
            raise ValueError("Time t must be non-negative.")
        
        position, attitude = self(t)
        transform = np.eye(4)
        transform[:3, :3] = attitude.as_matrix()
        transform[:3, 3] = position
        pcd_transformed = self.pcd.transform(transform)
        return pcd_transformed
    
    def getMesh(self, t : float = 0) -> tm.Trimesh:
        """Get the mesh of the dynamic obstacle.

    
        Parameters:
            t (float): Time in seconds. Default is 0.

        Returns:
            tm.Trimesh: The mesh representing the obstacle.
        """
        if t < 0:
            raise ValueError("Time t must be non-negative.")
        
        position, attitude = self(t)
        transform = np.eye(4)
        transform[:3, :3] = attitude.as_matrix()
        transform[:3, 3] = position
        mesh_transformed = self.mesh.copy()
        mesh_transformed.apply_transform(transform)
        return mesh_transformed

    def to_dict(self):
        return {
            "motion": self.motion.__dict__ | {"__type__": type(self.motion).__name__},
            "attitude": self.attitude.__dict__ | {"__type__": type(self.attitude).__name__},
            "pcd_points": np.asarray(self.pcd.points).tolist(),
            "mesh_vertices": self.mesh.vertices.tolist(),
            "mesh_faces": self.mesh.faces.tolist()
        }

    @staticmethod
    def from_dict(data):
        # Reconstruct motion
        motion_type = data["motion"].pop("__type__")
        motion_cls = {
            "NoMotion": NoMotion,
            "CircleMotion": CircleMotion,
            "LinearMotion": LinearMotion,
            "SineMotion": SineMotion
        }[motion_type]
        motion = motion_cls(**data["motion"])

        # Reconstruct attitude
        attitude_type = data["attitude"].pop("__type__")
        attitude_cls = {
            "NoAttitudeMotion": NoAttitudeMotion,
            "CircleAttitudeMotion": CircleAttitudeMotion,
            "LinearAttitudeMotion": LinearAttitudeMotion,
            "SineAttitudeMotion": SineAttitudeMotion
        }[attitude_type]
        attitude = attitude_cls(**data["attitude"])

        # Reconstruct PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(data["pcd_points"]))

        # Reconstruct Trimesh
        mesh = tm.Trimesh(vertices=np.array(data["mesh_vertices"]),
                        faces=np.array(data["mesh_faces"]),
                        process=False)

        return Obstacle(motion=motion, attitude=attitude, pcd=pcd, mesh=mesh)