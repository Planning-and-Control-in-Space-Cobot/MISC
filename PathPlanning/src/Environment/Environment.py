import open3d as o3d
import numpy as np
import scipy.spatial.transform as trf
import pyvista as pv
import time
import coal
from typing import Optional, Any


class EnvironmentHandler:
    """
    Build a voxel mesh and COAL BVH from either:
      - an Open3D PointCloud (pcd=...)
      - a Map instance with getPointCloud() (map_obj=...)
    """

    def __init__(self,
                 pcd: Optional[o3d.geometry.PointCloud] = None,
                 voxel_size: float = 0.1,
                 map_obj: Optional[Any] = None):
        """
        Args:
            pcd: Open3D point cloud.
            voxel_size: size of voxels.
            map_obj: your Map instance (must provide getPointCloud(); optional
                     getBoundsMin/getBoundsMax/getStartState/getEndState).
        """
        self.voxel_size = voxel_size
        self.map = map_obj  # keep a reference if provided

        # --- Resolve input point cloud ---
        if map_obj is not None:
            if not hasattr(map_obj, "getPointCloud"):
                raise TypeError("map_obj must provide getPointCloud().")
            self.pcd = map_obj.getPointCloud()
            if not isinstance(self.pcd, o3d.geometry.PointCloud):
                raise TypeError("Map.getPointCloud() must return an Open3D PointCloud.")
            # Optional metadata from Map
            self.boundsMin = getattr(map_obj, "getBoundsMin", lambda: None)()
            self.boundsMax = getattr(map_obj, "getBoundsMax", lambda: None)()
            self.startState = getattr(map_obj, "getStartState", lambda: None)()
            self.endState   = getattr(map_obj, "getEndState",   lambda: None)()
        else:
            if pcd is None:
                raise ValueError("Provide either map_obj or pcd.")
            self.pcd = pcd
            self.boundsMin = None
            self.boundsMax = None
            self.startState = None
            self.endState = None

        # Ensure contiguous Nx3 float array in the PointCloud
        pts = np.asarray(self.pcd.points)
        self.pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=float))

        # --- Build VoxelGrid and fast voxel mesh ---
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=self.voxel_size)
        (self.voxel_mesh,
         self.vertices,
         self.quads,
         self.triangleIndex,
         self.triangleVertex,
         self.timeTaken) = self._fast_voxel_mesh(self.voxel_grid)

        # --- Build COAL BVH mesh ---
        self._build_coal_mesh(self.triangleIndex, self.triangleVertex)

    @classmethod
    def from_map(cls, map_obj: Any, voxel_size: float = 0.1) -> "EnvironmentHandler":
        """Convenience constructor when you already have a Map instance."""
        return cls(pcd=None, voxel_size=voxel_size, map_obj=map_obj)

    def _fast_voxel_mesh(self, voxel_grid):
        """Vectorized voxel-mesh construction from an Open3D VoxelGrid."""
        timeStart = time.time()

        voxel_size = voxel_grid.voxel_size
        origin = voxel_grid.origin
        voxels = voxel_grid.get_voxels()

        centers = (np.array([v.grid_index for v in voxels]) * voxel_size
                   + origin + voxel_size / 2.0)

        cube = (np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]) - 0.5) * voxel_size

        faces_template = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ])

        N = centers.shape[0]
        points = np.repeat(centers, 8, axis=0) + np.tile(cube, (N, 1))  # (N*8, 3)

        offsets = (np.arange(N) * 8).reshape(-1, 1, 1)
        faces = faces_template[None, :, :] + offsets                   # (N, 6, 4)
        quads = faces.reshape(-1, 4).astype(np.int64)                  # (N*6, 4)

        tris = np.empty((len(quads) * 2, 4), dtype=np.int32)
        tris[0::2, 0] = 3; tris[0::2, 1:] = quads[:, [0, 1, 2]]
        tris[1::2, 0] = 3; tris[1::2, 1:] = quads[:, [0, 2, 3]]
        tris_flat = tris.flatten()

        triangleIndices = np.empty((len(quads) * 2, 3), dtype=np.int64)
        triangleIndices[0::2] = quads[:, [0, 1, 2]]
        triangleIndices[1::2] = quads[:, [0, 2, 3]]

        timeEnd = time.time()
        return (
            pv.PolyData(points, tris_flat),
            points,
            quads,
            triangleIndices,
            points,                   # vertices array used as triangleVertex
            timeEnd - timeStart,
        )

    def _build_coal_mesh(self, triangleIndex: np.ndarray, triangleVertex: np.ndarray):
        """Create COAL BVH from triangle data."""
        mesh = coal.BVHModelOBBRSS()
        mesh.beginModel(triangleIndex.shape[0], triangleVertex.shape[0])
        mesh.addTriangles(triangleIndex)
        mesh.addVertices(triangleVertex)
        mesh.endModel()
        self.envMesh = mesh

    def buildEllipsoid(self, ellipsoid_radii: np.ndarray = np.array([0.24, 0.24, 0.10])):
        if not isinstance(ellipsoid_radii, np.ndarray) or ellipsoid_radii.shape != (3,):
            raise ValueError("ellipsoid_radii must be a numpy array with shape (3,).")
        return coal.Ellipsoid(ellipsoid_radii)

    def buildBox(self, box_size: np.ndarray = np.array([0.45, 0.45, 0.12])):
        if not isinstance(box_size, np.ndarray) or box_size.shape != (3,):
            raise ValueError("box_size must be a numpy array with shape (3,).")
        return coal.Box(box_size)

    def buildSinglePoint(self):
        return coal.Sphere(0.001)

    def collide(self,
                obj1: coal.CollisionObject,
                p1: np.ndarray = np.zeros((3,)),
                q1: trf.Rotation = trf.Rotation.from_euler("xyz", [0, 0, 0])):
        if not isinstance(p1, np.ndarray) or p1.shape != (3,):
            raise TypeError("Position must be a numpy array of shape (3,).")
        if not isinstance(q1, trf.Rotation):
            raise TypeError("Quaternion must be a scipy Rotation object.")

        T1 = coal.Transform3s()
        T1.setTranslation(p1)
        T1.setRotation(q1.as_matrix())

        T2 = coal.Transform3s()
        T2.setTranslation(np.zeros((3, 1)))
        T2.setRotation(np.eye(3))

        colReq = coal.CollisionRequest()
        colRes = coal.CollisionResult()

        coal.collide(obj1, T1, self.envMesh, T2, colReq, colRes)

        if colRes.isCollision():
            contact = colRes.getContact(0)
            depth = contact.penetration_depth
            nearestPoint1 = contact.getNearestPoint1()
            nearestPoint2 = contact.getNearestPoint2()
            normal = contact.normal
            colRes.clear()
            return True, depth, nearestPoint1, nearestPoint2, normal
        else:
            colRes.clear()
            return False, None, None, None, None

    def distance(self,
                 obj1: coal.CollisionGeometry,
                 p1: np.ndarray = np.zeros((3,)),
                 q1: trf.Rotation = trf.Rotation.from_euler("xyz", [0, 0, 0])):
        if not isinstance(p1, np.ndarray) or p1.shape != (3,):
            raise TypeError("Position must be a numpy array of shape (3,).")
        if not isinstance(q1, trf.Rotation):
            raise TypeError("Quaternion must be a scipy Rotation object.")

        T1 = coal.Transform3s()
        T1.setTranslation(p1)
        T1.setRotation(q1.as_matrix())

        T2 = coal.Transform3s()
        T2.setTranslation(np.zeros((3, 1)))
        T2.setRotation(np.eye(3))

        distReq = coal.DistanceRequest()
        distRes = coal.DistanceResult()

        coal.distance(obj1, T1, self.envMesh, T2, distReq, distRes)
        minDistance = distRes.min_distance
        pt1 = distRes.getNearestPoint1()
        pt2 = distRes.getNearestPoint2()
        normal = distRes.normal
        distRes.clear()
        return minDistance, pt1, pt2, normal

    def visualizeMap(self, plotter: pv.Plotter):
        plotter.add_mesh(self.voxel_mesh, color="white", show_edges=True)
        return plotter

    def getMesh(self):
        return self.voxel_mesh
