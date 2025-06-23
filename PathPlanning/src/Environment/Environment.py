import open3d as o3d
import numpy as np
import scipy.spatial.transform as trf
import pyvista as pv
import time
import coal


class EnvironmentHandler:
    """Class to handle the environment, which is a voxel grid built from a point
        cloud.
    This class is used to convert a point cloud into a mesh that is compatible 
        with coal for flexible collision detection.
    """
    
    def __init__(self, pcd, voxel_size=0.1):
        """Receives a point cloud and a voxel size, and builds a voxel grid that
          is compatible with fcl for colision detection.

        Args:
            pcd (open3d.geometry.PointCloud): point cloud to be converted to 
                voxel grid
            voxel_size (float): size of the voxels in the grid

        Returns:
            None
        """
        self.voxel_size = voxel_size
        self.pcd = pcd
        pts = np.asarray(self.pcd.points)
        pts -= pts.min(axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(pts)

        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            self.pcd, voxel_size=self.voxel_size
        )
        (
            self.voxel_mesh,
            self.vertices,
            self.quads,
            self.triangleIndex,
            self.triangleVertex,
        ) = self._fast_voxel_mesh(self.voxel_grid)

        self._build_coal_mesh(self.triangleIndex, self.triangleVertex)

    def _fast_voxel_mesh(self, voxel_grid):
        """Builds a Voxel mesh from a voxel grid. This function is fully
            vectorized and optimized for speed.
        This function also considers the voxels to be all the same size, as does
             not join voxels that are adjacent to each other.

        Parameters:
            voxel_grid (o3d.geometry.VoxelGrid): The voxel grid to convert to a 
              mesh.
        
        Returns:
            visualization mesh (pyvista.PolyData): The mesh representation of 
                the voxel grid.
            points (np.ndarray): The centers of the voxels.
            quads (np.ndarray): The quad faces of the voxels.
            triangleIndices (np.ndarray): The indices of the triangles in the 
                mesh.
            triangleVertex (np.ndarray): The vertices of the triangles in the 
                mesh.
        """
        timeStart = time.time()

        voxel_size = voxel_grid.voxel_size
        origin = voxel_grid.origin
        voxels = voxel_grid.get_voxels()

        # Get centers
        centers = (
            np.array([v.grid_index for v in voxels]) * voxel_size
            + origin
            + voxel_size / 2
        )
        min_bounds = centers.min(axis=0)
        centers -= min_bounds

        # Cube corners: (8, 3)
        cube = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
            )
            - 0.5
        ) * voxel_size

        # Faces template: (6, 4)
        faces_template = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [1, 2, 6, 5],
                [0, 3, 7, 4],
            ]
        )

        # Vectorized vertices: repeat centers, tile cube
        N = centers.shape[0]
        points = np.repeat(centers, 8, axis=0) + np.tile(
            cube, (N, 1)
        )  # (N*8, 3)

        # Vectorized faces: replicate faces_template with correct offsets
        offsets = (np.arange(N) * 8).reshape(-1, 1, 1)  # (N, 1, 1)
        faces = faces_template[None, :, :] + offsets  # (N, 6, 4)
        quads = faces.reshape(-1, 4)  # (N*6, 4)

        # Convert quads to triangle list for PolyData: each quad → 2 triangles
        tris = np.empty(
            (len(quads) * 2, 4), dtype=np.int32
        )  # Each face: [3, i, j, k]

        tris[0::2, 0] = 3
        tris[0::2, 1:] = quads[:, [0, 1, 2]]

        tris[1::2, 0] = 3
        tris[1::2, 1:] = quads[:, [0, 2, 3]]

        tris_flat = tris.flatten()

        triangleIndices = np.empty((len(quads) * 2, 3), dtype=np.int64)
        triangleIndices[0::2] = quads[:, [0, 1, 2]]
        triangleIndices[1::2] = quads[:, [0, 2, 3]]

        # Return PyVista mesh + raw data
        print(f"Voxel grid was build in {time.time() - timeStart} seconds")
        return (
            pv.PolyData(points, tris_flat),
            points,
            quads,
            triangleIndices,
            points,
        )

    def _build_coal_mesh(
        self, triangleIndex: np.ndarray, triangleVertex: np.ndarray
    ):
        """Builds the coal mesh for the environment from the information related
             to the voxel grid

        Parameters:
            triangleIndex (np.ndarray): Nx3 Array where each row represents a 
                triangle, and each column represents a vertex index
            triangleVertex (np.ndarray): Nx3 Array where each row represents a 
                vertex in 3D space

        Returns:
            None
        """
        mesh = coal.BVHModelOBBRSS()
        mesh.beginModel(triangleIndex.shape[0], triangleVertex.shape[0])
        mesh.addTriangles(triangleIndex)
        mesh.addVertices(triangleVertex)
        mesh.endModel()
        self.envMesh = mesh

    def buildEllipsoid(
        self, ellipsoid_radii: np.ndarray = np.array([0.24, 0.24, 0.10])
    ):
        """Creates an ellipsoid to encompass the robot for the collision
            detection.

        Args:
            ellipsoid_radii (np.ndarray): radii of the ellipsoid in x, y, z 
                directions

        Returns:
            fcl.CollisionObject: fcl collision object representing the ellipsoid

        Raises:
            TypeError: if ellipsoid_radii is not a numpy array
            ValueError: if ellipsoid_radii is not a 1D array of length 3
        """
        if not isinstance(ellipsoid_radii, np.ndarray):
            raise TypeError("Ellipsoid radii must be a numpy array.")

        if ellipsoid_radii.shape != (3,):
            raise ValueError("Ellipsoid radii must be a 1D array of length 3.")

        return coal.Ellipsoid(ellipsoid_radii)

    def buildBox(self, box_size: np.ndarray = np.array([0.45, 0.45, 0.12])):
        """Creates a box to encompass an object for the collision detection.

        Parameters:
            box_size (np.ndarray): size of the box in x, y, z directions

        Returns:
            fcl.CollisionObject: fcl collision object representing the box
        """
        if not isinstance(box_size, np.ndarray):
            raise TypeError("Box size must be a numpy array.")

        if box_size.shape != (3,):
            raise ValueError("Box size must be a 1D array of length 3.")

        return coal.Box(box_size)

    def buildSinglePoint(self):
        """Creates a single point to encompass the robot for the collision
            detection.

        Args:
            None

        Returns:
            fcl.CollisionGeometry: fcl collision object representing the point

        Raises:
            None
        """
        return coal.Sphere(0.001)

    def collide(
        self,
        obj1: coal.CollisionObject,
        p1: np.ndarray = np.zeros((3,)),
        q1: trf.Rotation = trf.Rotation.from_euler("xyz", [0, 0, 0]),
    ):  # type: ignore
        """Collision between two object and returns the result information.

        The second coal object is the environment mesh that was build from the  
            point cloud passed to the Environment Handler

        Parameters:
            obj1 (coal.CollisionObject): first coal object
            p1 (np.ndarray): position of the first object
            q1 (np.ndarray): quaternion of the first object

        Returns:
            isCollision (bool): True if there is a collision, False otherwise
            depth (float): depth of the collision if there is a collision, 
                None otherwise
            nearestPoint1 (np.ndarray) : nearest point on the object if there 
                is a collision, None otherwise
            nearestPoint2 (np.ndarray) : nearest point on the environment mesh 
                if there is a collision, None otherwise
            normal (np.ndarray): normal of the collision if there is a 
                collision, None otherwise

        Raises:
            TypeError: if obj1 is not a coal.CollisionObject, p1 is not a numpy 
                array of shape (3, 1), or q1 is not a scipy Rotation object
            ValueError: if the two objects are in collision

        """
        if not isinstance(p1, np.ndarray) or p1.shape != (3,):
            raise TypeError("Position must be a numpy array of shape (3).")

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
            return (True, depth, nearestPoint1, nearestPoint2, normal)
        else:
            colRes.clear()
            return (False, None, None, None, None)

    def distance(
        self,
        obj1: coal.CollisionGeometry,
        p1: np.ndarray = np.zeros((3,)),
        q1: trf.Rotation = trf.Rotation.from_euler("xyz", [0, 0, 0]),
    ):
        """Compute the distance between two objects that are not in collision.

        The second coal object is the environment mesh that was built from the
        point cloud passed to the Environment Handler.

        Parameters:
            obj1 (coal.CollisionObject): First coal object.
            p1 (np.ndarray): Position of the first object.
            q1 (np.ndarray): Quaternion of the first object.

        Returns:
            minDistance (float): Minimum distance between the two objects.
            pt1 (np.ndarray): Nearest point on the first object.
            pt2 (np.ndarray): Nearest point on the second object.

        Raises:
            ValueError: If the two objects are in collision.
            TypeError: If obj1 is not a coal.CollisionObject, p1 is not a numpy
                array of shape (3, 1), or q1 is not a scipy Rotation object.
        """
        if not isinstance(p1, np.ndarray) or p1.shape != (3,):
            raise TypeError("Position must be a numpy array of shape (3,).")

        if not isinstance(q1, trf.Rotation):
            raise TypeError("Quaternion must be a scipy Rotation object.")

        isCollision, depth, _, _, _ = self.collide(obj1, p1, q1)

        #if isCollision:
        #    print(f"Depth {depth} between objects, cannot compute distance.")
        #    raise ValueError(
        #        "Objects are in collision, cannot compute distance."
        #    )

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
        """Visualizes the voxel grid and the point cloud.

        Args:
            plotter (pv.Plotter): PyVista plotter object to visualize the voxel
                grid

        Returns:
            None
        """
        plotter.add_mesh(self.voxel_mesh, color="white", show_edges=True)
        return plotter

    def getMesh(self):
        """Returns the voxel mesh of the environment.

        Args:
            None

        Returns:
            pv.PolyData: voxel mesh of the environment
        """
        return self.voxel_mesh
