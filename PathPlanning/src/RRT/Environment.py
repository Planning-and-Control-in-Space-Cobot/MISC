import fcl
import open3d as o3d
import numpy as np
import scipy.spatial.transform as trf
import pyvista as pv
import time


class EnvironmentHandler:
    def __init__(self, pcd, voxel_size=0.01):
        """
        Receives a point cloud and a voxel size, and builds a voxel grid that is
        compatible with fcl for colision detection.

        Args:
            pcd (open3d.geometry.PointCloud): point cloud to be converted to voxel grid
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
        self.voxel_mesh, self.vertices, self.triangle_faces = self._fast_voxel_mesh(
            self.voxel_grid
        )

        self.fcl_env = self._build_voxel_fcl()

    def _fast_voxel_mesh(self, voxel_grid):
        import time
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
        points = (
            np.repeat(centers, 8, axis=0) + np.tile(cube, (N, 1))
        )  # (N*8, 3)

        # Vectorized faces: replicate faces_template with correct offsets
        offsets = (np.arange(N) * 8).reshape(-1, 1, 1)  # (N, 1, 1)
        faces = faces_template[None, :, :] + offsets  # (N, 6, 4)
        quads = faces.reshape(-1, 4)  # (N*6, 4)

        # Convert quads to triangle list for PolyData: each quad → 2 triangles
        tris = np.empty((len(quads) * 2, 4), dtype=np.int32)  # Each face: [3, i, j, k]

        tris[0::2, 0] = 3
        tris[0::2, 1:] = quads[:, [0, 1, 2]]

        tris[1::2, 0] = 3
        tris[1::2, 1:] = quads[:, [0, 2, 3]]

        tris_flat = tris.flatten()

        # Return PyVista mesh + raw data
        print(f"Voxel grid was build in {time.time() - timeStart} seconds")
        return pv.PolyData(points, tris_flat), points, quads

    def _build_voxel_fcl(self):
        timeStart = time.time()
        model = fcl.BVHModel()
        model.beginModel(len(self.vertices), len(self.triangle_faces))
        model.addSubModel(
            self.vertices.astype(np.float32), self.triangle_faces.astype(np.int32)
        )
        model.endModel()
        print(f"Voxel grid was converted to fcl in {time.time() - timeStart} seconds")
        return fcl.CollisionObject(model)

    def buildEllipsoid(
        self, ellipsoid_radii: np.ndarray = np.array([0.24, 0.24, 0.10])
    ):
        """
        Creates an ellipsoid to encompass the robot for the collision detection.

        Args:
            ellipsoid_radii (np.ndarray): radii of the ellipsoid in x, y, z directions

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

        shape = fcl.Ellipsoid(
            ellipsoid_radii[0], ellipsoid_radii[1], ellipsoid_radii[2]
        )
        tf = fcl.Transform(np.eye(3), np.array([0, 0, 0]))

        return fcl.CollisionObject(shape, tf)

    def buildBox(self, box_size : np.ndarray = np.array([0.45, 0.45, 0.12])):
        '''
        Creates a box to encompass an object for the collision detection.

        Parameters:
            box_size (np.ndarray): size of the box in x, y, z directions
            
        Returns:
            fcl.CollisionObject: fcl collision object representing the box
        '''
        if not isinstance(box_size, np.ndarray):
            raise TypeError("Box size must be a numpy array.")
        
        if box_size.shape != (3,):
            raise ValueError("Box size must be a 1D array of length 3.")
        
        shape = fcl.Box(box_size[0], box_size[1], box_size[2])
        tf = fcl.Transform(np.eye(3), np.array([0, 0, 0]))
        return fcl.CollisionObject(shape, tf)

    def buildSinglePoint(self):
        """
        Creates a single point to encompass the robot for the collision detection.

        Args:
            None

        Returns:
            fcl.CollisionGeometry: fcl collision object representing the point

        Raises:
            None
        """
        shape = fcl.Sphere(0.001)  # Small sphere to represent a point
        tf = fcl.Transform(np.eye(3), np.array([0, 0, 0]))

        return fcl.CollisionObject(shape, tf)

    def numCollisions(
        self,
        obj1: fcl.CollisionGeometry,
        p1: np.ndarray = np.zeros((3, 1)),
        q1: np.ndarray = np.array([0, 0, 0, 1]),
        obj2: fcl.CollisionGeometry = None,
        p2: np.ndarray = np.zeros((3, 1)),
        q2: np.ndarray = np.array([0, 0, 0, 1]),
    ):
        """
        Computes the number of collions between two fcl objects.

        By default we receive only one fcl object and its tranform and compute to the generated environemnt

        But a second fcl object and respective transformation can be passed to the function
        """
        if obj2 is None:
            obj2 = self.fcl_env

        rot1 = trf.Rotation.from_quat(q1).as_matrix()
        tf1 = fcl.Transform(rot1, p1)
        obj1.setTransform(tf1)

        rot2 = trf.Rotation.from_quat(q2).as_matrix()
        tf2 = fcl.Transform(rot2, p2)
        obj2.setTransform(tf2)

        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()

        ret = fcl.collide(obj1, obj2, req, res)

        return ret

    def getCollisionPoints(
        self,
        obj1 : fcl.CollisionGeometry,
        p1: np.ndarray = np.zeros((3, 1)),
        q1: np.ndarray = np.array([0, 0, 0, 1]),
        obj2: fcl.CollisionGeometry = None,
        p2: np.ndarray = np.zeros((3, 1)),
        q2: np.ndarray = np.array([0, 0, 0, 1]),
    ):
        """
        Returns the collision points between two fcl objects.
        By default we receive only one fcl object and its tranform and compute to the generated environemnt
        Parameters:
            obj1 (fcl.CollisionObject): first fcl object
            p1 (np.ndarray): position of the first object
            q1 (np.ndarray): quaternion of the first object
            obj2 (fcl.CollisionObject): second fcl object
            p2 (np.ndarray): position of the second object
            q2 (np.ndarray): quaternion of the second object
        Returns:
            collision_points (list): list of collision points
        """ 
        if obj2 is None:
            obj2 = self.fcl_env

        #print(f"p1 {p1} q1 {q1} p2 {p2} q2 {q2}")

        rot1 = trf.Rotation.from_quat(q1).as_matrix()
        tf1 = fcl.Transform(rot1, p1)
        obj1.setTransform(tf1)

        rot2 = trf.Rotation.from_quat(q2).as_matrix()
        tf2 = fcl.Transform(rot2, p2)
        obj2.setTransform(tf2)

        req = fcl.CollisionRequest(enable_contact=True)
        res = fcl.CollisionResult()

        ret = fcl.collide(obj1, obj2, req, res)

        return res.contacts[0].pos if ret else None

    def getClosestPoints(
        self,
        obj1: fcl.CollisionGeometry,
        p1: np.ndarray = np.zeros((3, 1)),
        q1: np.ndarray = np.array([0, 0, 0, 1]),
        obj2: fcl.CollisionGeometry = None,
        p2: np.ndarray = np.zeros((3, 1)),
        q2: np.ndarray = np.array([0, 0, 0, 1]),
    ):
        """
        Returns the closest point between two fcl objects in both objects as well as the distance between the points.

        By default we receive only one fcl object and its transform and compute to the generated environment

        Parameters:
            obj1 (fcl.CollisionObject): first fcl object
            p1 (np.ndarray): position of the first object
            q1 (np.ndarray): quaternion of the first object
            obj2 (fcl.CollisionObject): second fcl object
            p2 (np.ndarray): position of the second object
            q2 (np.ndarray): quaternion of the second object

        Returns:
            distance (float): Minimum distance between the two objects
            p_obj1 (np.ndarray): Closest point on the first object
            p_obj2 (np.ndarray): Closest point on the second object
        """

        if obj2 is None:
            obj2 = self.fcl_env
        rot1 = trf.Rotation.from_quat(q1).as_matrix()
        tf1 = fcl.Transform(rot1, p1)
        obj1.setTransform(tf1)


        rot2 = trf.Rotation.from_quat(q2).as_matrix()
        tf2 = fcl.Transform(rot2, p2)
        obj2.setTransform(tf2)

        req = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
        res = fcl.DistanceResult()

        ret = fcl.distance(obj1, obj2, req, res)
        if res.o1 == obj1:
            pt1 = res.nearest_points[0]
            pt2 = res.nearest_points[1]
        else:
            pt1 = res.nearest_points[1]
            pt2 = res.nearest_points[0]
        

        return (
            res.min_distance,
            pt1, 
            pt2
        )

    def getNormalPlane(
        self,
        obj1: fcl.CollisionGeometry,
        p1: np.ndarray = np.zeros((3, 1)),
        q1: np.ndarray = np.array([0, 0, 0, 1]),
        obj2: fcl.CollisionGeometry = None,
        p2: np.ndarray = np.zeros((3, 1)),
        q2: np.ndarray = np.array([0, 0, 0, 1]),
    ):
        """
        Returns the normal of the plane of collision between two fcl objects.
        By default we receive only one fcl object and its tranform and compute to the generated environemnt

        Parameters:
            obj1 (fcl.CollisionObject): first fcl object
            p1 (np.ndarray): position of the first object
            q1 (np.ndarray): quaternion of the first object
            obj2 (fcl.CollisionObject): second fcl object
            p2 (np.ndarray): position of the second object
            q2 (np.ndarray): quaternion of the second object

        Returns:
            normal (np.ndarray): normal of the plane of collision
        """

        if obj2 is None:
            obj2 = self.fcl_env

        if self.numCollisions(obj1, p1, q1, obj2, p2, q2) > 0:
            return np.zeros(3)

        _, p_ellipsoid, p_plane = self.getClosestPoints(obj1, p1, q1, obj2, p2, q2)

        return -(p_plane - p_ellipsoid) / np.linalg.norm(p_plane - p_ellipsoid)

    def visualizeMap(self, plotter : pv.Plotter):
        """
        Visualizes the voxel grid and the point cloud.

        Args:
            None

        Returns:
            None
        """
        plotter.add_mesh(self.voxel_mesh, color="white", show_edges=True)
        return plotter
    
    def getMesh(self):
        """
        Returns the voxel mesh of the environment.

        Args:
            None

        Returns:
            pv.PolyData: voxel mesh of the environment
        """
        return self.voxel_mesh
    
    def _debugPointCloud(self):
        '''
        Visualized the point cloud as the voxel grid built from it to ensure the discretization is correct and not too coarse.

        Args: 
            None

        Returns:
            None
        '''
        # Mesh handling

        pv_ = pv.Plotter()
        #pv_.add_mesh(pv.PolyData(np.asarray(self.pcd.points)), color="blue", point_size=4, render_points_as_spheres=True)
        pv_.add_mesh(self.voxel_mesh, color="white", show_edges=True, opacity=0.5)
        print(f"Voxel grid has {len(self.voxel_grid.get_voxels())} voxels")
        pv_.show_grid()
        pv_.add_legend(
            [
                ("Point Cloud", "blue"),
                ("Voxel Grid", "white"),
            ],
            loc="upper left",
        )
        pv_.show()
