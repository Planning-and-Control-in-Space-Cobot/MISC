import open3d as o3d
import pyvista as pv
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R, Slerp

class State:
    def __init__(self, x, v, q, w):
        self.x = x
        self.v = v
        self.q = q
        self.w = w

    def distance_to(self, other):
        pos_dist = np.linalg.norm(self.x - other.x)
        quat_dist = 1 - np.abs(np.dot(self.q, other.q))
        return pos_dist + quat_dist

class CollisionEnvironment3D:
    def __init__(self, pcd, voxel_size=0.1, ellipsoid_radii=(0.24, 0.24, 0.1)):
        self.voxel_size = voxel_size
        self.ellipsoid_radii = ellipsoid_radii

        self.pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(self.pcd.points)
        pts -= pts.min(axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(pts)

        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=self.voxel_size)
        self.voxel_mesh, self.vertices, self.triangle_faces = self._fast_voxel_mesh(self.voxel_grid)

        self.fcl_env = self._build_voxel_fcl()
        self.ellipsoid_fcl = self._build_ellipsoid_fcl()

    def _fast_voxel_mesh(self, voxel_grid):
        voxel_size = voxel_grid.voxel_size
        origin = voxel_grid.origin
        voxels = voxel_grid.get_voxels()
        centers = np.array([v.grid_index for v in voxels]) * voxel_size + origin + voxel_size / 2
        min_bounds = centers.min(axis=0)
        centers -= min_bounds

        cube = (np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]) - 0.5) * voxel_size

        faces_template = np.array([
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]
        ])

        all_pts, all_faces, offset = [], [], 0
        for c in centers:
            pts = cube + c
            all_pts.append(pts)
            all_faces.append(faces_template + offset)
            offset += 8

        points = np.vstack(all_pts)
        quads = np.vstack(all_faces)
        tris = []
        for quad in quads:
            tris.append([3, quad[0], quad[1], quad[2]])
            tris.append([3, quad[0], quad[2], quad[3]])
        tris = np.array(tris, dtype=np.int32).flatten()
        return pv.PolyData(points, tris), points, quads

    def _build_voxel_fcl(self):
        model = fcl.BVHModel()
        model.beginModel(len(self.vertices), len(self.triangle_faces))
        model.addSubModel(self.vertices.astype(np.float32), self.triangle_faces.astype(np.int32))
        model.endModel()
        return fcl.CollisionObject(model)

    def _build_ellipsoid_fcl(self):
        shape = fcl.Ellipsoid(*self.ellipsoid_radii)
        tf = fcl.Transform(np.eye(3, dtype=np.float32), np.array([0, 12, 1], dtype=np.float32))
        return fcl.CollisionObject(shape, tf)

    def num_collisions(self, state):
        rot = R.from_quat(state.q).as_matrix().astype(np.float32)
        tf = fcl.Transform(rot, state.x.astype(np.float32))
        self.ellipsoid_fcl.setTransform(tf)
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(self.fcl_env, self.ellipsoid_fcl, req, res)
        return int(res.is_collision)
    
    def get_closest_points(self, state):
        """
        Returns the closest points between the ellipsoid (in given state)
        and the voxel environment, if they do not collide.

        Returns:
            distance (float): minimum distance between ellipsoid and voxel env
            point_on_ellipsoid (np.ndarray): closest point on ellipsoid
            point_on_env (np.ndarray): closest point on voxel mesh
        """
        rot = R.from_quat(state.q).as_matrix().astype(np.float32)
        tf = fcl.Transform(rot, state.x.astype(np.float32))
        self.ellipsoid_fcl.setTransform(tf)

        req = fcl.DistanceRequest(enable_nearest_points=True)
        res = fcl.DistanceResult()

        fcl.distance(self.fcl_env, self.ellipsoid_fcl, req, res)

        return res.min_distance, np.array(res.nearest_points[1]), np.array(res.nearest_points[0])
    
    def get_normal_plane(self, state):
        _, p_ellipsoid, p_plane = self.get_closest_points(state)
        #if self.num_collisions(state) > 0:
        #    breakpoint()
        if np.linalg.norm(p_plane - p_ellipsoid) < 1e-5:
            return np.zeros(3)
        return -(p_plane - p_ellipsoid) / np.linalg.norm(p_plane - p_ellipsoid)

class RRTPlanner3D:


    class Node:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent

    def __init__(self, env: CollisionEnvironment3D, step_size=0.3, goal_tolerance=0.3,
                 goal_sample_rate=0.4, max_iters=10000):
        self.env = env
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.path = []

    def _interpolate(self, s1, s2, alpha):
        x = (1 - alpha) * s1.x + alpha * s2.x
        slerp = Slerp([0, 1], R.from_quat([s1.q, s2.q]))
        q = slerp([alpha])[0].as_quat()
        return State(x=x, v=np.zeros(3), q=q, w=np.zeros(3))

    def _is_valid(self, state):
        return self.env.num_collisions(state) == 0

    def _motion_valid(self, from_state, to_state):
        dist = from_state.distance_to(to_state)
        steps = int(np.ceil(dist / (self.step_size / 2)))
        for i in range(steps + 1):
            alpha = i / steps
            if not self._is_valid(self._interpolate(from_state, to_state, alpha)):
                return False
        return True

    def _sample(self, goal):
        if np.random.rand() < self.goal_sample_rate:
            return goal
        pos = np.random.uniform([0, -5, 0],  [10, 20, 2])
        quat = R.random().as_quat()
        return State(pos, np.zeros(3), quat, np.zeros(3))

    def _reconstruct(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def plan(self, start, goal):
        tree = [self.Node(start)]
        for _ in range(self.max_iters):
            rand = self._sample(goal)
            nearest = min(tree, key=lambda n: n.state.distance_to(rand))
            new = self._interpolate(nearest.state, rand,
                                    min(self.step_size / nearest.state.distance_to(rand), 1.0))
            if self._motion_valid(nearest.state, new):
                node = self.Node(new, nearest)
                tree.append(node)
                if new.distance_to(goal) < self.goal_tolerance:
                    return self._reconstruct(node)
        return []

    def run(self):
        start = State(np.array([0, -3, 1.0]), np.zeros(3), np.array([0, 0, 0, 1]), np.zeros(3))
        goal = State(np.array([20.0, 15.0, 1.0]), np.zeros(3), np.array([0, 0, 0, 1]), np.zeros(3))

        print("🚀 Planning...")
        self.path = self.plan(start, goal)

        if self.path:
            print(f"✅ Found path with {len(self.path)} states.")
        else:
            print("❌ No path found.")

    def get_path(self):
        return self.path

    def visualize_path(self):
        plotter = pv.Plotter()
        plotter.add_mesh(self.env.voxel_mesh, color="red", opacity=0.4)

        if self.path:
            for s in self.path:
                mesh = pv.ParametricEllipsoid(*self.env.ellipsoid_radii)
                T = np.eye(4)
                T[:3, :3] = R.from_quat(s.q).as_matrix()
                T[:3, 3] = s.x
                mesh.transform(T, inplace=True)
                plotter.add_mesh(mesh, color="lime", opacity=1.0)

        plotter.add_axes_at_origin()
        plotter.add_axes()
        
        plotter.show()

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pcd_path = os.path.join(script_dir, "map.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)

    env = CollisionEnvironment3D(pcd)
    planner = RRTPlanner3D(env)
    planner.run()
    planner.visualize_path()

    print("=== Path ===")
    for i, state in enumerate(planner.path):
        print(f"State {i}:")
        print(f"  Position: {state.x}")
        print(f"  Quaternion: {state.q}")

    # Get closest points 
    start = planner.path[0]
    distance, p_ellipsoid, p_plane = env.get_closest_points(start)
    print (f"Distance: {distance}")
    print (f"Closest point on ellipsoid: {p_ellipsoid}")
    print (f"Closest point on plane: {p_plane}")

if __name__ == "__main__":
    main()
