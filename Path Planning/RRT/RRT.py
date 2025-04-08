import open3d as o3d
import pyvista as pv
import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R, Slerp
from dataclasses import dataclass
from typing import List, Optional

# === Função para converter VoxelGrid em Mesh PyVista e FCL ===
def fast_voxel_mesh(voxel_grid):
    voxel_size = voxel_grid.voxel_size
    origin = voxel_grid.origin
    voxels = voxel_grid.get_voxels()
    centers = np.array([voxel.grid_index for voxel in voxels]) * voxel_size + origin + voxel_size / 2

    min_bounds = centers.min(axis=0)
    centers -= min_bounds

    unit_cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32) - 0.5
    unit_cube *= voxel_size

    face_template = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ])

    all_points = []
    all_faces = []
    offset = 0

    for center in centers:
        cube = unit_cube + center
        all_points.append(cube)
        faces = face_template + offset
        all_faces.append(faces)
        offset += 8

    points = np.vstack(all_points)
    faces = np.vstack(all_faces)

    tri_faces = []
    for quad in faces:
        tri_faces.append([3, quad[0], quad[1], quad[2]])
        tri_faces.append([3, quad[0], quad[2], quad[3]])

    tri_faces = np.array(tri_faces, dtype=np.int32).flatten()
    return pv.PolyData(points, tri_faces), points, tri_faces.reshape(-1, 4)[:, 1:], min_bounds

# === Definição de estado em SE(3) ===
@dataclass
class State:
    x: np.ndarray
    v: np.ndarray
    q: np.ndarray
    w: np.ndarray

    def distance_to(self, other: "State") -> float:
        pos_dist = np.linalg.norm(self.x - other.x)
        quat_dist = 1 - np.abs(np.dot(self.q, other.q))
        return pos_dist + quat_dist

# === Nó da árvore RRT ===
class Node:
    def __init__(self, state: State, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent

# === Interpolação em SE(3) ===
def interpolate_SE3(s1: State, s2: State, alpha: float) -> State:
    x = (1 - alpha) * s1.x + alpha * s2.x
    key_rots = R.from_quat([s1.q, s2.q])
    slerp = Slerp([0, 1], key_rots)
    q_interp = slerp([alpha])[0].as_quat()
    return State(x=x, v=np.zeros(3), q=q_interp, w=np.zeros(3))

# === RRT com verificação FCL ===
class FCLRRT:
    def __init__(self, x_init, x_goal, ellipsoid_fcl, voxel_fcl,
                 max_iters=1000, step_size=0.3, goal_tolerance=0.3, goal_sample_rate=0.1):
        self.x_init = x_init
        self.x_goal = x_goal
        self.ellipsoid_fcl = ellipsoid_fcl
        self.voxel_fcl = voxel_fcl
        self.max_iters = max_iters
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_sample_rate = goal_sample_rate
        self.tree = [Node(x_init)]

    def sample_state(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.x_goal
        pos = np.random.uniform(low=[0, 12.8, 0], high=[10, 13.2, 2])
        quat = R.random().as_quat()
        return State(pos, np.zeros(3), quat, np.zeros(3))

    def nearest_node(self, x_rand):
        return min(self.tree, key=lambda node: node.state.distance_to(x_rand))

    def steer(self, from_state, to_state):
        dist = from_state.distance_to(to_state)
        alpha = min(self.step_size / dist, 1.0)
        return interpolate_SE3(from_state, to_state, alpha)

    def is_state_valid(self, state: State) -> bool:
        rot = R.from_quat(state.q).as_matrix().astype(np.float32)
        pos = state.x.astype(np.float32)
        transform = fcl.Transform(rot, pos)
        self.ellipsoid_fcl.setTransform(transform)

        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        fcl.collide(self.voxel_fcl, self.ellipsoid_fcl, request, result)

        return not result.is_collision

    def reached_goal(self, state):
        return state.distance_to(self.x_goal) < self.goal_tolerance

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def plan(self):
        for _ in range(self.max_iters):
            x_rand = self.sample_state()
            x_nearest = self.nearest_node(x_rand)
            x_new = self.steer(x_nearest.state, x_rand)

            if self.is_state_valid(x_new):
                new_node = Node(x_new, x_nearest)
                self.tree.append(new_node)
                if self.reached_goal(x_new):
                    return self.reconstruct_path(new_node)
        return []

# === Criação do plano ===
def create_plane_mesh(center, normal, size=0.5):
    if np.allclose(normal, [0, 0, 1]):
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.cross(normal, [0, 0, 1])
        v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    half = size / 2
    corners = np.array([
        center + half * v1 + half * v2,
        center + half * v1 - half * v2,
        center - half * v1 - half * v2,
        center - half * v1 + half * v2,
    ])
    faces = [4, 0, 1, 2, 3]
    return pv.PolyData(corners, faces)

# === Carregar nuvem de pontos e voxelizar ===
pcd = o3d.io.read_point_cloud("map.pcd")
pcd = pcd.voxel_down_sample(voxel_size=0.05)
pcd_points = np.asarray(pcd.points)
min_bounds = pcd_points.min(axis=0)
pcd_points -= min_bounds
pcd.points = o3d.utility.Vector3dVector(pcd_points)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
voxel_mesh, vertices, triangle_faces, _ = fast_voxel_mesh(voxel_grid)

# === Criar objeto FCL do voxel grid ===
model = fcl.BVHModel()
model.beginModel(len(vertices), len(triangle_faces))
model.addSubModel(vertices.astype(np.float32), triangle_faces.astype(np.int32))
model.endModel()
fcl_obj = fcl.CollisionObject(model)

# === Criar robô como FCL ellipsoid ===
ellipsoid_radii = (0.24, 0.24, 0.1)
ellipsoid_shape = fcl.Ellipsoid(*ellipsoid_radii)
rotation = np.eye(3, dtype=np.float32)
translation = np.array([0, 12, 1], dtype=np.float32)
ellipsoid_transform = fcl.Transform(rotation, translation)
ellipsoid_fcl = fcl.CollisionObject(ellipsoid_shape, ellipsoid_transform)

# === Estados inicial e final ===
x_init = State(x=np.array([-1.0, 12.0, 1.0]), v=np.zeros(3), q=np.array([0, 0, 0, 1]), w=np.zeros(3))
x_goal = State(x=np.array([10.0, 7.0, 1.0]), v=np.zeros(3), q=np.array([0, 0, 0, 1]), w=np.zeros(3))

# === Planejar com RRT ===
rrt = FCLRRT(x_init, x_goal, ellipsoid_fcl, fcl_obj)
print("🚀 Planejando caminho...")
path = rrt.plan()

# === Visualizar resultado ===
plotter = pv.Plotter()
plotter.add_mesh(voxel_mesh, color="red", opacity=0.4, label="Voxel Grid")

if path:
    print(f"✅ Caminho encontrado com {len(path)} estados.")
    for state in path:
        ellipsoid_mesh = pv.ParametricEllipsoid(*ellipsoid_radii)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(state.q).as_matrix()
        T[:3, 3] = state.x
        ellipsoid_mesh.transform(T, inplace=True)
        plotter.add_mesh(ellipsoid_mesh, color="lime", opacity=1.0)

        rot = R.from_quat(state.q).as_matrix().astype(np.float32)
        pos = state.x.astype(np.float32)
        transform = fcl.Transform(rot, pos)
        ellipsoid_fcl.setTransform(transform)

        request = fcl.DistanceRequest(enable_nearest_points=True)
        result = fcl.DistanceResult()
        fcl.distance(fcl_obj, ellipsoid_fcl, request, result)

        if result.min_distance > 0:
            pt_voxel = result.nearest_points[0]
            pt_robot = result.nearest_points[1]
            normal = pt_robot - pt_voxel
            normal /= np.linalg.norm(normal)

            plane_mesh = create_plane_mesh(pt_voxel, normal, size=0.5)
            plotter.add_mesh(plane_mesh, color="cyan", opacity=0.5)
else:
    print("❌ Nenhum caminho encontrado.")

plotter.add_axes()
plotter.add_legend()
plotter.show()
