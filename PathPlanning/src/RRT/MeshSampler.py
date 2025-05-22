import trimesh
import numpy as np

class MeshSampler:
    def __init__(self, mesh: trimesh.Trimesh):
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Input must be a trimesh.Trimesh object.")
        self.mesh = mesh

    @classmethod
    def from_file(cls, filepath: str):
        """Load a mesh from a file (OBJ, PLY, STL, etc.)."""
        mesh = trimesh.load_mesh(filepath)
        return cls(mesh)

    @classmethod
    def from_ellipsoid(cls, center, radii):
        """Create an ellipsoid mesh using trimesh primitives."""
        ellipsoid = trimesh.primitives.Ellipsoid(center=center, radii=radii)
        return cls(ellipsoid)

    def sample_uniform(self, num_points: int = 100) -> np.ndarray:
        """Sample approximately evenly distributed points on the mesh surface."""
        points, _ = trimesh.sample.sample_surface_even(self.mesh, num_points)
        return np.array(points)

    def visualize(self, sampled_points=None):
        """Visualize the mesh and optionally sampled points."""
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh)

        if sampled_points is not None:
            cloud = trimesh.points.PointCloud(sampled_points, colors=[255, 0, 0])
            scene.add_geometry(cloud)

        scene.show()
