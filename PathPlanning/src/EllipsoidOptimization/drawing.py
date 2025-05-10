import trimesh
import numpy as np
import pyvista as pv

# Load the CAD model
scene = trimesh.load_mesh("space_cobot.stl")
# Extract the first mesh from the scene
space_cobot_mesh = (
    scene.geometry.values()[0] if isinstance(scene, trimesh.Scene) else scene
)

# Extract vertices and faces
vertices = space_cobot_mesh.vertices
faces = np.hstack(
    [[3] + list(face) for face in space_cobot_mesh.faces]
)  # PyVista format

# Create a PyVista mesh
cad_mesh = pv.PolyData(vertices, faces)

# Define the ellipsoid parameters
ellipsoid_center = np.array([0, 0, 0])  # Position of the ellipsoid (center)
a_robot, b_robot, c_robot = 0.24, 0.24, 0.1  # Semi-axes (20cm radius, 10cm height)

# Generate an ellipsoid mesh
theta = np.linspace(0, 2 * np.pi, 30)
phi = np.linspace(0, np.pi, 30)
theta, phi = np.meshgrid(theta, phi)

x = a_robot * np.cos(theta) * np.sin(phi) + ellipsoid_center[0]
y = b_robot * np.sin(theta) * np.sin(phi) + ellipsoid_center[1]
z = c_robot * np.cos(phi) + ellipsoid_center[2]

ellipsoid_points = np.c_[x.ravel(), y.ravel(), z.ravel()]
ellipsoid_faces = []

# Generate faces for the ellipsoid
for i in range(len(theta) - 1):
    for j in range(len(phi) - 1):
        p1 = i * len(phi) + j
        p2 = p1 + 1
        p3 = (i + 1) * len(phi) + j
        p4 = p3 + 1
        ellipsoid_faces.append([3, p1, p2, p3])  # Triangle 1
        ellipsoid_faces.append([3, p2, p3, p4])  # Triangle 2

ellipsoid_mesh = pv.PolyData(ellipsoid_points, np.array(ellipsoid_faces))

# Create a PyVista plotter
plotter = pv.Plotter()
plotter.add_mesh(cad_mesh, color="gray", opacity=0.6, label="CAD Model")
plotter.add_mesh(ellipsoid_mesh, color="red", opacity=0.4, label="Ellipsoid")

# Set plot properties
plotter.add_axes()
plotter.add_legend()
plotter.show()
