import os

import pyvista as pv 
import numpy as np 
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from Environment import EnvironmentHandler


robotMesh = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))

simplePcd = o3d.io.read_point_cloud(os.path.join(os.path.dirname(__file__), "simpleMap.pcd"))
middlePcd = o3d.io.read_point_cloud(os.path.join(os.path.dirname(__file__), "middleMap.pcd"))
complexPcd = o3d.io.read_point_cloud(os.path.join(os.path.dirname(__file__), "complexMap.pcd"))

simpleEnv = EnvironmentHandler(simplePcd)
middleEnv = EnvironmentHandler(middlePcd)
complexEnv = EnvironmentHandler(complexPcd)

pv_ = pv.Plotter()

pv_.add_mesh(simpleEnv.voxel_mesh, color="white", opacity=1.0, show_edges=True, line_width=1.0)
simpleMinPos = np.array([-5, 0, -2.5])
simpleMaxPos = np.array([4, 8, 2.5])
simpleStartPos = np.array([3.5, 0.5, 2.0])
simpleGoalPos = np.array([3, 7, 0])
simpleStartQ = np.array([0, 0, 0, 1])
simpleGoalQ = np.array([0, 0, 0, 1])

transform = np.eye(4)
transform[:3, 3] = simpleStartPos
robotMesh_copy = robotMesh.copy()
robotMesh_copy.transform(transform)

pv_.add_mesh(robotMesh_copy, color="green", opacity=1.0, show_edges=True, line_width=1.0)

transform[:3, 3] = simpleGoalPos
robotMesh_copy = robotMesh.copy()
robotMesh_copy = robotMesh.copy()
robotMesh_copy.transform(transform)

pv_.add_mesh(robotMesh_copy, color="red", opacity=1.0, show_edges=True, line_width=1.0)

cube = pv.Box(bounds=(simpleMinPos[0], simpleMaxPos[0], simpleMinPos[1], simpleMaxPos[1], simpleMinPos[2], simpleMaxPos[2]))
pv_.add_mesh(cube, color="blue", opacity=0.1, show_edges=True, line_width=1.0)
pv_.add_axes_at_origin()
pv_.show_grid()
pv_.show()

pv_ = pv.Plotter()
pv_.add_mesh(middleEnv.voxel_mesh, color="white", opacity=1.0, show_edges=True, line_width=1.0  )

middleMinPos = np.array([1.5, 1, 0])
middleMaxPos = np.array([4.0, 6, 10])
middleStartPos = np.array([2.75, 2, 1])
middleGoalPos = np.array([2.75, 2, 7])
middleStartQ = np.array([0, 0.707, 0, 0.707])
middleGoalQ = np.array([0, 0.707, 0, 0.707])


startMesh = robotMesh.copy()
transform[:3, 3] = middleStartPos
transform[:3, :3] = R.from_quat(middleStartQ).as_matrix()
startMesh.transform(transform)

pv_.add_mesh(startMesh, color="green", opacity=1.0, show_edges=True, line_width=1.0)

endMesh = robotMesh.copy()
transform[:3, 3] = middleGoalPos
transform[:3, :3] = R.from_quat(middleGoalQ).as_matrix()
endMesh.transform(transform)

pv_.add_mesh(endMesh, color="red", opacity=1.0, show_edges=True, line_width=1.0)

cube = pv.Box(bounds=(middleMinPos[0], middleMaxPos[0], middleMinPos[1], middleMaxPos[1], middleMinPos[2], middleMaxPos[2]))
pv_.add_mesh(cube, color="blue", opacity=.1, show_edges=True, line_width=1.0)
pv_.add_axes_at_origin()
pv_.show_grid()
pv_.show()

pv_ = pv.Plotter()
complexMapMinPos = np.array([0.0, 3.0, 0.0])
complexMapMaxPos = np.array([3.0, 6.5, 7.0])

startPos = np.array([0.5, 3.5, 1.0])
startQ = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
endPos = np.array([0.5, 5.0, 6.0])
endQ = np.array([0.0, 0.0, 0.0, 1.0])

startMesh = robotMesh.copy()
endMesh = robotMesh.copy()
transform[:3, 3] = startPos
transform[:3, :3] = R.from_quat(startQ).as_matrix()
startMesh.transform(transform)
pv_.add_mesh(startMesh, color="green", opacity=1.0, show_edges=True, line_width=1.0)

transform[:3, 3] = endPos
transform[:3, :3] = R.from_quat(endQ).as_matrix()
endMesh.transform(transform)

pv_.add_mesh(endMesh, color="red", opacity=1.0, show_edges=True, line_width=1.0)

pv_.add_mesh(complexEnv.voxel_mesh, color="white", opacity=1., show_edges=True, line_width=1.0)
pv_.add_axes_at_origin()
cube = pv.Box(bounds=(complexMapMinPos[0], complexMapMaxPos[0], complexMapMinPos[1], complexMapMaxPos[1], complexMapMinPos[2], complexMapMaxPos[2]))
pv_.add_mesh(cube, color="blue", opacity=0.6, show_edges=True, line_width=1.0)
pv_.add_axes_at_origin()
pv_.show_grid()
pv_.show()
