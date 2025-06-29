import pyvista as pv
import trimesh as tm 

import os

mesh = tm.load_mesh(os.path.join(os.path.dirname(__file__), "space_cobot.stl"))
box = tm.primitives.Box(extents=(0.45, 0.45, 0.12), transform=mesh.bounds)
#box = pv.Box((
#    -0.225, 0.225, -0.225, 0.225, -0.06, 0.06
#))

pv_ = pv.Plotter()
pv_.add_mesh(mesh, color='blue', show_edges=True, opacity=0.5)
pv_.add_mesh(box, color='red', show_edges=True, opacity=0.5)
pv_.show_grid()
pv_.show()



