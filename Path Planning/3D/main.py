import numpy as np
import trimesh as tm
import pyvista as pv
import scipy.spatial.transform as trf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from long_horizon_mpc import LongHorizonMPC
from Map import Map
from SDF import SDF
from RRT import RRT

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Simulator"))
from Simulator import Simulator
from SpaceCobotModel import SpaceCobot

# New imports
from ellipsoid_Optimizer import EllipsoidOptimizer, AABBObstacle

def main():
    np.set_printoptions(linewidth=200, suppress=True, precision=6)
    scene = tm.load("space_cobot.stl")
    space_cobot_mesh = scene.geometry.values()[0] if isinstance(scene, tm.Scene) else scene
    vertices = space_cobot_mesh.vertices
    faces = np.hstack([[3] + list(face) for face in space_cobot_mesh.faces])
    space_cobot_mesh = pv.PolyData(vertices, faces)

    np.set_printoptions(linewidth=np.inf)

    box = tm.creation.box(
        extents=[2, 2, 2],
        transform=tm.transformations.translation_matrix([7, 2, 7]),
    )

    sphere = tm.creation.icosphere(radius=1.5, subdivisions=3)
    sphere.apply_translation([17, 17, 17])
    obstacles = [(box, "red"), (sphere, "blue")]
    obstacles = []

    map_instance = Map(size_x=5, size_y=5, size_z=5, obstacles=obstacles)

    rrt_planner = RRT(
        start=np.array([0, 0, 0]),
        goal=np.array([10, 10, 10]),
        map_instance=map_instance,
    )

    sdf = SDF(
        definition_x=5,
        definition_y=5,
        definition_z=5,
        max_value=10,
        map_instance=map_instance,
    )
    sdf.compute_sdf()

    path = rrt_planner.plan_path()

    if path is None:
        print("No valid path found.")
        return

    path_array = np.array(path)

    sdf_interpolant = sdf.interpolation()

    initial_position = [0, 0, 0]
    initial_velocity = [0, 0, 0]
    initial_quaternion = trf.Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_quat()
    initial_angular_velocity = [0, 0, 0]

    initial_state = np.concatenate(
        [initial_position, initial_velocity, initial_quaternion, initial_angular_velocity]
    )

    final_pos = [4, 4, 4]
    final_vel = [0, 0, 0]
    final_quat = trf.Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_quat()
    final_ang_vel = [0, 0, 0]

    final_state = np.concatenate(
        [final_pos, final_vel, final_quat, final_ang_vel]
    )

    m = np.load("mass.npy")
    A = np.load("A_matrix.npy")
    J = np.load("J_matrix.npy")

    LHMPC = LongHorizonMPC(sdf_interpolant, map_instance)
    LHMPC.setup_problem(
        N=20,
        dt=0.1,
        A_=A,
        J_=J,
        m_=m,
        sdf_interpoland=sdf_interpolant,
    )

    sc_Model = SpaceCobot(m, J, A)
    sim = Simulator(sc_Model)

    pos = np.array(initial_position)
    vel = np.array(initial_velocity)
    quat = np.array(initial_quaternion)
    ang_vel = np.array(initial_angular_velocity)

    state = [initial_state]
    warm_start_pos = None
    warm_start_vel = None
    warm_start_quat = None
    warm_start_ang_vel = None

    while np.linalg.norm(pos - final_pos) > 0.5 or np.linalg.norm(vel) > 0.1:
        sol = LHMPC.solve_problem(
            pos,
            vel,
            quat,
            ang_vel,
            final_pos,
            final_vel,
            final_quat,
            final_ang_vel,
            warm_start_pos,
            warm_start_vel,
            warm_start_quat,
            warm_start_ang_vel,
        )

        if sol is None:
            print("No solution found")
            break

        warm_start_pos = sol.value(LHMPC.p)
        warm_start_vel = sol.value(LHMPC.v)
        warm_start_quat = sol.value(LHMPC.q)
        warm_start_ang_vel = sol.value(LHMPC.w)

        u = sol.value(LHMPC.u[:, 0])
        states = sim.simulate(initial_state, u, 0.1)
        state.append(np.array(states[:, -1]))
        pos = states[0:3, -1]
        vel = states[3:6, -1]
        quat = states[6:10, -1]
        ang_vel = states[10:13, -1]
        print(f"pos: {pos}, desired: {final_pos}")

        initial_state = np.hstack((pos, vel, quat, ang_vel))

    # Define AABB obstacles
    aabb_obstacles_list = [
        AABBObstacle(x_min=1.5, y_min=1.5, z_min=1.5, x_max=2.5, y_max=2.5, z_max=2.5, safety_margin=1.0),
        #AABBObstacle(x_min=-1, y_min=-1, z_min=0, x_max=0, y_max=1, z_max=1, safety_margin=0.2),
    ]

    # Initialize and Setup the Ellipsoid Optimizer
    ellipsoid_optimizer = EllipsoidOptimizer(m, J, A)
    ellipsoid_optimizer.setup_problem(N=40, dt=0.1, aabb_obstacles_list=aabb_obstacles_list)

    # Solve for optimal trajectory while avoiding AABB obstacles
    solution = ellipsoid_optimizer.solve(
        initial_position, initial_velocity, initial_quaternion, initial_angular_velocity, final_pos, final_quat
    )

    state = np.array(state).T
    pos_states = state[0:3, :]
    vel_states = state[3:6, :]
    quat_states = state[6:10, :]
    ang_vel_states = state[10:13, :]

    vel_norms = np.linalg.norm(vel_states, axis=0)

    vel_min = np.min(vel_norms)
    vel_max = np.max(vel_norms)

    colormap = matplotlib.colormaps["viridis"]

    plotter = map_instance.plot()

    #for pos, vel, quat in zip(pos_states.T, vel_norms, quat_states.T):
    #    cp_mesh = space_cobot_mesh.copy()
    #    cp_mesh.points = trf.Rotation.from_quat(quat).apply(cp_mesh.points)
    #    cp_mesh.translate(pos, inplace=True)
    #    plotter.add_mesh(cp_mesh, scalars=np.full(cp_mesh.n_points, vel), cmap="viridis", clim=[vel_min, vel_max])

    ellipsoid_robot = pv.ParametricEllipsoid(0.24, 0.24, 0.10)

    if solution is None: 
        print("No solution found")
        return
    pos_ = solution.value(ellipsoid_optimizer.p).T
    quat_ = solution.value(ellipsoid_optimizer.q).T

    for p, q in zip(pos_, quat_):
        new_ellipse = ellipsoid_robot.copy()
        
        # Convert quaternion to a 3x3 rotation matrix
        rotation_matrix = trf.Rotation.from_quat(q).as_matrix()
        
        # Create a 4x4 homogeneous transformation matrix
        transformation_matrix = np.eye(4)  # Identity matrix
        transformation_matrix[:3, :3] = rotation_matrix  # Insert rotation
        transformation_matrix[:3, 3] = p  # Insert translation

        # Apply transformation
        new_ellipse.transform(transformation_matrix)
        plotter.add_mesh(new_ellipse, color='green', opacity=0.5)
    


    #spline = pv.Spline(solution.value(ellipsoid_optimizer.p).T, solution.value(ellipsoid_optimizer.p).shape[1] * 10)
    #plotter.add_mesh(spline, color='blue', line_width=3, label='Ellipsoid Optimizer Path')

    obstacle_ellipsoid = pv.ParametricEllipsoid(0.5, 0.5, 1.0)
    obstacle_ellipsoid.translate(2, 2, 2)

    #plotter.add_mesh(obstacle_ellipsoid, color='red', opacity=0.5, label='Obstacle Ellipsoid')
    box_obstacle = pv.Box([1.5, 2.5, 1.5, 2.5, 1.5, 2.5])
    plotter.add_mesh(box_obstacle, color='red', opacity=0.1, label='Obstacle Ellipsoid')


    print("Ellipsoid Path: ", solution.value(ellipsoid_optimizer.p).T)

    plotter.add_legend()
    plotter.show()

if __name__ == "__main__":
    main()