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

def main():
    scene = tm.load("space_cobot.stl")
    # Extract the first mesh from the scene
    space_cobot_mesh = scene.geometry.values()[0] if isinstance(scene, tm.Scene) else scene

    # Extract vertices and faces
    vertices = space_cobot_mesh.vertices
    faces = np.hstack([[3] + list(face) for face in space_cobot_mesh.faces])  # PyVista format

    # Create a PyVista mesh
    space_cobot_mesh = pv.PolyData(vertices, faces)

    # Numpy print options
    np.set_printoptions(linewidth=np.inf)

    # Define obstacles (box and sphere)
    box = tm.creation.box(
        extents=[2, 2, 2],
        transform=tm.transformations.translation_matrix([3, 3, 3]),
    )
    sphere = tm.creation.icosphere(radius=1.5, subdivisions=3)
    sphere.apply_translation([7, 7, 7])
    obstacles = [(box, "red"), (sphere, "blue")]

    # Create the map instance
    map_instance = Map(size_x=10, size_y=10, size_z=10, obstacles=obstacles)

    # Instantiate the RRT planner
    rrt_planner = RRT(
        start=np.array([0, 0, 0]),
        goal=np.array([10, 10, 10]),
        map_instance=map_instance,
    )

    # Create the SDF class
    sdf = SDF(
        definition_x=10,
        definition_y=10,
        definition_z=10,
        map_instance=map_instance,
    )
    sdf.compute_sdf()

    # Compute the path
    path = rrt_planner.plan_path()

    if path is None:
        print("No valid path found.")
        return

    # Convert path to numpy array for visualization
    path_array = np.array(path)

    sdf_interpolant = sdf.interpolation()

    print(
        "Box distance (3, 3, 3)",
        -tm.proximity.signed_distance(box, [[3, 3, 3]]),
    )

    ## Define initial state for both the simulator and the Long Horizon MPC
    initial_position = [1, 1, 1] # meters (x, y, z)
    initial_velocity = [0, 0, 0] # m/s (x, y, z)
    initial_quaternion = trf.Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_quat() # Scalar last : x, y, z, w  
    initial_angular_velocity = [0, 0, 0] # rad/s  (x, y, z) 

    initial_state = np.concatenate(
        [initial_position, initial_velocity, initial_quaternion, initial_angular_velocity]
    ) 

    final_pos = [9, 9, 9] # meters (x, y, z)
    final_vel = [0, 0, 0] # m/s (x, y, z)
    final_quat = trf.Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_quat() # Scalar last : x, y, z, w
    final_ang_vel = [0, 0, 0] # rad/s  (x, y, z)

    final_state = np.concatenate(
        [final_pos, final_vel, final_quat, final_ang_vel]
    )

    m = np.load("mass.npy")
    A = np.load("A_matrix.npy")
    J = np.load("J_matrix.npy")


    LHMPC = LongHorizonMPC(sdf_interpolant, map_instance)
    LHMPC.setup_problem(
        N=40,
        dt=0.5,
        A_ = A,
        J_ = J, 
        m_ = m,
        sdf_interpoland=sdf_interpolant,
    )

    sc_Model = SpaceCobot(m, J, A)
    sim = Simulator(sc_Model)

    pos = np.array(initial_position)
    vel = np.array(initial_velocity)
    quat = np.array(initial_quaternion)
    ang_vel = np.array(initial_angular_velocity)

    state = [initial_state]

    while np.linalg.norm(pos - final_pos) > 1.0:
        sol = LHMPC.solve_problem(
                    pos, 
                    vel, 
                    quat,
                    ang_vel, 
                    final_pos, 
                    final_vel, 
                    final_quat, 
                    final_ang_vel
                    )

        if sol is None:
            print("No solution found")
            break

        u = sol.value(LHMPC.u[:, 0])
        states = sim.simulate(initial_state, u, 0.5)
        state.append(np.array(states[:, -1]))
        pos = states[0:3, -1]
        vel = states[3:6, -1]
        quat = states[6:10, -1]
        ang_vel = states[10:13, -1]
        print(f"pos: {pos}, desired: {final_pos}")

        #print (f"rot cost : {np.linalg.trace(np.eye(3) - trf.Rotation.from_quat(quat).as_matrix() @ trf.Rotation.from_quat(final_quat).as_matrix().T)}")


        initial_state = np.hstack((pos, vel, quat, ang_vel))

    state =  np.array(state).T
    pos_states = state[0:3, :]
    vel_states = state[3:6, :]
    quat_states = state[6:10, :]
    ang_vel_states = state[10:13, :]

    vel_norms = np.linalg.norm(vel_states, axis=0)

    vel_min = np.min(vel_norms)
    vel_max = np.max(vel_norms)
    
    colormap = matplotlib.colormaps["viridis"]  # Recommended

    plotter = map_instance.plot()

    for pos, vel, quat in zip(pos_states.T, vel_norms, quat_states.T):
        cp_mesh =  space_cobot_mesh.copy()
        cp_mesh.points = trf.Rotation.from_quat(quat).apply(cp_mesh.points)
        cp_mesh.translate(pos, inplace=True)
        plotter.add_mesh(cp_mesh, scalars=np.full(cp_mesh.n_points, vel), cmap="viridis", clim=[vel_min, vel_max])

    plotter.show()

    att = trf.Rotation.from_quat(quat_states.T).as_euler("xyz", degrees=True)
    times = np.linspace(1, att.shape[0],att.shape[0]) * 0.5
    plt.plot(times, att[:, 0], label="Roll", color="Blue")
    plt.plot(times, att[:, 1], label="Pitch", color="Blue")
    plt.plot(times, att[:, 2], label="Yaw", color="Blue")
    plt.legend()
    plt.xlabel("Time - s")
    plt.ylabel("Angle - Degrees")
    plt.title("Variations of the angle during horizon")
    plt.show()

if __name__ == "__main__":
    main()
