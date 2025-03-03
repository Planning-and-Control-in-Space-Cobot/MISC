import numpy as np

import trimesh as tm

import pyvista as pv

from long_horizon_mpc import LongHorizonMPC
from Map import Map
from SDF import SDF
from RRT import RRT



def main():
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

    sdf = SDF(
        definition_x=10,
        definition_y=10,
        definition_z=10,
        map_instance=map_instance,
    )
    sdf.compute_sdf()
    sdf.plot().show()

    # Compute the path
    path = rrt_planner.plan_path()

    if path is None:
        print("No valid path found.")
        return

    # Convert path to numpy array for visualization
    path_array = np.array(path)

    # Plot the map
    plotter = map_instance.plot()

    # Plot the path
    plotter.add_mesh(pv.PolyData(path_array), color="green", line_width=4)
    plotter.show()

    print("Computed path:", path)
    sdf_interpolant = sdf.interpolation()
    try:
        sdf_value, (coords) = sdf.get_sdf_with_coords(2, 5, 2)
        print(
            "CasADi SDF Interpolation created.",
            sdf_value,
            " sdf query ",
            sdf_interpolant(coords),
            " coords ",
            coords,
        )
    except IndexError:
        print("Index out of bounds")

    print(
        "Box distance (3, 3, 3)",
        -tm.proximity.signed_distance(box, [[3, 3, 3]]),
    )

    LHMPC = LongHorizonMPC(sdf_interpolant, map_instance)
    LHMPC.setup_problem(
        N=100,
        dt=0.1,
        A_=np.load("A_matrix.npy"),
        J_=np.load("J_matrix.npy"),
        m_=np.load("mass.npy"),
        sdf_interpoland=sdf_interpolant,
    )

    sol = LHMPC.solve_problem(
        start_pos=np.array([0, 0, 0]),
        start_vel=np.array([0, 0, 0]),
        start_quat=np.array([0, 0, 0, 1]),
        start_ang_vel=np.array([0, 0, 0]),
        goal_pos=np.array([10, 10, 10]),
        goal_vel=np.array([0, 0, 0]),
        goal_quat=np.array([0, 0, 0, 1]),
        goal_ang_vel=np.array([0, 0, 0]),
    )

    if sol is not None:
        print(sol.value(LHMPC.p))
        print("\n")
        print(sol.value(LHMPC.v))

        plotter = map_instance.plot()
        p = sol.value(LHMPC.p)
        v = sol.value(LHMPC.v)
        v = np.linalg.norm(v, axis=0)

        polydata_ = pv.PolyData(p.T)
        polydata_["velocity"] =v 

        plotter.add_mesh(polydata_,
                        scalars="velocity",  # Use velocity to color the points
                        cmap="viridis",  # Choose a colormap
                        point_size=10,  # Adjust point size
                        render_points_as_spheres=True)  # Improve visualization)
        plotter.add_scalar_bar(title="Velocity Magnitude")
        plotter.show(auto_close=False)
        plotter.close()




    # to = TrajectoryOptimizer(sdf_interpolant, map)
    # to.setup_problem(len(path), np.load("A_matrix.npy"), np.load("J_matrix.npy"), np.load("mass.npy"), initial_path=path)
    # sol = to.solve_problem(
    #    np.array([0, 0, 0]),
    #    np.array([0, 0, 0]),
    #    np.array([0, 0, 0, 1]),
    #    np.array([0, 0, 0]),
    #    np.array([10, 10, 10]),
    #    np.array([0, 0, 0]),
    #    np.array([0, 0, 0, 1]),
    #    np.array([0, 0, 0])
    # )

    # if sol is not None:
    #    print(sol.value(to.p))
    #    plotter = to.plot_trajectory(map_instance)
    #    plotter.show()

    # if  sol is not None:
    #    p = sol.value(to.p)

    #    for x, y, z in p.T:
    #        print(f"x: {x}, y: {y}, z: {z} - sdf: {sdf_interpolant([x, y, z])}")

    # trajectory_optimizer = TrajectoryOptimizer(start=(0, 0, 0), goal=(10, 10, 10), map_instance=map_instance, sdf_object=sdf, initial_path=path)

    # trajectory_optimizer.create_optimization_problem(N=20)

    # sol, x, v, t, q, w = trajectory_optimizer.optimize_path(
    #    start_pos=np.array([0, 0, 0]),
    #    start_vel=np.array([0, 0, 0]),
    #    start_q=np.array([1, 0, 0, 0]),
    #    start_w=np.array([0, 0, 0]),
    #    end_pos=np.array([1, 0, 0]),
    #    end_vel=np.array([0, 0, 0]),
    #    end_q=np.array([1, 0, 0, 0]),
    #    end_w=np.array([0, 0, 0])
    # )


if __name__ == "__main__":
    main()
