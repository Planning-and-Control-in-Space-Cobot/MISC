import numpy as np
import scipy.spatial.transform as trf

from controller import FullSystemDynamics
from ThrustToRpm import ThrustToRpm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    J = np.load("J_matrix.npy")
    A = np.load("A_matrix.npy")
    thrust_to_rpm_model = np.load("thrust_model.npy")

    desired_pos = np.array([1.0, 1.0, 1.0])
    desired_attitude = trf.Rotation.from_euler("xyz", [0.0, 0.0, 0.0])

    current_pos = np.array([0.0, 0.0, 0.0])
    current_attitude = trf.Rotation.from_euler("xyz", [0.1, 0.2, 0.3])

    mpc = FullSystemDynamics(J, np.array([0.0, 0.0, 0.0]), A, 3.6, 50)

    thrust_to_rpm = ThrustToRpm(thrust_to_rpm_model, -2, 2)
    print(thrust_to_rpm.get_rpm(1.0))

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "tol": 1e-2,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 5,
            "linear_solver": "mumps",
            "mu_strategy": "adaptive",
            "hessian_approximation": "limited-memory",
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-6,
            "warm_start_bound_frac": 1e-6,
            "max_cpu_time": 1.5,
        },
    #"print_time": 0,
    #"jit": True,
    #"jit_cleanup": True,
    #"jit_options": {
    #    "flags": "",
    #},
    }

    mpc.setup_problem_no_reference(solver_options)

    sol, _ =  mpc.solve_problem_no_reference(
        np.zeros(3), ## w_dot
        np.zeros(3), # w 
        current_attitude, # q
        np.zeros(3), # v_dot
        np.zeros(3), # v
        current_pos, # p
        desired_attitude, # desired q
        desired_pos, # desired_pos 
        np.zeros(6) # u0
    )

    u_cost = sol.value(mpc.u_cost)
    att_cost = sol.value(mpc.attitude_cost)
    pos_cost = sol.value(mpc.position_cost)

    plt.figure(1)
    plt.plot(u_cost)
    plt.title("u_cost")
    plt.figure(2)
    plt.plot(att_cost)
    plt.title("att_cost")
    plt.figure(3)
    plt.plot(pos_cost)
    plt.title("pos_cost")
    plt.figure(4)
    plt.plot(u_cost + att_cost + pos_cost)
    plt.title("total_cost")


    for u_i in sol.value(mpc.u).T:
        for u in u_i:
            rpm = thrust_to_rpm.get_rpm(u)
            print(f"RPM: {rpm} - Thrust: {u}")
        print("\n")

    # Create a 3D plot
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')

    x = sol.value(mpc.p)[0]
    y = sol.value(mpc.p)[1]
    z = sol.value(mpc.p)[2]

    # Plot the trajectory
    ax.plot(x, y, z, label='Trajectory', color='b')

    # Highlight the start and end points
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='r', label='Start', s=100)
    ax.scatter(desired_pos[0], desired_pos[1], desired_pos[2], color='g', label='End', s=100)


    # Add labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set title and legend
    ax.set_title('3D Trajectory')
    ax.legend()

    # Show the plot
    plt.show()
    plt.show() 

if __name__ == "__main__":
    main()