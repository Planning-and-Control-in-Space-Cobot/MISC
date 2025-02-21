import numpy as np
import matplotlib.pyplot as plt

def plot_mpc_data(state_trajectory, control_inputs, time_steps):
    """
    Plots the state trajectory, control inputs, and quaternion norm over time.

    Parameters:
        state_trajectory (np.ndarray): 2D array of shape (n_states, n_steps) for state variables.
        control_inputs (np.ndarray): 2D array of shape (n_inputs, n_steps) for control inputs.
        time_steps (np.ndarray): 1D array for time steps.
    """
    # Names of state and input variables
    state_variable_names = [
        "Velocity (v_x)", "Velocity (v_y)", "Velocity (v_z)",
        "Position (p_x)", "Position (p_y)", "Position (p_z)",
        "Angular Velocity (ω_x)", "Angular Velocity (ω_y)", "Angular Velocity (ω_z)",
        "Quaternion (q_w)", "Quaternion (q_x)", "Quaternion (q_y)", "Quaternion (q_z)"
    ]

    input_variable_names = [f"Input (u_{i+1})" for i in range(control_inputs.shape[0])]

    # Plot state trajectory
    for i, state_name in enumerate(state_variable_names):
        if i < state_trajectory.shape[0]:  # Only plot states provided
            plt.figure()
            plt.plot(time_steps, state_trajectory[i, :], marker="o")
            plt.title(f"Evolution of {state_name}")
            plt.xlabel("Time Step")
            plt.ylabel(state_name)
            plt.grid()
            plt.show()

    # Plot control inputs
    for i, input_name in enumerate(input_variable_names):
        if i < control_inputs.shape[0]:  # Only plot inputs provided
            plt.figure()
            plt.plot(time_steps, control_inputs[i, :], marker="o")
            plt.title(f"Evolution of {input_name}")
            plt.xlabel("Time Step")
            plt.ylabel(input_name)
            plt.grid()
            plt.show()

    # Calculate and plot quaternion norm if quaternion data is available
    if state_trajectory.shape[0] >= 13:
        quaternion_norm = np.sqrt(np.sum(state_trajectory[9:13, :]**2, axis=0))
        plt.figure()
        plt.plot(time_steps, quaternion_norm, marker="o", color="r")
        plt.title("Norm of Quaternion (q) Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Quaternion Norm")
        plt.grid()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example data
    state_trajectory = np.load("state_trajectory.npy")
    control_inputs = np.load("control_inputs.npy")

    time_steps = np.arange(state_trajectory.shape[1])  # Generate time steps based on data size

    # Call the plot function
    plot_mpc_data(state_trajectory, control_inputs, time_steps)
