import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class PX4DataPlotter:
    def __init__(self, positions_file, linear_velocities_file, quaternions_file, angular_velocities_file):
        self.positions = np.load(positions_file)
        self.linear_velocities = np.load(linear_velocities_file)
        self.quaternions = np.load(quaternions_file)
        self.angular_velocities = np.load(angular_velocities_file)
        
    def plot_positions(self):
        time = self.positions[:, -1]  # Timestamp
        plt.figure()
        plt.plot(time, self.positions[:, 0], label='X')
        plt.plot(time, self.positions[:, 1], label='Y')
        plt.plot(time, self.positions[:, 2], label='Z')
        plt.xlabel('Time (ms)')
        plt.ylabel('Position (m)')
        plt.title('Position Over Time')
        plt.legend()
        plt.grid()
        

    def plot_angular_speeds(self):
        time = self.angular_velocities[:, -1]  # Timestamp
        plt.figure()
        plt.plot(time, self.angular_velocities[:, 0], label='Roll Rate')
        plt.plot(time, self.angular_velocities[:, 1], label='Pitch Rate')
        plt.plot(time, self.angular_velocities[:, 2], label='Yaw Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('Angular Speed (rad/s)')
        plt.title('Angular Speeds Over Time')
        plt.legend()
        plt.grid()
        

    def plot_linear_speeds(self):
        time = self.linear_velocities[:, -1]  # Timestamp
        plt.figure()
        plt.plot(time, self.linear_velocities[:, 0], label='X')
        plt.plot(time, self.linear_velocities[:, 1], label='Y')
        plt.plot(time, self.linear_velocities[:, 2], label='Z')
        plt.xlabel('Time (ms)')
        plt.ylabel('Linear Speed (m/s)')
        plt.title('Linear Speeds Over Time')
        plt.legend()
        plt.grid()
        

    def plot_euler_angles(self):
        time = self.quaternions[:, -1]  # Timestamp
        quats = self.quaternions[:, :4]  # Extract quaternion values
        euler_angles = R.from_quat(quats).as_euler('xyz', degrees=True)
        
        plt.figure()
        plt.plot(time, euler_angles[:, 0], label='Roll')
        plt.plot(time, euler_angles[:, 1], label='Pitch')
        plt.plot(time, euler_angles[:, 2], label='Yaw')
        plt.xlabel('Time (ms)')
        plt.ylabel('Euler Angles (degrees)')
        plt.title('Euler Angles Over Time')
        plt.legend()
        plt.grid()
        


def main():
    # Load and plot data
    plotter = PX4DataPlotter(
        "positions.npy",
        "linear_velocities.npy",
        "quaternions.npy",
        "angular_velocities.npy"
    )
    plotter.plot_positions()
    plotter.plot_angular_speeds()
    plotter.plot_linear_speeds()
    plotter.plot_euler_angles()
    plt.show()


if __name__ == "__main__":
    main()
