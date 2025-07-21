"""MIT License.

Copyright (c) 2025 André Rebelo Teixeira

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import scipy.spatial.transform as trf
import matplotlib.pyplot as plt
import os
import sys

from simulator import Model3D, Simulator

# Import BagDecoder module from the BagDecoder package
bag_decoder_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "BagDecoder"
)
sys.path.append(bag_decoder_path)
import BagDecoder

# Configure NumPy print options
np.set_printoptions(linewidth=200, suppress=True, precision=6)

# Load the bag file
bag_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "Data",
    "rosbag2_2025_03_02-19_18_30",
    "rosbag2_2025_03_02-19_18_30_0.mcap",
)

# Define topics to extract
topics = [
    "/fmu/out/vehicle_angular_velocity",
    "/fmu/out/vehicle_odometry",
    "/space_cobot_0/motor_command",
    "/fmu/out/vehicle_attitude",
]

# Read ROS bag
bag_reader = BagDecoder.BagReader(bag_path, topics)
bag_reader.read_bag()
messages = bag_reader.get_msgs()
time_0 = bag_reader.bag_start_time  # Used to normalize timestamps

# Extract data from topics
attitude_msgs = messages["/fmu/out/vehicle_attitude"]
attitude_timestamps = np.array([msg[1] for msg in attitude_msgs])
attitude_array_ = np.array(
    [
        [msg[0].q[1], msg[0].q[2], msg[0].q[3], msg[0].q[0]]
        for msg in attitude_msgs
    ]
)
attitude = np.array([trf.Rotation.from_quat(q) for q in attitude_array_])

actuation_msgs = messages["/space_cobot_0/motor_command"]
motor_timestamps = np.array([msg[1] for msg in actuation_msgs])
motor_commands = np.array([msg[0].velocity[0] for msg in actuation_msgs])
motor_commands[0] = 0

position_msgs = messages["/fmu/out/vehicle_odometry"]
pos = np.array([msg[0].position for msg in position_msgs])
vel = np.array([msg[0].velocity for msg in position_msgs])
pos_timestamps = np.array([msg[1] for msg in position_msgs])

angular_velocity_msgs = messages["/fmu/out/vehicle_angular_velocity"]
angular_velocity = np.array([msg[0].xyz for msg in angular_velocity_msgs])
angular_timestamps = np.array([msg[1] for msg in angular_velocity_msgs])


# Create simulator model
J = np.diag([0.085225, 0.085225, 0.085225])
A = np.load("A_matrix.npy")
print(A)
mass = 3.4

model = Model3D(mass, J, A)

# Set initial state
initial_pos = np.array([0, 0, 0])
initial_vel = np.array([0, 0, 0])
initial_quat = np.array(
    [0, 0, 0, 1]
)  # Identity quaternion to remove transformations
initial_w = np.array([0, 0, 0])
initial_state = np.hstack((initial_pos, initial_vel, initial_quat, initial_w))

simulator = Simulator(model, initial_state, dt=0.01)
actuation = np.array([0.5, 0.1, -0.5, 0.2, -0.2, -0.1])
actuation = np.array([0.1] * 6)
actuation = np.array([0.6, 0.0, 0.4, 0.0, 0.2, 0.0])
states = simulator.simulate(actuation, 5)

# Extract simulator data
p_sim = states[0:3, :]
v_sim = states[3:6, :]
w_sim = states[10:13, :]
q_sim = states[6:10, :]
simulator_times = np.arange(0, 5, 0.01)

r_sim = np.array(
    [trf.Rotation.from_quat(q).as_euler("xyz", degrees=True) for q in q_sim.T]
)

# Plot everything in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=False)

# Plot position (Real vs Simulated)
start_pos = pos[0, :]
axs[0, 0].plot(
    (pos_timestamps - time_0) / 1e9,
    pos[:, 0] - start_pos[0],
    label="X_real",
    color="r",
)
axs[0, 0].plot(
    (pos_timestamps - time_0) / 1e9,
    pos[:, 1] - start_pos[1],
    label="Y_real",
    color="g",
)
axs[0, 0].plot(
    (pos_timestamps - time_0) / 1e9,
    pos[:, 2] - start_pos[2],
    label="Z_real",
    color="b",
)
offset = (motor_timestamps[0] - time_0) / 1e9
axs[0, 0].plot(
    simulator_times + offset,
    p_sim[0, :],
    label="X_sim",
    color="r",
    linestyle="--",
)
axs[0, 0].plot(
    simulator_times + offset,
    p_sim[1, :],
    label="Y_sim",
    color="g",
    linestyle="--",
)
axs[0, 0].plot(
    simulator_times + offset,
    -p_sim[2, :],
    label="Z_sim",
    color="b",
    linestyle="--",
)
axs[0, 0].set_title("Position (Real vs Simulated)")
axs[0, 0].set_ylim([-3, 3])
axs[0, 0].legend()

# Linera velocity
start_vel = vel[0, :]
axs[0, 1].plot(
    (pos_timestamps - time_0) / 1e9,
    vel[:, 0] - start_vel[0],
    label="X_real",
    color="r",
)
axs[0, 1].plot(
    (pos_timestamps - time_0) / 1e9,
    vel[:, 1] - start_vel[1],
    label="Y_real",
    color="g",
)
axs[0, 1].plot(
    (pos_timestamps - time_0) / 1e9,
    vel[:, 2] - start_vel[2],
    label="Z_real",
    color="b",
)
axs[0, 1].plot(
    simulator_times + offset,
    v_sim[0, :],
    label="X_sim",
    color="r",
    linestyle="--",
)
axs[0, 1].plot(
    simulator_times + offset,
    v_sim[1, :],
    label="Y_sim",
    color="g",
    linestyle="--",
)
axs[0, 1].plot(
    simulator_times + offset,
    -v_sim[2, :],
    label="Z_sim",
    color="b",
    linestyle="--",
)
axs[0, 1].legend()
axs[0, 1].set_title("Linear Velocity (Real vs Simulated)")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Velocity (m/s)")

r_sim_gz = trf.Rotation.from_matrix(
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
)
counter = 0
for q in q_sim.T:
    r_ = trf.Rotation
    print(q)
    print(
        r_.from_euler(
            "xyz",
            r_.from_quat(q).as_euler("xyz", degrees=True)
            - np.array([0, 0, 90]),
            degrees=True,
        ),
        " ",
        counter,
    )
    counter += 1

r_sim_handled = np.array(
    [
        trf.Rotation.from_euler(
            "xyz",
            trf.Rotation.from_quat(q).as_euler("xyz", degrees=True)
            - np.array([0, 0, 90]),
            degrees=True,
        ).as_euler("xyz", degrees=True)
        for q in q_sim.T
    ]
)
r_sim_handled = np.array([r_sim_gz.apply(r) for r in r_sim_handled])


# PLot Attitude
start_orientation = attitude[0].as_euler("xyz", degrees=True)
axs[1, 0].plot(
    simulator_times + offset,
    r_sim_handled[:, 0],
    label="roll_sim",
    color="r",
    linestyle="--",
)
axs[1, 0].plot(
    simulator_times + offset,
    r_sim_handled[:, 1],
    label="pitch_sim",
    color="g",
    linestyle="--",
)
axs[1, 0].plot(
    simulator_times + offset,
    r_sim_handled[:, 2],
    label="yaw_sim",
    color="b",
    linestyle="--",
)
axs[1, 0].plot(
    (attitude_timestamps - time_0) / 1e9,
    [R.as_euler("xyz", degrees=True)[0] for R in attitude],
    label="roll_real",
    color="r",
)
axs[1, 0].plot(
    (attitude_timestamps - time_0) / 1e9,
    [R.as_euler("xyz", degrees=True)[1] for R in attitude],
    label="pitch_real",
    color="g",
)
axs[1, 0].plot(
    (attitude_timestamps - time_0) / 1e9,
    [R.as_euler("xyz", degrees=True)[2] for R in attitude],
    label="yaw_real",
    color="b",
)
axs[1, 0].set_title("Attitude (Real vs Simulated)")
axs[1, 0].legend()
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Quaternion")

w_sim_handled = np.array([r_sim_gz.apply(w) for w in w_sim.T])

# Plot angular velocity (Real vs Simulated)
start_omega = angular_velocity[0, :]
axs[1, 1].plot(
    (angular_timestamps - time_0) / 1e9,
    angular_velocity[:, 0] - start_omega[0],
    label="X_real",
    color="r",
)
axs[1, 1].plot(
    (angular_timestamps - time_0) / 1e9,
    angular_velocity[:, 1] - start_omega[1],
    label="Y_real",
    color="g",
)
axs[1, 1].plot(
    (angular_timestamps - time_0) / 1e9,
    angular_velocity[:, 2] - start_omega[2],
    label="Z_real",
    color="b",
)
axs[1, 1].plot(
    simulator_times + offset,
    w_sim_handled[:, 0],
    label="X_sim",
    color="r",
    linestyle="--",
)
axs[1, 1].plot(
    simulator_times + offset,
    w_sim_handled[:, 1],
    label="Y_sim",
    color="g",
    linestyle="--",
)
axs[1, 1].plot(
    simulator_times + offset,
    w_sim_handled[:, 2],
    label="Z_sim",
    color="b",
    linestyle="--",
)
axs[1, 1].set_title("Angular Velocity (Real vs Simulated)")
axs[1, 1].legend()
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Angular Velocity (rad/s)")
axs[1, 1].set_xlim(
    [0, (bag_reader.bag_end_time - bag_reader.bag_start_time) / 1e9]
)

plt.tight_layout()
plt.show()
