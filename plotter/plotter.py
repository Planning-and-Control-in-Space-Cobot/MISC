from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

# Load the variables from .npy files in the same folder as the script
p = np.load("p.npy")
q = np.load("q.npy")
u = np.load("u.npy")
v = np.load("v.npy")
v_dot = np.load("v_dot.npy")
w = np.load("w.npy")
w_dot = np.load("w_dot.npy")
momentum = np.load("Moment.npy")
force = np.load("F.npy")

A = np.load("src/MPC/full_system_controller/configs/A_matrix.npy")

# Time array for plotting
time = np.linspace(0, 0.1 * (p.shape[1] - 1), p.shape[1])
time_mf = np.linspace(0, 0.1 * (momentum.shape[1] - 1), momentum.shape[1])

# Convert quaternions to Euler angles (roll, pitch, yaw)
# Assuming the quaternion format is [x, y, z, w]
quaternion = q.T  # Transpose to ensure shape (N, 4)
rotation = R.from_quat(quaternion)
euler_angles = rotation.as_euler("xyz", degrees=True)  # Convert to degrees

# List to store all figures for showing at the end
figures = []

## Position
plt.figure()
plt.plot(time, p[0, :], label="x")
plt.plot(time, p[1, :], label="y")
plt.plot(time, p[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position")
plt.legend()
plt.grid(True)

## Orientation
# Convert to euler angles
plt.figure()
Rotations = [R.from_quat(q[:, i]) for i in range(10)]
plt.plot(time, [r.as_euler("xyz", degrees=True)[0] for r in Rotations], label="Roll")
plt.plot(time, [r.as_euler("xyz", degrees=True)[1] for r in Rotations], label="Pitch")
plt.plot(time, [r.as_euler("xyz", degrees=True)[2] for r in Rotations], label="Yaw")

plt.xlabel("Time (s)")
plt.ylabel("Orientation (degrees)")
plt.title("Orientation")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(v[0, :], label="x")
plt.plot(v[1, :], label="y")
plt.plot(v[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity")
plt.legend()

plt.figure()
plt.plot(v_dot[0, :], label="x")
plt.plot(v_dot[1, :], label="y")
plt.plot(v_dot[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration")
plt.legend()

plt.figure()
plt.plot(w[0, :], label="x")
plt.plot(w[1, :], label="y")
plt.plot(w[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity")
plt.legend()

plt.figure()
plt.plot(w_dot[0, :], label="x")
plt.plot(w_dot[1, :], label="y")
plt.plot(w_dot[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (rad/s²)")
plt.title("Angular Acceleration")
plt.legend()

plt.figure()
plt.plot(time_mf, momentum[0, :], label="x")
plt.plot(time_mf, momentum[1, :], label="y")
plt.plot(time_mf, momentum[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Momentum (kg·m/s)")
plt.title("Momentum")

plt.figure()
plt.plot(time_mf, force[0, :], label="x")
plt.plot(time_mf, force[1, :], label="y")
plt.plot(time_mf, force[2, :], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Force")

Forces_ = [A[0:3, :] @ u[:, i] for i in range(10)]

Moment_ = [A[3:, :] @ u[:, i] for i in range(10)]

plt.figure()
plt.plot(time, [f[0] for f in Forces_], label="x")
plt.plot(time, [f[1] for f in Forces_], label="y")
plt.plot(time, [f[2] for f in Forces_], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Force computed from the control input")

plt.figure()
plt.plot(time, [m[0] for m in Moment_], label="x")
plt.plot(time, [m[1] for m in Moment_], label="y")
plt.plot(time, [m[2] for m in Moment_], label="z")
plt.xlabel("Time (s)")
plt.ylabel("Moment (N·m)")
plt.title("Moment computed from the control input")


plt.show()
