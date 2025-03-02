from scipy.spatial.transform import Rotation as R
import numpy as np


def get_motor_info(motor_data, counter=0, CCW=None):
    print(f"param set-default CA_ROTOR{counter}_PX {motor_data[0]}")
    print(f"param set-default CA_ROTOR{counter}_PY {motor_data[1]}")
    print(f"param set-default CA_ROTOR{counter}_PZ {motor_data[2]}")
    print(
        f"param set-default CA_ROTOR{counter}_KM {0.05 * (1 if counter in CCW else -1)}  "
    )

    r = R.from_euler("xyz", motor_data[3:6], degrees=False)
    body_frame_thrust = r.apply([0, 0, 1])
    body_frame_thrust = body_frame_thrust / np.linalg.norm(body_frame_thrust)

    print(f"param set-default CA_ROTOR{counter}_AX {body_frame_thrust[0]}")
    print(f"param set-default CA_ROTOR{counter}_AY {body_frame_thrust[1]}")
    print(f"param set-default CA_ROTOR{counter}_AZ {body_frame_thrust[2]}\n\n")


poses = [
    [0.0, -0.1655, 0.0, 0.0, -0.958, 0.0],  # 1
    [0.0, 0.1655, 0.0, 0.0, 2.1817, 0.0],  # 2
    [0.1433, -0.0827, 0.0, 0.0, -0.9599, -1.0472],  # 3
    [-0.1433, 0.0827, 0.0, 0.0, 0.9599, 2.0944],  # 4
    [0.1433, 0.0827, 0.0, 0.0, 0.9599, -2.0944],  # 5
    [-0.1433, -0.0827, 0.0, 0.0, -0.9599, 1.047],  # 6
]

CCW = [1, 4, 5]


for i, pose in enumerate(poses):
    get_motor_info(pose, i, CCW)
