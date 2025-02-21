import numpy as np

x = np.load("x_values.npy")
v = np.load("v_values.npy")
q = np.load("q_values.npy")
w = np.load("w_values.npy")

u = np.load("u_values.npy")
t = np.load("t_values.npy")

m = np.load("mass.npy")


def quaternion_q_matrix(q):
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]

    q_matrix = np.array([
        [ q_w, -q_z,  q_y], 
        [ q_z,  q_w, -q_x], 
        [-q_y,  q_x,  q_w], 
        [-q_x, -q_y, -q_z]
    ])

    return q_matrix

Q = np.array([9.95622004e-01, -3.79790823e-03, -7.37601436e-02,   5.72823560e-02])
W = np.array([2.-1.28029250e+01,   7.36228322e-01,  1.01948881e+00])


q_next = Q + 0.5 * 2.1731303530144475 * quaternion_q_matrix(Q) @ W
print(q_next / np.linalg.norm(q_next + 1e-6))