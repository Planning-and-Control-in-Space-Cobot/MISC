import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as trf

from simulator import Simulator as numerical_simulator

'''
    This program compares the results from our numerical simulator (simulator.py) and the Gazebo simulator, this can be used to validate the dynamis of the system, making sure the dynamics used in the control are the correct dynamics
'''

def main():
    J = np.diag([0.03, 0.04, 0.06]) # Inertia matrix
    mass = 3.0 # Mass of the robot
    A = np.array([
       [-0.   , -0.709,  0.709, -0.   , -0.709,  0.709],
       [-0.819,  0.41 ,  0.41 , -0.819,  0.41 ,  0.41 ],
       [ 0.574,  0.574,  0.574,  0.574,  0.574,  0.574],
       [-0.   ,  0.087,  0.087, -0.   , -0.087, -0.087],
       [-0.1  , -0.05 ,  0.05 ,  0.1  ,  0.05 , -0.05 ],
       [-0.125,  0.125, -0.125,  0.125, -0.125,  0.125]
    ])

    sim = numerical_simulator(A, J, mass)

    X0 = np.zeros(13)
    X0[6] = 1 # Initial quaternion

    ## Add data from initial state
    # pos = [0, 0, 0.5]
    # q = [0.707, 0.0, 0.0, 0.707]
    X0[3:6] = np.array([0, 0, 0.5])
    X0[9:13] = np.array([.707, 0, 0, 0.707])

    ## Time to simulate
    T = 10 # seconds
    dt = 0.05 # seconds
    N = int(T / dt)
    U = np.zeros((6, N))

    # Actuation of motor at 1N at all time
    U[0, :] = 1

    state_trajectory = sim.simulate(X0, U, dt)
    




if __name__ == '__main__':
    main()
    