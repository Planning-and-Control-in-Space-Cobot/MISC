import numpy as np
import scipy.spatial.transform as trf

from controller import FullSystemDynamics
from Thrust

def main():
    J = np.load("J_matrix.npy")
    A = np.load("A_matrix.npy")

    desired_pos = np.array([0.5, 0.5, 0.5])
    desired_attitude = trf.Rotation.from_euler("xyz", [0.1, 0.2, 0.3])

    current_pos = np.array([0.5, 0.5, 0.5])
    current_attitude = trf.Rotation.from_euler("xyz", [0.1, 0.2, 0.3])

    mpc = FullSystemDynamics(J, np.array([0.0, 0.0, 0.0]), A, 3.6)

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

    sol, _ =  mpc.solve_problem_no_reference(np.zeros(3), np.zeros(3), current_attitude, np.zeros(3), np.zeros(3), current_pos, current_attitude, current_pos, np.zeros(6))

    breakpoint()
    
if __name__ == "__main__":
    main()