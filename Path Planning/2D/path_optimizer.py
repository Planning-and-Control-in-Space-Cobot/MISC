import casadi as ca
import numpy as np

class TrajectoryOptimizer():
    def __init__(self, 
                 initial_path = None, 
                 SDF = None, 
                 map = None): 
        self.initial_path = initial_path
        self.SDF = SDF
        self.map = map

        self.origin = map.start 
        self.goal = map.goal

    def optimize(self):
        N = len(self.initial_path)

        print("N: ", N)
        ### Steps to define the SDF interpolation to ensure continuous double derivatives and faster convergence
        sdf_rows, sdf_cols = self.SDF.get_sdf().shape
        x_size = (0, sdf_cols) # X is the column in the map
        y_size = (0, sdf_rows) # Y is the row in the map

        ## Knots for the interpolation
        d_knots = [
            np.linspace(*x_size, sdf_cols),
            np.linspace(*y_size, sdf_rows)
        ] 

        ## Flatten SDF matrix for the interpolation
        d_flat = self.SDF.get_sdf().ravel(order="F")

        SDF = ca.interpolant(
            "SDF",
            "bspline", 
            d_knots, 
            d_flat, 
            {"algorithm": "smooth_linear"}
        )

        ## Declare value of the opti class
        opti = ca.Opti() 

        ### Optimization variables
        ## This represents the positon of the robot in the world frame
        # x_1 x_2 x_3 ... x_N
        # y_1 y_2 y_3 ... y_N
        x = opti.variable(2, N)
        v = opti.variable(2, N)

        ## This represents the control inputs
        u = opti.variable(6, N)

        ## This represents the time
        t = opti.variable(1)

        ## Initial for value for time is 1s and the minimum time between each point must be greater than 0, since otherwise we have a negative time
        opti.set_initial(t, 1)
        opti.subject_to(t > 0)
    
        ### Optimization parameters such as the mass of the robot and the matrix A
        ## Actuation matrix
        A = opti.parameter(2, 6)
        opti.set_value(A, np.load("A_matrix.npy")[:2, :])

        ## Mass of the robot 
        m = opti.parameter(1) # mass in kg
        opti.set_value(m, 5)

        # Minimum distance to the obstacles
        d_min = 1

        opti.subject_to(x[:, 0] == self.origin) # Ensure the trajectory starts at the origin
        opti.subject_to(x[:, -1] == self.goal) # Ensure the trajectory ends at the goal
        opti.subject_to(v[:, 0] == 0) # Ensure the initial velocity is zero
        opti.subject_to(v[:, -1] == 0) # Ensure the final velocity is zero

        # Dynamical constraintS
        for i in range(N - 1):
            opti.subject_to(v[:, i+1] == v[:, i] + t * (A @ u[:, i]) / m)
            opti.subject_to(x[:, i+1] == x[:, i] + t * v[:, i])

        for i in range(N): 
            opti.subject_to(opti.bounded(-2, u[:, i], 2)) # Ensure the control inputs are bounded
            opti.subject_to(opti.bounded(0, x[:, i], 200)) # Ensure the trajectory is within the map limits


        cost = 0
        
        ## Obstacles Constraint

        for i in range(N-1):
            cost += ca.sumsqr(u[:, i]) * 1000  + 50 * t ** 2 - 20 * ca.log(SDF(x[::-1, i]))

        opti.minimize(cost)

        p_opts = {"expand": False}
        s_opts = {"max_iter": 10000}

        opti.solver("ipopt", p_opts, s_opts)

        # Number of variables and constraints
        num_vars = opti.nx
        num_constraints = opti.ng

        print("Number of variables:", num_vars)
        print("Number of constraints:", num_constraints)


        #opti.set_initial(x, np.array(self.initial_path).T)

        try:
            opti.set_initial(x, np.array(self.initial_path).T)

            sol = opti.solve()
            #print("t: ", sol.value(t))
            #print("x: ", sol.value(x))
            ##print("v: ", sol.value(v))
            return sol, sol.value(x), sol.value(u), sol.value(t), sol.value(v)
        except:
            print("Optimization failed")
            print("x ", opti.debug.value(x))
            print("u ", opti.debug.value(u))
            print("t ", opti.debug.value(t))
            print(self.goal)
            print(self.origin)
            return None, None, None, None, None


class PathLinearizer(): 
    def __init__(self, sol, A, m, dt):
        self.sol = sol  # CasADi solution
        self.A = A  # Actuation matrix
        self.m = m  # Robot mass
        self.dt = dt  # Time step

    def compute_linearized_dynamics(self, x_k, u_k):
        nx = x_k.shape[0]  # Number of states
        nu = u_k.shape[0]  # Number of control inputs

        # Define symbolic variables
        x = ca.MX.sym("x", nx)
        u = ca.MX.sym("u", nu)

        # Define system dynamics
        v_next = x[2:] + self.dt * (self.A @ u) / self.m
        x_next = x[:2] + self.dt * x[2:]

        x_dot = ca.vertcat(x_next, v_next)  # State derivative

        # Compute Jacobians (Linearization)
        A_sym = ca.jacobian(x_dot, x)  # A = df/dx
        B_sym = ca.jacobian(x_dot, u)  # B = df/du

        # CasADi functions for evaluation
        A_func = ca.Function("A_func", [x, u], [A_sym])
        B_func = ca.Function("B_func", [x, u], [B_sym])

        A_k = np.array(A_func(x_k, u_k))
        B_k = np.array(B_func(x_k, u_k))

        return A_k, B_k

    def linearize_trajectory(self, x_traj, u_traj): 
        N = x_traj.shape[1]
        A_list, B_list = [], []

        for k in range(N - 1):
            x_k = x_traj[:, k]
            u_k = u_traj[:, k]

            A_k, B_k = self.compute_linearized_dynamics(x_k, u_k)
            A_list.append(A_k)
            B_list.append(B_k)

        return A_list, B_list

    def compute_linearized_state_control(self, x_traj, u_traj):
        """
        Computes the linearized state and control trajectory using A_k, B_k.
        """
        N = x_traj.shape[1]
        x_lin = np.zeros_like(x_traj)
        u_lin = np.zeros_like(u_traj)

        # Initialize with the first state of the nonlinear trajectory
        x_lin[:, 0] = x_traj[:, 0]
        u_lin[:, 0] = u_traj[:, 0]

        A_list, B_list = self.linearize_trajectory(x_traj, u_traj)

        # Propagate linearized state forward
        for k in range(N - 1):
            u_lin[:, k] = u_traj[:, k]  # Keep initial control the same
            x_lin[:, k+1] = A_list[k] @ x_lin[:, k] + B_list[k] @ u_lin[:, k]

        return x_lin, u_lin
