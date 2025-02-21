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
        d_min = 10

        opti.subject_to(x[:, 0] == self.origin) # Ensure the trajectory starts at the origin
        opti.subject_to(x[:, -1] == self.goal) # Ensure the trajectory ends at the goal

        # Dynamical constraintS
        for i in range(N - 1):
            opti.subject_to(x[:, i+1] == x[:, i] + t * (A @ u[:, i]) / m)

        for i in range(N): 
            opti.subject_to(SDF(x[::-1, i]) >= d_min) # Ensure the trajectory is at least d_min distance from the obstacles
            opti.subject_to(opti.bounded(-2, u[:, i], 2)) # Ensure the control inputs are bounded
            opti.subject_to(opti.bounded(0, x[:, i], 200)) # Ensure the trajectory is within the map limits


        cost = 0
        
        ## Obstacles Constraint

        for i in range(N-1):
            cost += ca.sumsqr(x[:,i] - self.goal) + ca.sumsqr(u[:, i]) * 1000

        cost += 200000 * t ** 4

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
            return sol, sol.value(x), sol.value(u), sol.value(t)
        except:
            print("Optimization failed")
            print("x ", opti.debug.value(x))
            print("u ", opti.debug.value(u))
            print("t ", opti.debug.value(t))
            print(self.goal)
            print(self.origin)
            return None, None, None, None
