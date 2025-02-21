import casadi as ca 
import spatial_casadi as sc
import numpy as np 

class TrajectoryOptimizer():
    def __init__(self, sdf, map):
        self.sdf = sdf
        self.map = map

    def check_mechanical_parameters(self, m, J, A):
        if A.shape != (6, 6):
            return False 

        if J.shape != (3, 3):
            return False
        
        if np.all(J != np.diag(np.diagonal(J))):
            return False

        if np.all(J <= 0):
            return False
        
        if m.shape != (1,):
            return False

        if m <= 0:
            return False
        
        return True

    @staticmethod
    def quaternion_multiplication(q1, q2):
        '''
        q1 = [x y z w]
        q2 = [x y z w]

        We are going to mulitply q1 @ q2 according to the formula present in 
        'Indirect Kalman filter for 3D attitude Estimation'
        '''

        q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
        q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]

        q_ = ca.vertcat(
            ca.horzcat( q1w,  q1z, -q1y, q1x), 
            ca.horzcat(-q1z,  q1w,  q1x, q1y),
            ca.horzcat( q1y, -q1x,  q1w, q1z),
            ca.horzcat(-q1x, -q1y, -q1z, q1w)
        )

        return ca.vertcat(q2x, q2y, q2z, q2w)

    def quaternion_integration(self, q, w, dt):
        '''
        q = [x y z w]
        w = [wx wy wz]
        dt = scalar

        We are going to use a formula (123) from 'Indirect Kalman filter for 3D attitude Estimation'
        '''

        w_norm = ca.sqrt (ca.mtimes(w.T, w) + 1e-3)

        q_ = ca.vertcat(
            w / w_norm * ca.sin(w_norm * dt / 2), 
            ca.cos(w_norm * dt / 2)
        )

        return self.quaternion_multiplication(q_, q)

    def setup_problem(self,
                    N,
                    A_, 
                    J_,
                    m_, 
                    actuation_cost = 1000,
                    time_cost = 50, 
                    sdf_cost = 20):

        if not self.check_mechanical_parameters(m_, J_, A_):
            raise ValueError("Invalid mechanical parameters")

        self.opti = ca.Opti()

        self.N = N

        self.w = self.opti.variable(3, self.N) # Ang vel in rad/s
        self.v = self.opti.variable(3, self.N) # Lin vel in m/s
        self.q = self.opti.variable(4, self.N) # Quat format [x y z w]
        self.p = self.opti.variable(3, self.N) # Position in m
        self.u = self.opti.variable(6, self.N) # Control inputs in N
        self.dt = self.opti.variable(1) # Delta Time in s

        self.J = self.opti.parameter(3, 3) # Inertia matrix in kg m^2
        self.A = self.opti.parameter(6, 6) # Mixer Matrix 
        self.m = self.opti.parameter(1) # Mass in kg
    
        ## Initial Condition Parameters
        self.start_pos = self.opti.parameter(3)
        self.start_vel = self.opti.parameter(3)
        self.start_quat = self.opti.parameter(4)
        self.start_ang_vel = self.opti.parameter(3)

        ## Final Condition Parameters
        self.goal_pos = self.opti.parameter(3)
        self.goal_vel = self.opti.parameter(3)
        self.goal_quat = self.opti.parameter(4)
        self.goal_ang_vel = self.opti.parameter(3)

        # Parameter Values
        self.opti.set_value(self.J, J_)
        self.opti.set_value(self.A, A_)
        self.opti.set_value(self.m, m_) 
    
        ## Constraints

        # Time Constraints
        self.opti.subject_to(self.dt > 0) # Ensure it is positive

        # Initial State Constraints
        self.opti.subject_to(self.p[:, 0] == self.start_pos)
        self.opti.subject_to(self.v[:, 0] == self.start_vel)
        self.opti.subject_to(self.q[:, 0] == self.start_quat)
        self.opti.subject_to(self.w[:, 0] == self.start_ang_vel)

        # Final State Constraints
        self.opti.subject_to(self.p[:, -1] == self.goal_pos)
        self.opti.subject_to(self.v[:, -1] == self.goal_vel)
        self.opti.subject_to(self.q[:, -1] == self.goal_quat)
        self.opti.subject_to(self.w[:, -1] == self.goal_ang_vel)

        # Dynamics Constraints
        for i in range(self.N - 1):
            F = ca.mtimes([self.A[3:, :], self.u[:, i]]) # Force in N
            M = ca.mtimes([self.A[:3, :], self.u[:, i]]) # Torque in Nm

            R = sc.Rotation.from_quat(self.q[:, i], "xyzw")

            # P[t+1] = P[t] + dt * V[t]
            self.opti.subject_to(self.p[:, i+1] == self.p[:, i] + self.dt * self.v[:, i])
            # V[t+1] = V[t] + dt * R * F[t] / m
            self.opti.subject_to(self.v[:, i+1] == self.v[:, i] + self.dt * R.as_matrix() @ F / self.m)
            
            self.opti.subject_to(self.q[:, i+1] == self.quaternion_integration(self.q[:, i], self.w[:, i], self.dt))

            self.opti.subject_to(self.w[:, i+1] == self.w[:, i] + self.dt * ca.inv(self.J) @ (M - ca.cross(self.w[:, i], self.J @ self.w[:, i])))
        
        ## Full horizon constraints
        for i in range(N):
            self.opti.subject_to(self.opti.bounded(-2, self.u[:, i], 2))

            ## Later this should be uncommented
            #self.opti.subject_to(self.sdf(self.p[:, i]) > 0)

            self.opti.subject_to(ca.norm_2(self.q[:, i]) == 1)

        ## Cost Function
        cost = 0

        for i in range(N):
            cost += self.u[:, i].T @ self.u[:, i] * actuation_cost
            cost += self.dt * time_cost * self.dt 

        self.opti.minimize(cost)

        p_opts = {"expand": False}
        s_opts = {
            "max_cpu_time": 10, 
            "max_iter": 1000
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def solve_problem(self,
                    start_pos, 
                    start_vel,
                    start_quat,
                    start_ang_vel,
                    goal_pos,
                    goal_vel,
                    goal_quat,
                    goal_ang_vel):

        self.opti.set_value(self.start_pos, start_pos)
        self.opti.set_value(self.start_vel, start_vel)
        self.opti.set_value(self.start_quat, start_quat)
        self.opti.set_value(self.start_ang_vel, start_ang_vel)

        self.opti.set_value(self.goal_pos, goal_pos)
        self.opti.set_value(self.goal_vel, goal_vel)
        self.opti.set_value(self.goal_quat, goal_quat)
        self.opti.set_value(self.goal_ang_vel, goal_ang_vel)

        self.opti.set_initial(self.dt, 0.1)
        
        for i in range(self.N):
            self.opti.set_initial(self.q[:, i], start_quat)

        try:
            sol = self.opti.solve()
            return sol
        except:
            return None







    

