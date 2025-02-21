# MIT License
#
# Copyright (c) 2024 Andre Rebelo Teixeira
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = "Andre Rebelo Teixeira"
__email__ = "__andre.r.teixeira@hotmail.com__"
__version__ = "0.0.1"

import casadi as ca
import numpy as np
import scipy.spatial.transform as trf

import time

class FullSystemDynamics:
    def __init__(
        self,
        J: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        m: float,
        N: int = 10,
        T: float = 1 / 50,
    ):
        """
        Constructor for the the full system dynamics class

        Parameters:
            J - np.ndarray - Inertia matrix of the system
            c - np.ndarray - Vector with center of mass o the robot
            A - np.ndarray - Matrix with the transformation of the propellers
            m - float - Mass of the robot
            N - int - Horizon size for the simulation in the controller
            T - float - Time step for each iteration in the controller

        Returns:
            obj - FullSystemDynamics - The new object of type FullSystemDynamics created
        """
        self.J_ = J
        self.c_ = c
        self.A_ = A
        self.m = m
        self.N = N
        self.T = T

    def setup_trajectory_following_problem(self, opts: dict = {}):
        pass

    def setup_problem_no_reference(self, opts: dict = {}):
        """
        This function setups the optimization problem for the system

        Parameters:
            opts - dict - Dictionary with the options for the optimization problem

        Returns:
            None
        """
        if hasattr(self, "opti"):
            raise ValueError(
                "The optimization problem is already setup, please reset the object to create a new one"
            )
        skew = self.skew

        self.opti = ca.Opti()

        ## State variables
        self.w = self.opti.variable(3, self.N)  # Angular velocity in rad/s
        self.w_dot = self.opti.variable(3, self.N)  # Angular acceleration in rad/s²
        self.v = self.opti.variable(3, self.N)  # Linear velocity in m/s
        self.v_dot = self.opti.variable(3, self.N)  # Linear acceleration in m/s²
        self.q = self.opti.variable(4, self.N)  # Quaternion in format [x y z w]
        self.p = self.opti.variable(3, self.N)  # Position in m

        ## Control Variables
        self.u = self.opti.variable(
            6, self.N
        )  # Control input in the format [thrust1 thrust2 thrust3 thrust4 thrust5 thrust6]

        # Inertia Matrix of the robot
        self.J = self.opti.parameter(3, 3)
        self.opti.set_value(self.J, self.J_)

        # Mass of the robot
        self.c = self.opti.parameter(3)
        self.opti.set_value(self.c, [0.0, 0.0, 0.0])

        # Transformation matrix of the propellers
        self.A = self.opti.parameter(6, 6)  # A = np.vstack [A_F, A_M]
        self.opti.set_value(self.A, self.A_)

        # Gravity vector
        self.g = self.opti.parameter(3, 1)
        self.opti.set_value(self.g, np.array([0, 0, 0]))

        # Angular Velocity
        self.w0 = self.opti.parameter(3)
        self.opti.set_value(self.w0, np.array([0, 0, 0]))

        # Angular Acceleration
        self.w_dot0 = self.opti.parameter(3)
        self.opti.set_value(self.w_dot0, np.array([0, 0, 0]))

        # Linear Velocity
        self.v0 = self.opti.parameter(3)
        self.opti.set_value(self.v0, np.array([0, 0, 0]))

        # Linear Acceleration
        self.v_dot0 = self.opti.parameter(3)
        self.opti.set_value(self.v_dot0, np.array([0, 0, 0]))

        # Quaternion
        self.q0 = self.opti.parameter(4)
        self.opti.set_value(
            self.q0, trf.Rotation.from_euler("xyz", [0, 0, 0]).as_quat()
        )

        # Position
        self.p0 = self.opti.parameter(3)
        self.opti.set_value(self.p0, np.array([0, 0, 0]))

        # Actuation in current time step 
        self.u0 = self.opti.parameter(6)
        self.opti.set_value(self.u0, np.array([0, 0, 0, 0, 0, 0]))

        # Desired rotation
        self.desired_rot = self.opti.parameter(3, 3)
        self.opti.set_value(
            self.desired_rot, trf.Rotation.from_euler("xyz", [0, 0, 0]).as_matrix()
        )

        # Desired position
        self.desired_pos = self.opti.parameter(3)
        self.opti.set_value(self.desired_pos, np.array([0, 0, 0]))

        ## Constraints for initial conditions
        self.opti.subject_to(self.w[:, 0] == self.w0)
        self.opti.subject_to(self.w_dot[:, 0] == self.w_dot0)
        self.opti.subject_to(self.v[:, 0] == self.v0)
        self.opti.subject_to(self.v_dot[:, 0] == self.v_dot0)
        self.opti.subject_to(self.q[:, 0] == self.q0)
        self.opti.subject_to(self.p[:, 0] == self.p0)
        self.opti.subject_to(self.u[:, 0] == self.u0)

        for i in range(self.N):
            ## Force and Moment constraints
            R = self.quaternion_rotation_matrix(self.q[:, i])

            # Combine dynamics constraints into a single constraint
            combined_dynamics = ca.vertcat(
                R.T
                @ (
                    self.m @ ca.MX.eye(3) @ self.v_dot[:, i]
                    + self.m @ self.skew(self.c) @ self.w_dot[:, i]
                ),
                self.m @ self.skew(self.c) @ self.v_dot[:, i]
                + self.J @ self.w_dot[:, i]
                + self.skew(self.w[:, i]) @ self.J @ self.w[:, i],
            )

            # Apply the combined constraint using vstack
            self.opti.subject_to(combined_dynamics == self.A @ self.u[:, i])

        for i in range(self.N - 1):
            self.opti.subject_to(
                self.w[:, i + 1] == self.w[:, i] + self.T * self.w_dot[:, i]
            )

            self.opti.subject_to(
                self.v[:, i + 1] == self.v[:, i] + self.T * self.v_dot[:, i]
            )

            self.opti.subject_to(
                self.p[:, i + 1] == self.p[:, i] + self.T * self.v[:, i]
            )

            self.opti.subject_to(
                self.q[:, i + 1]
                == self.quaternion_zeroth_order_integrator(
                    self.q[:, i], self.w[:, i], self.T
                )
            )

        ## Variable Bounds
        self.opti.subject_to(self.opti.bounded(-2, self.u[:, :], 2))

        ## Cost Function
        self.cost = 0

        self.attitude_cost = self.opti.variable(self.N)
        self.position_cost = self.opti.variable(self.N)
        self.u_cost = self.opti.variable(self.N)

        # Iterate through the horizon
        for i in range(self.N):
            self.cost += (self.p[:, i] - self.desired_pos).T @ i @ (
                self.p[:, i] - self.desired_pos
            )

            self.opti.subject_to(self.position_cost[i] == (self.p[:, i] - self.desired_pos).T @ i @ ( self.p[:, i] - self.desired_pos))

            self.cost += self.u[:, i].T @ 0.00001 @ self.u[:, i]
            self.opti.subject_to(self.u_cost[i] == self.u[:, i].T @ 0.1 @ self.u[:, i])


            self.cost += i @ self.rotation_error(self.desired_rot, self.quaternion_rotation_matrix(self.q[:, i]))
            self.opti.subject_to(self.attitude_cost[i] == i @ self.rotation_error(self.desired_rot, self.quaternion_rotation_matrix(self.q[:, i])))

        self.opti.solver("ipopt", opts)
        self.opti.minimize(self.cost)
        # self.opti.solver('sqpmethod', opts)

    def solve_problem_no_reference(
        self,
        w_dot0: np.ndarray,
        w: np.ndarray,
        q: trf.Rotation,
        accel: np.ndarray,
        vel: np.ndarray,
        pos: np.ndarray,
        desired_rot: trf.Rotation,
        desired_pos: np.ndarray,
        current_actuation: np.ndarray,
        prev_sol=None,
    ) -> np.ndarray:
        if not hasattr(self, "opti"):
            raise ValueError(
                "The optimization problem is not setup, please setup the problem before solving it"
            )

        self.opti.set_value(self.w_dot0, w_dot0)
        self.opti.set_value(self.w0, w)
        self.opti.set_value(self.q0, q.as_quat())
        self.opti.set_value(self.v_dot0, accel)
        self.opti.set_value(self.v0, vel)
        self.opti.set_value(self.p0, pos)
        self.opti.set_value(self.desired_rot, desired_rot.as_matrix())
        self.opti.set_value(self.desired_pos, desired_pos)
        self.opti.set_value(self.u0, current_actuation)

        if prev_sol is not None:
            self.opti.set_initial(self.u[:, :-1], prev_sol.value(self.u[:, 1:]))
            self.opti.set_initial(self.w[:, :-1], prev_sol.value(self.w[:, 1:]))
            self.opti.set_initial(self.w_dot[:, :-1], prev_sol.value(self.w_dot[:, 1:]))
            self.opti.set_initial(self.v[:, :-1], prev_sol.value(self.v[:, 1:]))
            self.opti.set_initial(self.v_dot[:, :-1], prev_sol.value(self.v_dot[:, 1:]))
            self.opti.set_initial(self.q[:, :-1], prev_sol.value(self.q[:, 1:]))

        try:
            sol = self.opti.solve()
            print("Good solution")
            return sol, True 
        except RuntimeError as e:
            return None, False

    @staticmethod
    def position_squared_error(desired_pos: ca.MX, current_pos: ca.MX) -> ca.MX:
        """
        This function calculates the squared error between two positions

        Parameters:
            desired_pos - ca.MX - Desired position
            current_pos - ca.MX - Current position

        Returns:
            error - ca.MX - Squared error between the two positions
        """
        return ca.norm_2(desired_pos - current_pos)

    @staticmethod
    def rotation_error(desired_rot: ca.MX, current_rot: ca.MX) -> ca.MX:
        """
        This function calculates the error between two rotation matrices

        Parameters:
            desired_rot - ca.MX - Desired rotation matrix
            current_rot - ca.MX - Current rotation matrix

        Returns:
            error - ca.MX - Error between the two rotation matrices
        """

        return ca.trace(ca.MX.eye(3) - desired_rot.T @ current_rot)

    @staticmethod
    def quaternion_zeroth_order_integrator(q: ca.MX, w: ca.MX, T: float) -> ca.MX:
        """
        This function creates a zeroth order integrator for the quaternion
        According to equation (123) https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7d789dd851ccf6100a3045ab9347b35eb86a9106

        Parameters:
            q - ca.MX - Quaternion to integrate
            w - ca.MX - Angular velocity of the system
            T - float - Time step for the integration

        Returns:
            q_new - ca.MX - Quaternion after the integration
        """
        if q.numel() != 4:
            raise ValueError("The quaternion must have 4 elements")

        if w.numel() != 3:
            raise ValueError("The angular velocity must have 3 elements")

        w_norm = ca.norm_2(w + 1e-6)

        p = ca.vertcat(w / w_norm * ca.sin(w_norm * T / 2), ca.cos(w_norm * T / 2))

        q_ = ca.vertcat(
            ca.horzcat(q[3], q[2], -q[1], q[0]),
            ca.horzcat(-q[2], q[3], q[0], q[1]),
            ca.horzcat(q[1], -q[0], q[3], q[2]),
            ca.horzcat(-q[0], -q[1], -q[2], q[3]),
        )

        return q_ @ p

    @staticmethod
    def quaternion_rotation_matrix(q: ca.MX) -> ca.MX:
        """
        This function create a rotation matrix from a quaternion considering right handed rotations

        Parameters:
            qx - float - Quaternion x component
            qy - float - Quaternion y component
            qz - float - Quaternion z component
            qw - float - Quaternion w component

        Returns:
            rot - ca.MX - Rotation matrix created
        """

        if q.numel() != 4:
            raise ValueError("The quaternion must have 4 elements")

        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        return ca.vertcat(
            ca.horzcat(
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ),
            ca.horzcat(
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ),
            ca.horzcat(
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ),
        )

    @staticmethod
    def skew(v: ca.MX) -> ca.MX:
        """
        This function creates a skew symmetric matrix from a vector

        Parameters:
            v - ca.MX - Vector to create the skew symmetric matrix

        Returns:
            M - ca.MX - Skew symmetric matrix created
        """

        if v.numel() != 3:
            raise ValueError("The vector must have 3 elements")

        return ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0),
        )
