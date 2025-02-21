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

import numpy as np
from enum import Enum


class MotorType(Enum):
    CW = 1
    CCW = 2


class ThrustToRpm:
    def __init__(
        self,
        polynomials_coef: np.ndarray,
        min_thrust: float = -1.0,
        max_thrust: float = 1.0,
    ):
        """
        Constructor for the ThrustToRpm class.

        Parameters:
            polynomials_coef (np.ndarray): Coefficients for the motor's thrust-to-rpm polynomial.
                The format of the array should be [a, b, c, d] following the equation:
                thrust = a * rpm^3 + b * rpm^2 + c * rpm + d.
            min_thrust (float): Minimum valid thrust value. Default is -1.0.
            max_thrust (float): Maximum valid thrust value. Default is 1.0.
        """
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        if len(polynomials_coef) != 4:
            raise ValueError("The polynomials_coef array must have 4 elements.")
        self.polynomials_coef = polynomials_coef

    def get_rpm(self, thrust: float) -> float:
        """
        Returns the RPM of the motor given the thrust and motor type.

        Parameters:
            thrust (float): Desired thrust value.
            motor_type (MotorType): The type of motor (CW or CCW).

        Returns:
            float: The calculated RPM, or None if the thrust is out of bounds or no valid RPM is found.
        """

        thrust = round(thrust, 2)
        if not (self.min_thrust <= thrust <= self.max_thrust):
            print(f"Thrust value {thrust} is out of bounds.")
            return None

        if thrust == 0:
            return 0.0

        coefs = np.copy(self.polynomials_coef)
        coefs[-1] -= thrust
        roots = np.roots(coefs)
        roots = roots[np.isreal(roots)].real

        return roots[0] if len(roots) > 0 else 0.0
