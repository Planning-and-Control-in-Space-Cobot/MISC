"""MIT License.

Copyright (c) 2025 André Rebelo Teixeira

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from scipy.integrate import solve_ivp

from AbstractModel import Model
from typing import Callable

class Simulator:
    """A simulator that integrates the Model3D dynamics over time."""

    def __init__(self, model : Model, dt : float = 0.001):
        """Initialize the simulator.

        Parameters:
            model (Model3D): The dynamics model.
            initial_state (np.array): Initial 13D state vector.
            dt (float): Simulation time step.
        """
        self.model = model
        self.dt = dt

    def step(self, state : np.ndarray, u : np.ndarray):
        """Simulate one time step forward.

        Parameters:
            u (np.array): Control input [Fx, Fy, Fz, Mx, My, Mz].

        Returns:
            np.array: Updated state after one step.
        """
        t_span = (0, self.dt)  # Integrate over one time step
        sol = solve_ivp(
            fun=lambda t, s: self.model.f(t, s, u),
            t_span=t_span,
            y0=state,
            method="RK45",
            t_eval=[self.dt],
        )

        if sol.status != 0:
            raise ValueError(f"Integration failed: {sol.message}")
        
        return sol.y[:, -1]


    def simulate(self, state : np.ndarray,  u: np.ndarray, t : float):
        """Simulate the system for a given time period with a constant input

        Parameters:
            u (np.array): Control input [Fx, Fy, Fz, Mx, My, Mz].
            t (float): Time to simulate forward.

        Returns:
            np.array: Final state after simulation.
        """
        n_steps = int(t / self.dt)
        sol = solve_ivp(
            fun=lambda t, s: self.model.f(t, s, u),
            t_span=(0, t),
            y0=self.state,
            method="RK45",
            t_eval=np.linspace(0, t, n_steps),
        )

        if sol.status != 0:
            raise ValueError(f"Integration failed: {sol.message}")
        
        return sol.y 
    
