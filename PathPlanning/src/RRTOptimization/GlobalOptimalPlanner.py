import os 
import sys
from typing import List, Tuple

import numpy as np 
from scipy.spatial.transform import Rotation as R
import casadi as ca 
import spatial_casadi as sc
import pyvista as pv

from Environment import EnvironmentHandler
from RRTOptimization.Robot import Robot
from RRTOptimization.Obstacle import Obstacle
from RRTOptimization.OptimizationState import OptimizationState

class GlobalOptimalPlanner:
    def __init__(self, 
                 stateMinValues: np.ndarray, 
                 stateMaxValues: np.ndarray,
                 env: EnvironmentHandler,
                 robot: Robot):
        self.stateMinValues = stateMinValues
        self.stateMaxValues = stateMaxValues
        self.env = env
        self.robot = robot

    def optimize(
            self, 
            initialPath : List[OptimizationState], 
            obstacles : List[Obstacle], 
            maxDistances : List[float], 
            dt : float,
            xi : OptimizationState, 
            xf : OptimizationState, 
    ) -> Tuple[List[OptimizationState], float]:
        """Optimize the path considering the full robot planned trajectory
        
        Optimized the full path going from the initial state to the final state, 
        considering all the obstacles in the environment.
        
        Parameters:
            initialPath (list[OptimizationState]):
                List of optimization states representing the initial path.
            obstacles (list[Obstacle]):
                List of the obstacle in the environment for all the time steps.
            maxDistances (list[float]):
                List containing the maximum distance to the obstacles considered
                at each time step during the optimization process.
            xi (OptimizationState):
                First state of the robot in the optimization problem.
            xf (OptimizationState):
                Last state of the robot in the optimization problem.
            
        Returns:
            list[OptimizationState]:
                List of optimization states representing the optimized path.
            dt (float):
                Time step used in the optimization process.
        """
        opti = ca.Opti()
        N = len(initialPath)
         
        x = opti.variable(13, N)
        u = opti.variable(6, N)
        
        _dt = opti.variable(1)
        opti.subject_to(opti.bounded(0, _dt, 1.0))

        opti.subject_to(x[:, 0] == xi.get_state())
        opti.subject_to(x[:, -1] == xf.get_state())

        print(f"Initial Path: {initialPath[0].get_state()} Final Path: {initialPath[-1].get_state()}")

        for i in range(N - 1):
            opti.subject_to(x[:, i+1] == self.robot.f(x[:, i], u[:, i], _dt))
        
        for i in range(1, N):
            pos = x[0:3, i]
            R_q = sc.Rotation.from_quat(x[6:10, i])
            maxDistance = maxDistances[i]
            
            _obstacles = [o for o in obstacles if o.iteration == i]
            
            for obs in _obstacles:
                for v in self.robot.getVertices():
                    opti.subject_to(
                        obs.normal.reshape((1, 3)) @ (R_q.as_matrix() @ v + pos) >=
                        obs.normal.reshape((1, 3)) @ obs.closestPointObstacle #+ obs.safetyMargin
                    )
                
                #opti.subject_to(
                #    ca.sumsqr(x[0:3, i] - initialPath[i].x) <= maxDistance**2
                #)
                print(f"{maxDistance}")
    
        opti.subject_to(opti.bounded(-3, u, 3))
        opti.subject_to(opti.bounded(self.stateMinValues, x, self.stateMaxValues))
        
        for i in range(N):
            opti.subject_to(ca.sumsqr(x[6:10]) == 1)
        
        cost = 0
        cost += 1000 * _dt**2 
#        for i in range(N):
#            cost += u[:, i].T @ 0.1 @ u[:, i]

        opti.minimize(cost)
        opti.solver(
            "ipopt", 
            {
                "print_time" : False,
                "expand": True,
            }, 
            {
                "tol": 1e-6,                        # Overall convergence tolerance
                "constr_viol_tol": 1e-6,           # Constraint violation tolerance
                "acceptable_tol": 1e-6,            # Acceptable overall tolerance
                "acceptable_constr_viol_tol": 1e-6,# Acceptable constraint violation
                "max_iter" : 100,
                "print_level" : 0,
                "linear_solver" : "ma97",
                "mu_strategy" : "adaptive",
                "warm_start_init_point" : "yes",
                "hessian_approximation" : "limited-memory",
            }
        )

        for i in range(N):
            opti.set_initial(x[:, i], initialPath[i].get_state())
            opti.set_initial(u[:, i], initialPath[i].u.flatten())
        opti.set_initial(_dt, dt)

        try:
            sol = opti.solve_limited()
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            return initialPath, dt

        x = sol.value(x)
        u = sol.value(u) 
        dt = sol.value(_dt)

        optimizedPath = [OptimizationState(
            x=x[0:3, i], 
            v=x[3:6, i], 
            q=x[6:10, i], 
            w=x[10:13, i],
            u=u[:, i],
            i=i)
        for i in range(N)]

        return optimizedPath, dt

    
    def visualizeTrajectory(
            self, 
            initialPath : List[OptimizationState], 
            optimizedTrajectory : List[OptimizationState], 
            voxelMesh : pv.PolyData
    ) -> None:
        """Visualizes the initial and optimized trajectory in the environment
        
        In order to better understand the results from the optimization problem, 
        this function will allow the user to visualize the initial path and the 
        optimized trajectory in the environment where the robot is operating in.

        Parameters:
            initialPath (list[OptimizationState]):
                List of optimization states representing the initial path.
            optimizedTrajectory (list[OptimizationState]):
                List of optimization states representing the optimized 
                trajectory.
            voxelMesh (pv.PolyData):
                Voxel mesh representing the environment where the robot is 
                operating in.

        Returns:
            None
        """
        plotter = pv.Plotter()

        for s in initialPath:
            mesh = self.robot.getPVMesh(s.x, R.from_quat(s.q))
            plotter.add_mesh(mesh, color="green", show_edges=True, opacity=0.5)
        
        for s in optimizedTrajectory:
            mesh = self.robot.getPVMesh(s.x, R.from_quat(s.q))
            plotter.add_mesh(mesh, color="blue", show_edges=True, opacity=0.5)
            
        plotter.add_mesh(voxelMesh, color="gray", show_edges=True, opacity=0.5)
        plotter.add_axes()
        plotter.show_grid()
        plotter.add_text("Initial Path", position="upper_left", color="green")
        plotter.add_text("Optimized Trajectory", position="upper_right", color="blue")
        plotter.show()

