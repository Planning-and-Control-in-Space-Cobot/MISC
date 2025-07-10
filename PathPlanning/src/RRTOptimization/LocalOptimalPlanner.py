import os 
import sys
import time
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


class LocalOptimalPlanner:
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
            start : int
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
            start (int):
                Index of the current state in the optimization path for the 
                initial trajectory
            
        Returns:
            list[OptimizationState]:
                List of optimization states representing the optimized path.
            dt (float):
                Time step used in the optimization process.
        """
        #timestart = time.time()
        opti = ca.Opti()
        N = len(initialPath)
         
        x = opti.variable(13, N)
        u = opti.variable(6, N)
        
        opti.subject_to(x[:, 0] == xi.get_state())

        dynamicTime = time.time()
        for i in range(N - 1):
            opti.subject_to(x[:, i+1] == self.robot.f(x[:, i], u[:, i], dt))
        #print(f"Setup dynamics constraints time: {time.time() - dynamicTime:.4f} seconds")

        obstacleAvoidanceTime = time.time() 
        totalObstacles = 0
        for i in range(1, N):
            pos = x[0:3, i]
            R_q = sc.Rotation.from_quat(x[6:10, i])
            maxDistance = maxDistances[i]
            
            _obstacles = [o for o in obstacles if o.iteration == i + start]
            totalObstacles  += len(_obstacles)
            
            for obs in _obstacles:
                for v in self.robot.getVertices():
                    opti.subject_to(
                        obs.normal.reshape((1, 3)) @ (R_q.as_matrix() @ v + pos) >= 
                        obs.normal.reshape((1, 3)) @ obs.closestPointObstacle + obs.safetyMargin
                    )
                
                opti.subject_to(
                    ca.sumsqr(x[0:3, i] - initialPath[i].x) <= 2*maxDistance**2
                )
        #print(f"Setup obstacle avoidance constraints time: {time.time() - obstacleAvoidanceTime:.4f} seconds with {totalObstacles} obstacles")

        boundariesTime = time.time()
        opti.subject_to(opti.bounded(-3, u, 3))
        opti.subject_to(opti.bounded(self.stateMinValues, x, self.stateMaxValues))
        
        #for i in range(1, N):
        #    opti.subject_to(ca.sumsqr(x[6:10]) == 1)
        #print(f"Setup boundaries constraints time: {time.time() - boundariesTime:.4f} seconds")

        costTime = time.time() 
        cost = 0
        #cost += 10000 * ca.fabs(_dt - dt)
        for i in range(1, N):
            cost += (u[:, i] - initialPath[i].u).T @ 0.1 @ (u[:, i] - initialPath[i].u)
        
        for i in range(1, N):
            cost += ca.sumsqr(x[0:3, i] - initialPath[i].x)
            cost += 10 * (1 - ca.dot(x[6:10, i], initialPath[i].q)**2)
            cost += 0.001 * ca.sumsqr(x[3:6, i] - initialPath[i].v)
            cost += 0.001 * ca.sumsqr(x[10:13, i] - initialPath[i].w)
        
        #print(f"Setup cost time: {time.time() - costTime:.4f} seconds")
        timeStart = time.time()
        opti.minimize(cost)
        opti.solver(
            "ipopt", 
            {
                "print_time" : False,
                "expand" : True,
            }, 
            {
                "max_iter" : 100,
                "print_level" : 0, 
                # We are using wall time since we want to limit the total time 
                # of optimization and not only the time in cpu
                "max_wall_time" : 0.5, 
                "linear_solver" : "ma97",
                "mu_strategy" : "adaptive",
                "warm_start_init_point" : "yes",
                "nlp_scaling_method": "equilibration-based",
                "hessian_approximation" : "limited-memory",
            }
        )
        endTime = time.time()
        #print(f"Setup time: {endTime - timestart:.4f} seconds")


        for i in range(N):
            opti.set_initial(x[:, i], initialPath[i].get_state())
            opti.set_initial(u[:, i], initialPath[i].u.flatten())

        try:
            sol = opti.solve_limited()
        except RuntimeError as e:
            opti.debug.show_infeasibilities()
            print(f"Optimization failed: {e}")
            return  None, None, None

        x = sol.value(x)
        u = sol.value(u) 

        optimizedPath = [OptimizationState(
            x=x[0:3, i], 
            v=x[3:6, i], 
            q=x[6:10, i], 
            w=x[10:13, i],
            u=u[:, i],
            i=i)
        for i in range(N)]

        return optimizedPath, sol.value(cost), dt
    
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

