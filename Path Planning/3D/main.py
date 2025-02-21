import os
import sys 
import random 
import numpy as np
import casadi as ca
import spatial_casadi as sc

import trimesh as tm
import pymesh

import pyvista as pv
from scipy.spatial import cKDTree, KDTree
from scipy.interpolate import griddata

from path_optimizer import TrajectoryOptimizer


class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent

class RRT:
    def __init__(self, start, goal, map_instance, step_size=0.5, max_iterations=1000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.map = map_instance
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tree = [self.start]
    
    def get_nearest_node(self, sample):
        """Finds the nearest node in the tree to a sampled point."""
        positions = np.array([node.position for node in self.tree])
        tree = KDTree(positions)
        _, idx = tree.query(sample)
        return self.tree[idx]
    
    def is_collision_free(self, point):
        """Checks if a point is collision-free by checking with obstacles."""
        for mesh, _ in self.map.obstacles:
            if mesh.contains([point]):
                return False
        return True
    
    def steer(self, from_node, to_point):
        """Steers from a node toward a given point within the step size."""
        direction = to_point - from_node.position
        direction = direction / np.linalg.norm(direction) * self.step_size
        new_position = from_node.position + direction
        return Node(new_position, parent=from_node)
    
    def reached_goal(self, node):
        """Checks if a node is close enough to the goal."""
        return np.linalg.norm(node.position - self.goal.position) < self.step_size
    
    def plan_path(self):
        """Executes the RRT algorithm."""
        for _ in range(self.max_iterations):
            # Sample a random point or bias towards goal
            if random.random() < 0.2:  # 20% chance to sample goal directly
                sample = self.goal.position
            else:
                sample = np.array([
                    random.uniform(0, self.map.size_x),
                    random.uniform(0, self.map.size_y),
                    random.uniform(0, self.map.size_z)
                ])
            
            # Get nearest node and steer towards sample
            nearest_node = self.get_nearest_node(sample)
            new_node = self.steer(nearest_node, sample)
            
            # Add node if no collision
            if self.is_collision_free(new_node.position):
                self.tree.append(new_node)
                
                # Check if goal is reached
                if self.reached_goal(new_node):
                    return self.extract_path(new_node)
        
        return None  # No path found
    
    def extract_path(self, node):
        """Extracts the final path by tracing back from the goal."""
        path = []
        while node is not None:
            path.append(node.position)
            node = node.parent
        return path[::-1]  # Reverse to get start-to-goal order

class Map:
    def __init__(self, size_x, size_y, size_z, obstacles):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.obstacles = obstacles
    
    def plot(self):
        """Creates and returns a PyVista plotter with the full map dimensions plotted."""
        plotter = pv.Plotter()
        plotter.set_background("white")
        plotter.show_axes()
        plotter.show_grid()
        plotter.add_axes(interactive=True, line_width=3, labels_off=False)

        # Create a bounding box to represent the full map dimensions
        bounding_box = pv.Box(bounds=(0, self.size_x, 0, self.size_y, 0, self.size_z))
        plotter.add_mesh(bounding_box, color='gray', opacity=0.1, style='wireframe')

        for mesh, color in self.obstacles:
            faces = np.hstack([[len(face)] + list(face) for face in mesh.faces])
            poly_data = pv.PolyData(mesh.vertices, faces)
            plotter.add_mesh(poly_data, color=color, opacity=0.5, show_edges=True)
        
        return plotter

    def get_closest_distances(self, point):
        """
        Compute the distance from a 3D point to the closest face/vertex of each obstacle.

        :param point: (x, y, z) coordinates of the query point.
        :return: List of tuples containing (distance, closest_point, obstacle_color).
        """
        distances = []
        for mesh, color in self.obstacles:
            query = tm.proximity.ProximityQuery(mesh)
            closest_point, distance, _ = query.on_surface([point])
            distances.append((distance[0], closest_point[0], color))

        return distances

class SDF:
    def __init__(self, definition_x, definition_y, definition_z, map_instance):
        self.definition_x = definition_x
        self.definition_y = definition_y
        self.definition_z = definition_z
        self.map = map_instance
        self.sdf_grid = None
        self.sdf_values = None
        self.sdf_array = None


    def compute_sdf(self):
        """Compute the signed distance field (SDF) for the given map using Trimesh, processing one obstacle at a time."""
        # Generate grid points
        x_vals = np.linspace(0, self.map.size_x, self.definition_x)
        y_vals = np.linspace(0, self.map.size_y, self.definition_y)
        z_vals = np.linspace(0, self.map.size_z, self.definition_z)

        self.x_vals = x_vals
        self.y_vals = y_vals
        self.z_vals = z_vals

        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Store the SDF grid in the correct shape
        self.sdf_grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # Initialize a 3D array for SDF values
        self.sdf_values = np.full((self.definition_z, self.definition_y, self.definition_x), np.inf)

        # Process each obstacle separately
        for mesh, _ in self.map.obstacles:
            sdf_temp = -tm.proximity.signed_distance(mesh, self.sdf_grid)
            sdf_temp[sdf_temp < 0] = 0  # Set negative values to zero

            # Reshape to (z, y, x)
            sdf_temp = sdf_temp.reshape(self.definition_z, self.definition_y, self.definition_x, order="F")

            self.sdf_values = np.minimum(self.sdf_values, sdf_temp)
    
    def get_sdf(self):
        """Returns the SDF as a numpy array with integer-indexed positions."""
        self.compute_sdf()
        self.sdf_array = self.sdf_values.reshape((self.definition_x, self.definition_y, self.definition_z), order="F")
        return self.sdf_array   


    def get_sdf_with_coords(self, i, j, k):
        """Returns the SDF value and real-world coordinates of a given index (i, j, k)."""
        if 0 <= i < self.definition_x and 0 <= j < self.definition_y and 0 <= k < self.definition_z:
            sdf_value = self.sdf_values[i, j, k]
            coord_x = i * (self.x_vals[1] - self.x_vals[0])
            coord_y = j * (self.y_vals[1] - self.y_vals[0]) 
            coord_z = k * (self.z_vals[1] - self.z_vals[0])
            return sdf_value, (coord_x, coord_y, coord_z)
        else:
            raise IndexError("SDF index out of bounds")
    
    def interpolation(self):
        """Creates a CasADi interpolant for the SDF."""
        if self.sdf_array is None:
            self.get_sdf()
        
        sdf_rows, sdf_cols, sdf_pages = self.sdf_array.shape
        # Knots for the interpolation
        d_knots = [
            np.linspace(0, sdf_pages, self.definition_x),  # X-axis
            np.linspace(0, sdf_cols, self.definition_y),   # Y-axis
            np.linspace(0, sdf_rows, self.definition_z)    # Z-axis
        ]
        # Flatten SDF matrix for interpolation
        d_flat = self.sdf_array.ravel(order="F")

        return ca.interpolant(
            "SDF",
            "bspline", 
            d_knots, 
            d_flat, 
            #{"algorithm": "smooth_linear"}
        )
   
    def plot(self):
        """Creates and returns a PyVista plotter with the SDF plotted."""
        self.compute_sdf()
        plotter = self.map.plot()
        
        # Define the grid explicitly using RectilinearGrid
        x = np.linspace(0, self.map.size_x, self.definition_x)
        y = np.linspace(0, self.map.size_y, self.definition_y)
        z = np.linspace(0, self.map.size_z, self.definition_z)
        grid = pv.RectilinearGrid(x, y, z)
        grid.point_data["SDF Distance"] = self.sdf_values.ravel(order="F")
        
        # Plot interpolated SDF as a volume with transparency
        opacity_levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plotter.add_volume(grid, scalars="SDF Distance", cmap="bwr", opacity=opacity_levels, opacity_unit_distance=4.0)
        plotter.add_scalar_bar(title="SDF Distance")
        return plotter

'''
class TrajectoryOptimizer:
    def __init__(self, start, goal, map_instance, sdf_object, initial_path):
        ## Save the inputs to the class
        self.start = start
        self.goal = goal
        self.map = map_instance
        self.sdf_object = sdf_object
        self.initial_path = initial_path

        self.map_size = (self.map.size_x, self.map.size_y, self.map.size_z)
        self.sdf = self.sdf_object.interpolation()
    
    def create_optimization_problem(self, N, actuation_cost = 1000, time_cost = 50, sdf_cost = 20):
        opti = ca.Opti()
        self.opti = opti
        self.N = N

        # Define the optimization variables
        x = opti.variable(3, N) # Position of robot
        v = opti.variable(3, N) # Velocity of robot
        q = opti.variable(4, N) # Attitude of robot
        w = opti.variable(3, N) # Angular Speed of robot
        u = opti.variable(6, N) # Actuation of the robot
        t = opti.variable(1) # Delta time between each node in the trajectory

        #opti.set_initial(x, np.array(self.initial_path).T)

        self.x = x
        self.v = v
        self.q = q
        self.w = w
        self.u = u
        self.t = t

        # Load the mechanical parameters of the system 
        A = opti.parameter(6, 6) # Mixer Matrix for the system
        J = opti.parameter(3, 3)
        m = opti.parameter() # Mass of the robot

        # Checks the parameters are at least of the correct size as well as some dynamical properties  
        A_ = np.load("A_matrix.npy")
        if A_.shape != (6, 6):
            raise ValueError("The A matrix must be of size 6x6")
        
        J_ = np.load("J_matrix.npy")
        if J_.shape != (3, 3):
            raise ValueError("The J matrix must be of size 3x3")
        
        if np.all(J_ != np.diag(np.diagonal(J_))):
            raise ValueError("The J matrix must be diagonal")

        if np.all(J_ <= 0):
            raise ValueError("The J matrix must have strictly positive values") 
        
        m_ = np.load("mass.npy")
        if m_.shape != (1,):
            raise ValueError("The mass must be a scalar")

        if m_ <= 0:
            raise ValueError("The mass must be stricly positive")

        opti.set_value(A, A_)
                # Dynamics Constraints
        for i in range(self.N - 1):
            self.opti.subject_to(self.p[:, i+1] == self.p[:, i] + self.dt * self.v[:, i])
            self.opti.subject_to(self.v[:, i+1] == self.v[:, i] + self.dt * (self.A @ self.u[:, i]) / self.m)
            self.opti.subject_to(self.q[:, i+1] == self.q[:, i] + 0.5 * self.dt * ca.mtimes([self.q[:, i], self.w[:, i]]))
            self.opti.subject_to(self.w[:, i+1] == self.w[:, i] + 0.5 * self.dt * ca.mtimes([self.J, self.w[:, i]]))
        opti.set_value(J, J_)
        opti.set_value(m, m_)

        self.A = A
        self.m = m
        self.J = J

        # Initial Parameters
        self.start_pos = opti.parameter(3)
        self.start_vel = opti.parameter(3)
        self.start_att = opti.parameter(4)
        self.start_ang = opti.parameter(3)

        # Final Parameters
        self.goal_pos = opti.parameter(3)
        self.goal_vel = opti.parameter(3)
        self.goal_att = opti.parameter(4)
        self.goal_ang = opti.parameter(3)

        # Define the cost function
        cost = 0

        # Define the start constraints
        opti.subject_to(x[:, 0] == self.start_pos)
        opti.subject_to(v[:, 0] == self.start_vel)
        opti.subject_to(q[:, 0] == self.start_att)
        opti.subject_to(w[:, 0] == self.start_ang)

        # Define the goal constraints
        opti.subject_to(x[:, -1] == self.goal_pos)
        opti.subject_to(v[:, -1] == self.goal_vel)
        opti.subject_to(q[:, -1] == self.goal_att)
        opti.subject_to(w[:, -1] == self.goal_ang)

        # Define the time constraints 
        opti.subject_to(t > 0)
        opti.set_initial(t, 1)

        for i in range(N):
            opti.set_initial(q[:, i], np.array([1, 0, 0, 0])) # wxyz


        # Define the dynamics constraints of the system 

        for i in range(N-1):
            M = A[:3, :] @ u[:, i]

            # x = x + t*v
            opti.subject_to(x[:, i+1] == x[:, i] + t*v[:, i])

            # v = v + t*(1/m)*R(q)*F
            opti.subject_to(v[:, i+1] == v[:, i] + t * (1/m) @ sc.Rotation.from_quat(q[:, i]).as_matrix() @ (A[3:, :] @ u[:, i]))

            # w = w + t*J^(-1)*(M - w x Jw)
#            opti.subject_to(w[:, i+1] == w[:, i] + t * ca.inv(J) @ ((A[:3, :] @ u[:,i]) - ca.cross(w[:, i], J @ w[:, i])))
            opti.subject_to(w[:, i+1] == w[:, i] + t * ca.solve(J, (A[:3, :] @ u[:, i]) - ca.cross(w[:, i], J @ w[:, i])))


            # q = q + 1/2 * q_matrix(q)*w -> ensure the quaternion is normalized
            next_q = q[:, i] + 1/2 * t * self.quaternion_q_matrix(q[:, i]) @ w[:, i]

            opti.subject_to(q[:, i+1] ==  next_q / ca.norm_2(next_q + 1e-6))

        for i in range(N):

            # Actuation Constraints
            opti.subject_to(opti.bounded(-2, u[:, i], 2))

            opti.subject_to(ca.norm_2(q[:, i]) == 1)

            ## Map dimension constraints
            opti.subject_to(opti.bounded(0, x[:, 0], self.map_size[0]))
            opti.subject_to(opti.bounded(0, x[:, 1], self.map_size[1]))
            opti.subject_to(opti.bounded(0, x[:, 2], self.map_size[2]))

            ## SDF Crash Constraint
           # opti.subject_to(self.sdf(x[:, i]) > 1)

        ## Define the cost function
        for i in range(N):
            cost += u[:, i].T @ actuation_cost @ u[:, i] + time_cost * t ** 2# + sdf_cost * 1 / (self.sdf(x[:, i])) # later convert the cost to a -log instead
        
        opti.minimize(cost)

        p_opts = {"expand": False, "verbose": False}
        s_opts = {
            "max_cpu_time": 10,  
            "max_iter": 20000,  
            "nlp_scaling_method": "none",  

            # Allow small constraint violations
            "constr_viol_tol": 1e-4,  # Increase from 1e-6 to allow slight violations
            "tol": 1e-4,  
            "compl_inf_tol": 1e-4,  

            # Acceptable solution tolerances
            "acceptable_tol": 1e-4,  
            "acceptable_constr_viol_tol": 1e-4,  # Allow small violations
            "acceptable_compl_inf_tol": 1e-4,  
            "acceptable_obj_change_tol": 1e-4,  

            # Add slight bound relaxation to help feasibility
            "bound_relax_factor": 1e-4,  # Default is 0, increasing it allows IPOPT to relax constraints slightly
        }
        opti.solver("ipopt", p_opts, s_opts)
        print("Number of variables:", opti.nx)
        print("Number of constraints:", opti.ng)
    


    @staticmethod
    def quaternion_q_matrix(q : ca.casadi.MX):
        if q.numel() != 4: 
            raise ValueError("Quaternion must be of size 4")

        q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]

        return ca.vertcat(
            ca.horzcat( q_w, -q_z,  q_y), 
            ca.horzcat( q_z,  q_w, -q_x), 
            ca.horzcat(-q_y,  q_x,  q_w), 
            ca.horzcat(-q_x, -q_y, -q_z)
        )

    def optimize_path(self,
                    start_pos : np.ndarray,
                    start_vel : np.ndarray,
                    start_q : np.ndarray,
                    start_w : np.ndarray, 
                    end_pos : np.ndarray,
                    end_vel : np.ndarray,
                    end_q : np.ndarray,
                    end_w : np.ndarray):
        if start_pos is None or type (start_pos) != np.ndarray or start_pos.shape != (3,):
            raise ValueError("Start position must be of size 3x1")

        if start_vel is None or type (start_vel) != np.ndarray or start_vel.shape != (3,):
            raise ValueError("Start velocity must be of size 3x1")
        
        if start_q is None or type (start_q) != np.ndarray or start_q.shape != (4,):
            raise ValueError("Start quaternion must be of size 4x1")

        if start_w is None or type (start_w) != np.ndarray or start_w.shape != (3,):
            raise ValueError("Start angular velocity must be of size 3x1")
        
        if end_pos is None or type (end_pos) != np.ndarray or end_pos.shape != (3,):
            raise ValueError("End position must be of size 3x1")

        if end_vel is None or type (end_vel) != np.ndarray or end_vel.shape != (3,):
            raise ValueError("End velocity must be of size 3x1")
        
        if end_q is None or type (end_q) != np.ndarray or end_q.shape != (4,):
            raise ValueError("End quaternion must be of size 4x1")
        
        if end_w is None or type (end_w) != np.ndarray or end_w.shape != (3,):
            raise ValueError("End angular velocity must be of size 3x1")
        
        if 0.95 > np.linalg.norm(start_q) or np.linalg.norm(start_q) > 1.05 or 0.95 > np.linalg.norm(end_q) or np.linalg.norm(end_q) > 1.05:
            raise ValueError("The norm of the quaternions must be 1, some tolerance is allowed, but this is too much, make sure the value provided is a quaternion")

        self.opti.set_value(self.start_pos, start_pos)
        self.opti.set_value(self.start_vel, start_vel)
        self.opti.set_value(self.start_att, start_q)
        self.opti.set_value(self.start_ang, start_w)

        self.opti.set_value(self.goal_pos, end_pos)
        self.opti.set_value(self.goal_vel, end_vel)
        self.opti.set_value(self.goal_att, end_q)
        self.opti.set_value(self.goal_ang, end_w)

        for i in range(self.N):
            self.opti.set_initial(self.q[:, i], np.array([1, 0, 0, 0])) # wxyz
   
        try:
            sol = self.opti.solve()
            return sol, sol.value(self.x),sol.value(self.v), sol.value(self.t), sol.value(self.q), sol.value(self.w) 
        except:
            print("Optimization failed")
            print("X values:\n", self.opti.debug.value(self.x).T,end="\n\n")
            np.save("x_values.npy", self.opti.debug.value(self.x).T)
            print("VEL values:\n", self.opti.debug.value(self.v).T, end="\n\n")
            np.save("v_values.npy", self.opti.debug.value(self.v).T)
            print("W vals:\n", self.opti.debug.value(self.w).T, end="\n\n")
            np.save("w_values.npy", self.opti.debug.value(self.w).T)
            print("Q vals:\n", self.opti.debug.value(self.q).T, end="\n\n")
            np.save("q_values.npy", self.opti.debug.value(self.q).T)
            for i, q_ in enumerate(self.opti.debug.value(self.q).T):
                print(i, " norm is ", np.linalg.norm(q_).T)
            print("\nt ", self.opti.debug.value(self.t), end="\n\n")
            np.save("t_values.npy", self.opti.debug.value(self.t))

            print("U values:\n", self.opti.debug.value(self.u).T)
            np.save("u_values.npy", self.opti.debug.value(self.u).T)


            g_sol = self.opti.debug.value(self.opti.g)
            for index, constraint in enumerate(g_sol):
                if constraint > 1e-6:
                    print("Constraint ", index, " value: ", constraint)
                            
            self.opti.debug.show_infeasibilities()

            return None, None, None, None, None, None
            '''



def main():
    np.set_printoptions(linewidth=np.inf)

    # Define obstacles (box and sphere)
    box = tm.creation.box(extents=[2, 2, 2], transform=tm.transformations.translation_matrix([3, 3, 3]))
    sphere = tm.creation.icosphere(radius=1.5, subdivisions=3)
    sphere.apply_translation([7, 7, 7])
    obstacles = [(box, 'red'), (sphere, 'blue')]

    # Create the map instance
    map_instance = Map(size_x=10, size_y=10, size_z=10, obstacles=obstacles)
    
    # Instantiate the RRT planner
    rrt_planner = RRT(start=np.array([0, 0, 0]), goal=np.array([1, 0, 0]),
                       map_instance=map_instance)

    sdf = SDF(definition_x=10, definition_y=10, definition_z=10, map_instance=map_instance)
    sdf.compute_sdf()
    
    # Compute the path
    path = rrt_planner.plan_path()
    
    if path is None:
        print("No valid path found.")
        return
    
    # Convert path to numpy array for visualization
    path_array = np.array(path)
    
    # Plot the map
    plotter = map_instance.plot()
    
    # Plot the path
    plotter.add_mesh(pv.PolyData(path_array), color='green', line_width=4)
    plotter.show()
    
    print("Computed path:", path)
    sdf_interpolant = sdf.interpolation()
    try: 
        sdf_value, (coords) = sdf.get_sdf_with_coords(2, 5, 2)
        print("CasADi SDF Interpolation created.", sdf_value, " sdf query ", sdf_interpolant(coords), " coords ", coords)
    except IndexError:
        print("Index out of bounds")

    print("Box distance (3, 3, 3)", -tm.proximity.signed_distance(box, [[3, 3, 3]]))

    to = TrajectoryOptimizer(sdf_interpolant, map)
    to.setup_problem(20, np.load("A_matrix.npy"), np.load("J_matrix.npy"), np.load("mass.npy"))
    sol = to.solve_problem(
        np.array([0, 0, 0]), 
        np.array([0, 0, 0]), 
        np.array([0, 0, 0, 1]), 
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 0])
    )
    



    #trajectory_optimizer = TrajectoryOptimizer(start=(0, 0, 0), goal=(10, 10, 10), map_instance=map_instance, sdf_object=sdf, initial_path=path)

    #trajectory_optimizer.create_optimization_problem(N=20) 

    #sol, x, v, t, q, w = trajectory_optimizer.optimize_path(
    #    start_pos=np.array([0, 0, 0]),
    #    start_vel=np.array([0, 0, 0]),
    #    start_q=np.array([1, 0, 0, 0]),
    #    start_w=np.array([0, 0, 0]),
    #    end_pos=np.array([1, 0, 0]),
    #    end_vel=np.array([0, 0, 0]),
    #    end_q=np.array([1, 0, 0, 0]),
    #    end_w=np.array([0, 0, 0])
    #)


if __name__ == "__main__":
    main()
