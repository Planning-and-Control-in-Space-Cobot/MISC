import casadi as ca
import spatial_casadi as sc
import scipy.spatial.transform as trf
import numpy as np
import pyvista as pv
import json
import os
from datetime import datetime
from time import time
import itertools
import multiprocessing as mp


class Obstacle:
    def __init__(self, vertices_local, position, rotation=np.eye(3), safety_distance=0.01):
        self.vertices_local = vertices_local
        self.position = position
        self.rotation = rotation
        self.safety_distance = safety_distance

    def global_vertices(self):
        return self.rotation @ self.vertices_local + self.position[:, None]


class Map:
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)


class TrajectoryGeneration:
    def __init__(self, N=40, save_dir="results"):
        self.N = N
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.P_R_base = np.diag([1 / 0.24**2, 1 / 0.24**2, 1 / 0.10**2])
        self.map_env = self.create_map()

    def create_map(self):
        map_env = Map()
        vertices_obstacle = np.array([
            [0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        ])
        pos_list = [np.array([2.0, 0, 0]), np.array([2.0, 2.3, 0.0]), np.array([0.0, 2.3, 0.0])]
        for pos in pos_list:
            map_env.add_obstacle(Obstacle(vertices_obstacle, pos))
        return map_env

    def run(self, param_set, solver_opts=None):
        result = {
            "params": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in param_set.items()},
            "solver_opts": solver_opts,
            "status": "failed",
            "time": None,
            "final_error": None,
            "trajectory_file": None
        }
        t_start = time()

        try:
            opti = ca.Opti()
            x = opti.variable(13, self.N)
            u = opti.variable(6, self.N)
            dt = opti.variable(1)
            opti.subject_to(opti.bounded(0.05, dt, 0.5))
            opti.set_initial(dt, 0.1)

            x0 = opti.parameter(13)
            xf = opti.parameter(13)
            opti.set_value(x0, param_set["x0"])
            opti.set_value(xf, param_set["xf"])

            J = opti.parameter(3, 3)
            A = opti.parameter(6, 6)
            m = opti.parameter(1)
            opti.set_value(J, param_set["J"])
            opti.set_value(A, param_set["A"])
            opti.set_value(m, param_set["m"])

            pos, vel, quat, ang_vel = x[:3, :], x[3:6, :], x[6:10, :], x[10:, :]
            opti.subject_to(x[:, 0] == x0)

            for i in range(self.N - 1):
                F = ca.mtimes(A[:3, :], u[:, i])
                M = ca.mtimes(A[3:, :], u[:, i])
                R_current = sc.Rotation.from_quat(quat[:, i])
                opti.subject_to(pos[:, i + 1] == pos[:, i] + dt * vel[:, i])
                opti.subject_to(vel[:, i + 1] == vel[:, i] + dt * (1 / m) * ca.mtimes(R_current.as_matrix(), F))
                opti.subject_to(quat[:, i + 1] == self.quaternion_integration(quat[:, i], ang_vel[:, i], dt))
                opti.subject_to(ang_vel[:, i + 1] == ang_vel[:, i] + dt * ca.mtimes(ca.inv(J), M - ca.cross(ang_vel[:, i], ca.mtimes(J, ang_vel[:, i]))))

            for i in range(self.N):
                opti.subject_to(opti.bounded(-2, u[:, i], 2))
                opti.set_initial(quat[:, i], np.array([0, 0, 0, 1]))
                opti.subject_to(opti.bounded(0, pos[2, i], 5))
                opti.subject_to(pos[1, i] >= 0)

            for obs in self.map_env.obstacles:
                xi = opti.variable(3, self.N)
                mu_r = opti.variable(1, self.N)
                nu = opti.variable(1, self.N)

                V_O = obs.global_vertices()
                delta_min = obs.safety_distance

                for i in range(self.N):
                    opti.set_initial(xi[:, i], np.array([2 * delta_min, 0, 0]))
                    opti.set_initial(mu_r[0, i], 0)
                    opti.set_initial(nu[0, i], 0.1)

                    c_R = pos[:, i]
                    R_current = sc.Rotation.from_quat(quat[:, i]).as_matrix()
                    P_R_inv = R_current @ ca.DM(np.linalg.inv(self.P_R_base)) @ R_current.T

                    opti.subject_to(- (1 / 4) * ca.dot(xi[:, i], xi[:, i]) - ca.dot(xi[:, i], c_R) - mu_r[0, i] - nu[0, i] - delta_min ** 2 >= 0)
                    opti.subject_to(V_O.T @ xi[:, i] + mu_r[0, i] >= 0)
                    opti.subject_to(ca.dot(xi[:, i], xi[:, i]) >= 4 * delta_min ** 2)
                    opti.subject_to(nu[0, i] ** 2 >= ca.dot(xi[:, i], ca.mtimes(P_R_inv, xi[:, i])))
                    opti.subject_to(nu[0, i] >= 0)

            cost = 0
            for i in range(self.N):
                cost += ca.mtimes((pos[:, i] - xf[:3]).T, (pos[:, i] - xf[:3]))
            cost += 1000 * dt
            opti.minimize(cost)

            solver_opts["max_iter"] = 10000
            solver_opts["max_cpu_time"] = 180
            solver_opts["print_level"] = 0

            opti.solver("ipopt", {"print_time": 0}, solver_opts)
            sol = opti.solve()

            final_pos = sol.value(x[:3, -1])
            desired_pos = param_set["xf"][:3]
            error = np.linalg.norm(final_pos - desired_pos)

            result.update({
                "status": "success",
                "time": time() - t_start,
                "final_error": error
            })

            traj_filename = os.path.join(
                self.save_dir,
                f"traj_tol_{solver_opts['tol']:.1e}_viol_{solver_opts['constr_viol_tol']:.1e}_dual_{solver_opts['dual_inf_tol']:.1e}.npz"
            )
            np.savez(traj_filename, x=sol.value(x), u=sol.value(u))
            result["trajectory_file"] = traj_filename

        except Exception as e:
            result["error_msg"] = str(e)

        return result

    def save_result(self, all_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/batch_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(all_results, f, indent=4)

    def quaternion_integration(self, q, w, dt):
        w_norm = ca.sqrt(ca.mtimes(w.T, w) + 1e-3)
        q_ = ca.vertcat(w / w_norm * ca.sin(w_norm * dt / 2), ca.cos(w_norm * dt / 2))
        return self.quaternion_multiplication(q_, q)

    def quaternion_multiplication(self, q1, q2):
        q1x, q1y, q1z, q1w = q1[0], q1[1], q1[2], q1[3]
        q2x, q2y, q2z, q2w = q2[0], q2[1], q2[2], q2[3]
        q_ = ca.vertcat(
            ca.horzcat(q1w, q1z, -q1y, q1x),
            ca.horzcat(-q1z, q1w, q1x, q1y),
            ca.horzcat(q1y, -q1x, q1w, q1z),
            ca.horzcat(-q1x, -q1y, -q1z, q1w),
        )
        return q_ @ ca.vertcat(q2x, q2y, q2z, q2w)


def run_instance(args):
    tg_local = TrajectoryGeneration()
    return tg_local.run(*args)


if __name__ == "__main__":
    base_params = {
        "x0": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        "xf": np.array([8, 3, 0, 0, 0, 0, 0.707, 0, 0, 0.707, 0, 0, 0]),
        "J": np.load("J_matrix.npy"),
        "A": np.load("A_matrix.npy"),
        "m": np.load("mass.npy")
    }

    base_tol = 1e-4
    exponents = np.arange(-3, 4)
    scale_factors = 10.0 ** exponents

    param_sets = []
    for tol_sf, viol_sf, dual_inf_sf in itertools.product(scale_factors, repeat=3):
        solver_opts = {
            "tol": base_tol * tol_sf,
            "constr_viol_tol": base_tol * viol_sf,
            "dual_inf_tol": base_tol * dual_inf_sf,
        }
        param_sets.append((base_params, solver_opts))

    with mp.Pool(processes=4) as pool:
        all_results = pool.map(run_instance, param_sets)

    TrajectoryGeneration().save_result(all_results)
    print(f"Completed {len(all_results)} runs.")
