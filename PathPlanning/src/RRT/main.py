# study_birrt.py
import numpy as np
import scipy as sp
import pyvista as pv
import argparse
import open3d as o3d

import time
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import pickle

from Environment import EnvironmentHandler
import rrtcxx  # pybind11 module exposing RRTPlanner3D


# --------------------- Helpers (viz) ---------------------
def _add_pose_mesh(pv_plotter, robot_mesh, state, color):
    x = state.position
    q = state.q  # [x, y, z, w]
    T = np.eye(4)
    T[:3, :3] = sp.spatial.transform.Rotation.from_quat(q).as_matrix()
    T[:3, 3] = x
    cube = robot_mesh.copy()
    cube.transform(T)
    pv_plotter.add_mesh(cube, color=color, show_edges=True)


def state_list_to_dict(path):
    """Convert list[rrtcxx.State] to JSON-serializable list of dicts."""
    return [
        {
            "position": [float(v) for v in s.position],
            "orientation": [float(v) for v in s.q],  # [x, y, z, w]
        }
        for s in path
    ]


# --------------------- Randomized run descriptor ---------------------
class Run:
    def __init__(self, *, stepSize, goalBias, iterations,
                 useKNearest, kNearest,
                 useRStarRadius, gammaRStar,
                 rewireRadius, dimension):
        self.stepSize = float(stepSize)
        self.goalBias = float(goalBias)
        self.iterations = int(iterations)
        self.useKNearest = bool(useKNearest)
        self.kNearest = int(kNearest)
        self.useRStarRadius = bool(useRStarRadius)
        self.gammaRStar = float(gammaRStar)
        self.rewireRadius = float(rewireRadius)
        self.dimension = int(dimension)


# --------------------- Worker for study-case ---------------------
def execute_run(run: Run, common_params):
    """
    Run both Bi-RRT and Bi-RRT* with the same environment/start/goal/bounds,
    but with per-run parameters (including RRT* neighbor logic).
    """
    (env_path,
     start_pos, start_q, goal_pos, goal_q,
     boundsX, boundsY, boundsZ,
     use_payload, payload_translation, payload_size) = common_params

    env = EnvironmentHandler(o3d.io.read_point_cloud(env_path))
    triangle_vertex = env.triangleVertex
    triangle_index = env.triangleIndex

    planner = rrtcxx.RRTPlanner3D(
        triangle_vertex,
        triangle_index,
        np.array(payload_translation),
        np.array(payload_size),
        bool(use_payload),
        int(run.iterations),
        float(run.stepSize),
        float(run.goalBias),
        float(boundsX[0]), float(boundsX[1]),
        float(boundsY[0]), float(boundsY[1]),
        float(boundsZ[0]), float(boundsZ[1])
    )

    # Bi-RRT* neighbor settings
    planner.setUseKNearest(bool(run.useKNearest))
    planner.setKNearest(int(run.kNearest))
    planner.setUseRStarRadius(bool(run.useRStarRadius))
    planner.setGammaRStar(float(run.gammaRStar))
    planner.setRewireRadius(float(run.rewireRadius))   # used if useRStarRadius == False
    planner.setDimension(int(run.dimension))

    start_state = rrtcxx.State(np.array(start_pos), np.array(start_q))
    goal_state = rrtcxx.State(np.array(goal_pos), np.array(goal_q))

    # --- Bi-RRT ---
    birrt_path, birrt_time = planner.biRRT(start_state, goal_state)
    birrt_success = len(birrt_path) > 0
    birrt_pruned = planner.prunePath(birrt_path) if birrt_success else []

    # --- Bi-RRT* ---
    birrt_star_path, birrt_star_time = planner.biRRTStar(start_state, goal_state)
    birrt_star_success = len(birrt_star_path) > 0
    birrt_star_pruned = planner.prunePath(birrt_star_path) if birrt_star_success else []

    print(f"Run Finished")

    return {
        "params": {
            "stepSize": run.stepSize,
            "goalBias": run.goalBias,
            "iterations": run.iterations,
            "useKNearest": run.useKNearest,
            "kNearest": run.kNearest,
            "useRStarRadius": run.useRStarRadius,
            "gammaRStar": run.gammaRStar,
            "rewireRadius": run.rewireRadius,
            "dimension": run.dimension,
            "env": os.path.basename(env_path),
            "boundsX": list(map(float, boundsX)),
            "boundsY": list(map(float, boundsY)),
            "boundsZ": list(map(float, boundsZ)),
            "usePayload": bool(use_payload),
            "payloadTranslation": list(map(float, payload_translation)),
            "payloadSize": list(map(float, payload_size)),
            "startPos": list(map(float, start_pos)),
            "startQuat": list(map(float, start_q)),
            "goalPos": list(map(float, goal_pos)),
            "goalQuat": list(map(float, goal_q)),
        },
        "BiRRT": {
            "success": birrt_success,
            "timeSec": birrt_time,
            "pathLength": len(birrt_path) if birrt_success else 0,
            "prunedPathLength": len(birrt_pruned) if birrt_success else 0,
            "path": state_list_to_dict(birrt_path),
            "prunedPath": state_list_to_dict(birrt_pruned),
        },
        "BiRRTStar": {
            "success": birrt_star_success,
            "timeSec": birrt_star_time,
            "pathLength": len(birrt_star_path) if birrt_star_success else 0,
            "prunedPathLength": len(birrt_star_pruned) if birrt_star_success else 0,
            "path": state_list_to_dict(birrt_star_path),
            "prunedPath": state_list_to_dict(birrt_star_pruned),
        }
    }


# --------------------- Main ---------------------
def main():
    def strToBool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="RRT Path Planning (Bi-RRT vs Bi-RRT*)")

    # Minimal: user only needs to set --study-case and --env
    parser.add_argument("--study-case", type=strToBool, default=False,
                        help="Run randomized study across many parameter settings.")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment point cloud file (relative paths resolved from this script's folder).")

    # Defaults (Bi-RRT* and general) — can be overridden
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--goal-bias", type=float, default=0.05)

    # Start / goal / bounds (same in both modes; override if needed)
    parser.add_argument("--start-pos", type=float, nargs=3, default=[0.5, 3.0, 1.0])
    parser.add_argument("--start-quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
    parser.add_argument("--goal-pos", type=float, nargs=3, default=[0.5, 5.0, 6.0])
    parser.add_argument("--goal-quat", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])
    parser.add_argument("--bounds-x", type=float, nargs=2, default=[0.0, 3.0])
    parser.add_argument("--bounds-y", type=float, nargs=2, default=[3.0, 5.5])
    parser.add_argument("--bounds-z", type=float, nargs=2, default=[0.0, 7.0])

    # Payload toggles (kept for parity)
    parser.add_argument("--use-payload", type=strToBool, default=False)
    parser.add_argument("--payload-translation", type=float, nargs=3, default=[-0.45, 0.0, 0.0])
    parser.add_argument("--payload-size", type=float, nargs=3, default=[0.45, 0.45, 0.12])

    # ---------------- Study-case controls ----------------
    parser.add_argument("--num-runs", type=int, default=10,
                        help="How many runs in study-case (each run records all parameters).")

    # Randomization ranges (default: single-point around defaults so you can just run with minimal flags)
    parser.add_argument("--step-min", type=float, default=0.2)
    parser.add_argument("--step-max", type=float, default=0.2)

    parser.add_argument("--bias-min", type=float, default=0.05)
    parser.add_argument("--bias-max", type=float, default=0.05)

    parser.add_argument("--iter-min", type=int, default=100000)
    parser.add_argument("--iter-max", type=int, default=100000)

    # Bi-RRT* neighbor selection defaults = your preferred defaults
    parser.add_argument("--use-k-nearest", type=strToBool, default=False)
    parser.add_argument("--k-nearest", type=int, default=15)

    parser.add_argument("--use-rstar-radius", type=strToBool, default=True)
    parser.add_argument("--gamma-rstar", type=float, default=2.0)

    parser.add_argument("--rewire-radius", type=float, default=1.5)  # used if use-rstar-radius == False
    parser.add_argument("--dimension", type=int, default=6)

    # If you want randomized neighbor mode in study-case, set this in [0,1]; default keeps your exact defaults
    parser.add_argument("--p-k-nearest", type=float, default=-1.0,
                        help="If in [0,1], randomly choose k-nearest with this probability per run; "
                             "default -1 disables random switching (uses fixed flags).")

    parser.add_argument("--seed", type=int, default=0, help="Random seed for study-case (0 = nondet).")

    args = parser.parse_args()

    # Resolve env path relative to THIS script if not absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = args.env
    if not os.path.isabs(env_path):
        env_path = os.path.abspath(os.path.join(script_dir, env_path))
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    # Common environment/state/bounds
    start_pos = np.array(args.start_pos, dtype=float)
    start_q   = np.array(args.start_quat, dtype=float)
    goal_pos  = np.array(args.goal_pos, dtype=float)
    goal_q    = np.array(args.goal_quat, dtype=float)

    boundsX = np.array(args.bounds_x, dtype=float)
    boundsY = np.array(args.bounds_y, dtype=float)
    boundsZ = np.array(args.bounds_z, dtype=float)

    common = (
        env_path,
        start_pos, start_q, goal_pos, goal_q,
        boundsX, boundsY, boundsZ,
        args.use_payload, args.payload_translation, args.payload_size
    )

    # ------------------ STUDY-CASE MODE ------------------
    if args.study_case:
        rng = np.random.default_rng(args.seed if args.seed != 0 else None)

        runs = []
        for _ in range(int(args.num_runs)):
            # #If ranges collapse to a single point (defaults), this just repeats the same params (OK)
            stepSize = float(args.step_size)
            goalBias = float(rng.uniform(args.bias_min, args.bias_max)) if args.bias_min != args.bias_max else float(args.bias_min)
            iterations = int(rng.integers(args.iter_min, args.iter_max + 1)) if args.iter_min != args.iter_max else int(args.iter_min)

            # Neighbor selection for this run:
            if 0.0 <= args.p_k_nearest <= 1.0:
                useKNearest = bool(rng.random() < args.p_k_nearest)
                useRStarRadius = not useKNearest
            else:
                # fixed by flags
                useKNearest = bool(args.use_k_nearest)
                useRStarRadius = bool(args.use_rstar_radius)

            kNearest = int(args.k_nearest)
            gammaRStar = float(args.gamma_rstar)
            rewireRadius = float(args.rewire_radius)

            runs.append(Run(stepSize=stepSize,
                            goalBias=goalBias,
                            iterations=iterations,
                            useKNearest=useKNearest,
                            kNearest=kNearest,
                            useRStarRadius=useRStarRadius,
                            gammaRStar=gammaRStar,
                            rewireRadius=rewireRadius,
                            dimension=args.dimension))

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(partial(execute_run, common_params=common), runs)

        # Save results (paths included)
        output_json = "rrt_results.json"
        output_pkl = "rrt_results.pkl"
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        with open(output_pkl, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {output_json} and {output_pkl}")
        return

    # ------------------ SINGLE RUN (visualization) ------------------
    env = EnvironmentHandler(o3d.io.read_point_cloud(env_path))
    triangleIndex = env.triangleIndex
    triangleVertex = env.triangleVertex

    planner = rrtcxx.RRTPlanner3D(
        triangleVertex,
        triangleIndex,
        np.array(args.payload_translation),
        np.array(args.payload_size),
        bool(args.use_payload),
        int(args.iterations),
        float(args.step_size),
        float(args.goal_bias),
        float(boundsX[0]), float(boundsX[1]),
        float(boundsY[0]), float(boundsY[1]),
        float(boundsZ[0]), float(boundsZ[1])
    )

    # Bi-RRT* defaults (override via flags)
    planner.setUseKNearest(bool(args.use_k_nearest))
    planner.setKNearest(int(args.k_nearest))
    planner.setUseRStarRadius(bool(args.use_rstar_radius))
    planner.setGammaRStar(float(args.gamma_rstar))
    planner.setRewireRadius(float(args.rewire_radius))
    planner.setDimension(int(args.dimension))

    start = rrtcxx.State(start_pos, start_q)
    goal = rrtcxx.State(goal_pos, goal_q)

    t0 = time.time()
    birrt_path, birrt_time = planner.biRRT(start, goal)
    t1 = time.time()
    print(f"Bi-RRT            : wall={t1 - t0:.4f}s, cxx={birrt_time:.4f}s")

    t0 = time.time()
    birrt_star_path, birrt_star_time = planner.biRRTStar(start, goal)
    t1 = time.time()
    print(f"Bi-RRT* (RRT Star): wall={t1 - t0:.4f}s, cxx={birrt_star_time:.4f}s")

    prunedPath = planner.prunePath(birrt_path) if len(birrt_path) > 0 else []
    starPrunedPath = planner.prunePath(birrt_star_path) if len(birrt_star_path) > 0 else []

    # ------------------ PyVista visualization ------------------
    robotMesh = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))

    pv_ = pv.Plotter()
    triangles = np.hstack([np.full((triangleIndex.shape[0], 1), 3), triangleIndex])
    triangles = triangles.astype(np.int32).flatten()
    mesh_vis = pv.PolyData(triangleVertex, triangles)
    pv_.add_mesh(mesh_vis, color='cyan', show_edges=True, opacity=0.5)

    if len(birrt_path) == 0 and len(birrt_star_path) == 0:
        start_cube = robotMesh.copy()
        Ts = np.eye(4); Ts[:3, 3] = start_pos
        start_cube.transform(Ts)
        pv_.add_mesh(start_cube, color='red', show_edges=True)

        goal_cube = robotMesh.copy()
        Tg = np.eye(4); Tg[:3, 3] = goal_pos
        goal_cube.transform(Tg)
        pv_.add_mesh(goal_cube, color='red', show_edges=True)

        allowedEnvironment = pv.Box(bounds=(boundsX[0], boundsX[1], boundsY[0], boundsY[1], boundsZ[0], boundsZ[1]))
        pv_.add_mesh(allowedEnvironment, color='green', show_edges=True, opacity=0.3)

        print("No path found by either planner.")
        pv_.show()
        return

    for s in birrt_path:       _add_pose_mesh(pv_, robotMesh, s, color='blue')   # unpruned Bi-RRT
    for s in prunedPath:       _add_pose_mesh(pv_, robotMesh, s, color='red')    # pruned Bi-RRT
    for s in birrt_star_path:  _add_pose_mesh(pv_, robotMesh, s, color='green')  # unpruned Bi-RRT*
    for s in starPrunedPath:   _add_pose_mesh(pv_, robotMesh, s, color='orange') # pruned Bi-RRT*

    pv_.show_axes()
    pv_.show_grid()
    pv_.show()

    # Save paths for single run too (next to env)
    out_npz = os.path.join(os.path.dirname(env_path), "path.npz")
    if len(prunedPath) > 0:
        positions = np.array([p.position for p in prunedPath])
        orientations = np.array([p.q for p in prunedPath])
    else:
        positions = np.empty((0, 3)); orientations = np.empty((0, 4))

    if len(birrt_path) > 0:
        unprunedPositions = np.array([p.position for p in birrt_path])
        unprunedOrientations = np.array([p.q for p in birrt_path])
    else:
        unprunedPositions = np.empty((0, 3)); unprunedOrientations = np.empty((0, 4))

    np.savez(out_npz,
             positions=positions,
             orientations=orientations,
             unprunedPositions=unprunedPositions,
             unprunedOrientations=unprunedOrientations)

    print(f"Saved NPZ to {out_npz}")
    print(f"Bi-RRT: path={len(birrt_path)}, pruned={len(prunedPath)}")
    print(f"Bi-RRT*: path={len(birrt_star_path)}, pruned={len(starPrunedPath)}")


if __name__ == "__main__":
    main()
