# study_birrt.py
import os
import time
import json
import pickle
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pyvista as pv
import open3d as o3d
from scipy.spatial.transform import Rotation as R  # noqa: F401  (kept for quick local tests)

from Environment import EnvironmentHandler
from Map import Map  # Use your Map class that loads PCD by path
import rrtcxx  # pybind11 module exposing RRTPlanner3D


# ================================================
# Utilities
# ================================================
def state_list_to_dict(path):
    """Convert list[rrtcxx.State] to JSON-serializable list of dicts."""
    return [
        {
            "position": [float(v) for v in s.position],
            "orientation": [float(v) for v in s.q],  # [x, y, z, w]
        }
        for s in path
    ]


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def _exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def find_repo_root(start: str) -> str:
    """Walk upward until we find a folder named 'src' and return its parent."""
    p = os.path.abspath(start)
    while True:
        if os.path.basename(p) == "src":
            return os.path.dirname(p)  # repo root
        up = os.path.dirname(p)
        if up == p:  # reached filesystem root
            break
        p = up
    # Fallback: two levels above script
    return os.path.dirname(os.path.dirname(start))


# ================================================
# Per-run parameter carrier
# ================================================
class Run:
    def __init__(
        self,
        *,
        # Shared params (Bi-RRT & Bi-RRT*)
        iterations: int,
        stepSize: float,
        goalBias: float,
        # Bi-RRT* neighbor logic
        useKNearest: bool,
        kNearest: int,
        useRStarRadius: bool,
        gammaRStar: float,
        rewireRadius: float,
        dimension: int,
    ):
        self.iterations = int(iterations)
        self.stepSize = float(stepSize)
        self.goalBias = float(goalBias)

        self.useKNearest = bool(useKNearest)
        self.kNearest = int(kNearest)

        self.useRStarRadius = bool(useRStarRadius)
        self.gammaRStar = float(gammaRStar)
        self.rewireRadius = float(rewireRadius)

        self.dimension = int(dimension)


# ================================================
# One execution (both planners) for a given map
# ================================================
def execute_run(run: Run, common_params):
    """
    Run both Bi-RRT and Bi-RRT* with the same environment/start/goal/bounds,
    but with per-run parameters (including RRT* neighbor logic).
    """
    (
        pcd_path,
        start_pos, start_q,
        goal_pos, goal_q,
        boundsX, boundsY, boundsZ,
        use_payload,
        payload_translation, payload_size,
    ) = common_params

    # Build Map + Environment from PCD path and metadata
    map_obj = Map(
        pointCloudPath=pcd_path,
        startState=np.array([*start_pos, 0, 0, 0, *start_q, 0, 0, 0], dtype=float),
        endState=np.array([*goal_pos, 0, 0, 0, *goal_q, 0, 0, 0], dtype=float),
        boundsMin=np.array([boundsX[0], boundsY[0], boundsZ[0]], dtype=float),
        boundsMax=np.array([boundsX[1], boundsY[1], boundsZ[1]], dtype=float),
    )
    env = EnvironmentHandler.from_map(map_obj)
    triangle_vertex = env.triangleVertex
    triangle_index = env.triangleIndex

    planner = rrtcxx.RRTPlanner3D(
        triangle_vertex,
        triangle_index,
        np.array(payload_translation, dtype=float),
        np.array(payload_size, dtype=float),
        bool(use_payload),
        int(run.iterations),
        float(run.stepSize),
        float(run.goalBias),
        float(boundsX[0]), float(boundsX[1]),
        float(boundsY[0]), float(boundsY[1]),
        float(boundsZ[0]), float(boundsZ[1]),
    )

    # Bi-RRT* neighbor settings
    planner.setUseKNearest(bool(run.useKNearest))
    planner.setKNearest(int(run.kNearest))
    planner.setUseRStarRadius(bool(run.useRStarRadius))
    planner.setGammaRStar(float(run.gammaRStar))
    planner.setRewireRadius(float(run.rewireRadius))   # used only if useRStarRadius == False
    planner.setDimension(int(run.dimension))

    start_state = rrtcxx.State(np.array(start_pos, dtype=float), np.array(start_q, dtype=float))
    goal_state = rrtcxx.State(np.array(goal_pos, dtype=float), np.array(goal_q, dtype=float))

    # --- Bi-RRT ---
    birrt_path, birrt_time = planner.biRRT(start_state, goal_state)
    birrt_success = len(birrt_path) > 0
    birrt_pruned = planner.prunePath(birrt_path) if birrt_success else []

    # --- Bi-RRT* ---
    birrt_star_path, birrt_star_time = planner.biRRTStar(start_state, goal_state)
    birrt_star_success = len(birrt_star_path) > 0
    birrt_star_pruned = planner.prunePath(birrt_star_path) if birrt_star_success else []

    return {
        "params": {
            # Shared
            "iterations": run.iterations,
            "stepSize": run.stepSize,
            "goalBias": run.goalBias,
            # RRT* neighbor logic actually used in this run
            "useKNearest": run.useKNearest,
            "kNearest": run.kNearest,
            "useRStarRadius": run.useRStarRadius,
            "gammaRStar": run.gammaRStar,
            "rewireRadius": run.rewireRadius,
            "dimension": run.dimension,
            # Map setup (for reproducibility)
            "env": os.path.basename(pcd_path),
            "boundsX": [float(boundsX[0]), float(boundsX[1])],
            "boundsY": [float(boundsY[0]), float(boundsY[1])],
            "boundsZ": [float(boundsZ[0]), float(boundsZ[1])],
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
            "timeSec": float(birrt_time),
            "pathLength": int(len(birrt_path) if birrt_success else 0),
            "prunedPathLength": int(len(birrt_pruned) if birrt_success else 0),
            "path": state_list_to_dict(birrt_path),
            "prunedPath": state_list_to_dict(birrt_pruned),
        },
        "BiRRTStar": {
            "success": birrt_star_success,
            "timeSec": float(birrt_star_time),
            "pathLength": int(len(birrt_star_path) if birrt_star_success else 0),
            "prunedPathLength": int(len(birrt_star_pruned) if birrt_star_success else 0),
            "path": state_list_to_dict(birrt_star_path),
            "prunedPath": state_list_to_dict(birrt_star_pruned),
        },
    }


def _job(args_tuple):
    """Wrapper so we can pool over different env commons."""
    run, common = args_tuple
    return execute_run(run, common)


# ================================================
# Study logic (MAPS VIA Map .pkl METADATA)
# ================================================
def run_study(args):
    """
    Study-case:
      - Runs the SAME 3 maps (simple/middle/complex) loaded from Map PKLs in <repo_root>/Maps.
      - Runs each map args.num_runs times.
      - Randomizes parameters within user-given ranges for BOTH Bi-RRT and Bi-RRT* (where applicable).
      - Saves full parameters and paths to rrt_results.json / .pkl
    """
    rng = np.random.default_rng(args.seed if args.seed != 0 else None)

    # Resolve Maps directory (sibling to src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = find_repo_root(script_dir)
    maps_dir = os.path.join(repo_root, "Maps")

    # Helper to load map metadata from PKL
    def load_map_meta(pkl_name: str):
        pkl_path = pkl_name if os.path.isabs(pkl_name) else os.path.join(maps_dir, pkl_name)
        _exists(pkl_path)
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        pcd_path = meta["point_cloud_path"]
        if not os.path.isabs(pcd_path):
            pcd_path = os.path.join(maps_dir, pcd_path)
        _exists(pcd_path)
        start_state = meta["start_state"]
        end_state = meta["end_state"]
        bounds_min = meta["bounds_min"]
        bounds_max = meta["bounds_max"]
        # Build bounds arrays per axis
        boundsX = np.array([bounds_min[0], bounds_max[0]], dtype=float)
        boundsY = np.array([bounds_min[1], bounds_max[1]], dtype=float)
        boundsZ = np.array([bounds_min[2], bounds_max[2]], dtype=float)
        start_pos = start_state[0:3]
        start_q = start_state[6:10]
        goal_pos = end_state[0:3]
        goal_q = end_state[6:10]
        return {
            "pcd_path": pcd_path,
            "boundsX": boundsX,
            "boundsY": boundsY,
            "boundsZ": boundsZ,
            "start_pos": start_pos,
            "start_q": start_q,
            "goal_pos": goal_pos,
            "goal_q": goal_q,
        }

    # Load the three study maps from PKLs
    simple = load_map_meta("simpleMap.pkl")
    middle = load_map_meta("middleMap.pkl")
    complex_map = load_map_meta("complexMap.pkl")

    study_maps = [simple, middle, complex_map]

    # ---- Build jobs: args.num_runs per map ----
    jobs = []
    for m in study_maps:
        common = (
            m["pcd_path"],
            m["start_pos"], m["start_q"],
            m["goal_pos"], m["goal_q"],
            m["boundsX"], m["boundsY"], m["boundsZ"],
            args.use_payload, args.payload_translation, args.payload_size,
        )

        for _ in range(int(args.num_runs)):
            # --- Shared parameters (Bi-RRT + Bi-RRT*)
            stepSize = float(rng.uniform(args.step_min, args.step_max))

            goalBias = float(args.bias_min) if args.bias_min == args.bias_max else float(
                rng.uniform(args.bias_min, args.bias_max)
            )

            iterations = int(args.iter_min) if args.iter_min == args.iter_max else int(
                rng.integers(args.iter_min, args.iter_max + 1)
            )

            # --- RRT* neighbor logic variation ---
            if 0.0 <= args.p_k_nearest <= 1.0:
                useKNearest = bool(rng.random() < args.p_k_nearest)
                useRStarRadius = not useKNearest
            else:
                useKNearest = bool(args.use_k_nearest)
                useRStarRadius = bool(args.use_rstar_radius)

            kNearest = int(args.k_min) if args.k_min == args.k_max else int(
                rng.integers(args.k_min, args.k_max + 1)
            )

            gammaRStar = float(args.gamma_min) if args.gamma_min == args.gamma_max else float(
                rng.uniform(args.gamma_min, args.gamma_max)
            )

            rewireRadius = float(args.rewire_min) if args.rewire_min == args.rewire_max else float(
                rng.uniform(args.rewire_min, args.rewire_max)
            )

            dimension = int(args.dim_min) if args.dim_min == args.dim_max else int(
                rng.integers(args.dim_min, args.dim_max + 1)
            )

            run = Run(
                iterations=iterations,
                stepSize=stepSize,
                goalBias=goalBias,
                useKNearest=useKNearest,
                kNearest=kNearest,
                useRStarRadius=useRStarRadius,
                gammaRStar=gammaRStar,
                rewireRadius=rewireRadius,
                dimension=dimension,
            )
            jobs.append((run, common))

    # ---- Execute in parallel ----
    with Pool(processes=int(cpu_count() / 2)) as pool:
        results = pool.map(_job, jobs)

    # ---- Save ----
    with open("rrt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("rrt_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Results saved to rrt_results.json and rrt_results.pkl")


# ================================================
# Optional single-run visualization on complex map
# ================================================
def single_run_demo(args):
    # Resolve Maps directory (sibling to src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = find_repo_root(script_dir)
    maps_dir = os.path.join(repo_root, "Maps")

    # Load complex map from PKL metadata
    complex_pkl = os.path.join(maps_dir, "complexMap.pkl")
    _exists(complex_pkl)
    with open(complex_pkl, "rb") as f:
        meta = pickle.load(f)

    pcd_path = meta["point_cloud_path"]
    if not os.path.isabs(pcd_path):
        pcd_path = os.path.join(maps_dir, pcd_path)
    _exists(pcd_path)

    start_state = meta["start_state"]
    end_state = meta["end_state"]
    bounds_min = meta["bounds_min"]
    bounds_max = meta["bounds_max"]

    start_pos = start_state[0:3].astype(float)
    start_q   = start_state[6:10].astype(float)
    goal_pos  = end_state[0:3].astype(float)
    goal_q    = end_state[6:10].astype(float)
    boundsX   = np.array([bounds_min[0], bounds_max[0]], dtype=float)
    boundsY   = np.array([bounds_min[1], bounds_max[1]], dtype=float)
    boundsZ   = np.array([bounds_min[2], bounds_max[2]], dtype=float)

    # Build env via Map class
    map_obj = Map(
        pointCloudPath=pcd_path,
        startState=start_state.copy(),
        endState=end_state.copy(),
        boundsMin=bounds_min.copy(),
        boundsMax=bounds_max.copy(),
    )
    env = EnvironmentHandler.from_map(map_obj)
    tri_idx = env.triangleIndex
    tri_vtx = env.triangleVertex

    planner = rrtcxx.RRTPlanner3D(
        tri_vtx, tri_idx,
        np.array(args.payload_translation, dtype=float),
        np.array(args.payload_size, dtype=float),
        bool(args.use_payload),
        int(args.iterations),
        float(args.step_size),
        float(args.goal_bias),
        float(boundsX[0]), float(boundsX[1]),
        float(boundsY[0]), float(boundsY[1]),
        float(boundsZ[0]), float(boundsZ[1]),
    )

    # RRT* settings (fixed for single demo)
    planner.setUseKNearest(bool(args.use_k_nearest))
    planner.setKNearest(int(args.p_k_nearest))
    planner.setUseRStarRadius(bool(args.use_rstar_radius))
    planner.setDimension(int(args.dimension))

    start = rrtcxx.State(start_pos, start_q)
    goal = rrtcxx.State(goal_pos, goal_q)

    t0 = time.time()
    birrt_path, birrt_time = planner.biRRT(start, goal)
    t1 = time.time()
    print(f"Bi-RRT            : wall={t1 - t0:.4f}s, cxx={birrt_time:.4f}s")

    t0 = time.time()
    #star_path, star_time = planner.biRRTStar(start, goal)
    t1 = time.time()
    #print(f"Bi-RRT* (RRT Star): wall={t1 - t0:.4f}s, cxx={star_time:.4f}s")

    pruned = planner.prunePath(birrt_path) if birrt_path else []
    #star_pruned = planner.prunePath(star_path) if star_path else []

    # --- Visualization ---
    robot_mesh = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))
    pv_ = pv.Plotter()
    faces = np.hstack([np.full((tri_idx.shape[0], 1), 3), tri_idx]).astype(np.int32).ravel()
    mesh_vis = pv.PolyData(tri_vtx, faces)
    pv_.add_mesh(mesh_vis, color="cyan", show_edges=True, opacity=0.5)

    def _add_pose_mesh(pv_plotter, robot_mesh, state, color):
        x = state.position
        q = state.q
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = x
        cube = robot_mesh.copy()
        cube.transform(T)
        pv_plotter.add_mesh(cube, color=color, show_edges=True)

    for s in birrt_path:   _add_pose_mesh(pv_, robot_mesh, s, "blue")
    for s in pruned:       _add_pose_mesh(pv_, robot_mesh, s, "red")
   # for s in star_path:    _add_pose_mesh(pv_, robot_mesh, s, "yellow")
   # for s in star_pruned:  _add_pose_mesh(pv_, robot_mesh, s, "orange")

    pv_.show_axes()
    pv_.show_grid()
    pv_.show()

    # --- Save ONLY Bi-RRT pruned path to NPZ (no Bi-RRT* content) ---
    npz_path = os.path.join(maps_dir, "path.npz")
    positions = np.array([p.position for p in pruned]) if pruned else np.empty((0, 3))
    orientations = np.array([p.q for p in pruned]) if pruned else np.empty((0, 4))
    np.savez(npz_path, positions=positions, orientations=orientations)
    print(f"Saved Bi-RRT (pruned) path to {npz_path}")


# ================================================
# Main / Args
# ================================================
def main():
    def strToBool(v):
        if isinstance(v, bool):
            return v
        v = str(v).lower()
        if v in ("yes", "true", "t", "y", "1"):
            return True
        if v in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    p = argparse.ArgumentParser(description="3-map study: Bi-RRT vs Bi-RRT* with parameter variation, using Map PKLs.")

    # Mode
    p.add_argument("--study-case", type=strToBool, default=False,
                   help="Run the 3 hard-coded maps (via Map PKLs) with parameter variation.")
    p.add_argument("--single-run", type=strToBool, default=False,
                   help="Optional: visualize a single run on the complex map.")

    # Counts
    p.add_argument("--num-runs", type=int, default=1000,
                   help="Number of runs per map in study-case.")

    # Shared parameter ranges (affect both Bi-RRT and Bi-RRT*)
    p.add_argument("--step-min", type=float, default=0.05,
                   help="Min step size (old max_distance).")
    p.add_argument("--step-max", type=float, default=0.1,
                   help="Max step size (set equal to keep fixed).")
    p.add_argument("--bias-min", type=float, default=0.05,
                   help="Min goal bias.")
    p.add_argument("--bias-max", type=float, default=0.05,
                   help="Max goal bias.")
    p.add_argument("--iter-min", type=int, default=100000,
                   help="Min iterations.")
    p.add_argument("--iter-max", type=int, default=100000,
                   help="Max iterations.")

    # Bi-RRT* neighbor selection controls
    p.add_argument("--p-k-nearest", type=float, default=-1.0,
                   help="Probability to choose k-nearest per run (in [0,1]). -1 keeps fixed mode.")
    p.add_argument("--use-k-nearest", type=strToBool, default=False,
                   help="If p-k-nearest is -1, force k-nearest mode (True) or radius mode (False).")
    p.add_argument("--use-rstar-radius", type=strToBool, default=True,
                   help="If p-k-nearest is -1 and use-k-nearest=False, use R* radius (True) vs fixed radius (False).")

    # k-nearest range
    p.add_argument("--k-min", type=int, default=15)
    p.add_argument("--k-max", type=int, default=15)

    # R* gamma range
    p.add_argument("--gamma-min", type=float, default=2.0)
    p.add_argument("--gamma-max", type=float, default=2.0)

    # Fixed-radius range (used when NOT using R* radius)
    p.add_argument("--rewire-min", type=float, default=1.5)
    p.add_argument("--rewire-max", type=float, default=1.5)

    # Dimensionality (usually 6 for SE(3))
    p.add_argument("--dim-min", type=int, default=6)
    p.add_argument("--dim-max", type=int, default=6)

    # Payload (kept for parity)
    p.add_argument("--use-payload", type=strToBool, default=False)
    p.add_argument("--payload-translation", type=float, nargs=3, default=[-0.45, 0.0, 0.0])
    p.add_argument("--payload-size", type=float, nargs=3, default=[0.45, 0.45, 0.12])

    # Defaults for single-run path (when not studying)
    p.add_argument("--iterations", type=int, default=100000)
    p.add_argument("--step-size", type=float, default=0.05)
    p.add_argument("--goal-bias", type=float, default=0.05)
    p.add_argument("--dimension", type=int, default=6)  # single-run RRT* dimension

    p.add_argument("--seed", type=int, default=0, help="Random seed for study-case (0 = nondeterministic).")

    args = p.parse_args()

    if args.study_case:
        run_study(args)
        return

    if args.single_run:
        single_run_demo(args)
        return

    print("Nothing to do. Use --study-case true to run the 3-map study, or --single-run true for a quick visualization.")


if __name__ == "__main__":
    main()
