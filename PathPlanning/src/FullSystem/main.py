#!/usr/bin/env python3
"""
Bi-RRT (single run, pruned) -> Iterative Global Optimization (indefinite).
Visualizes EVERY optimization iteration (blocking window). No multiprocessing/threads.

Flow:
  1) Load map PKL (start/end/bounds/PCD).
  2) Build Environment from Map.
  3) Run Bi-RRT once (C++ rrtcxx), prune the path.
  4) Convert pruned Bi-RRT path -> OptimizationState list (force endpoints to map start/end).
  5) Indefinitely run GlobalOptimalPlanner.optimize() each iteration, accumulate collision hints, visualize.
  6) On Ctrl+C, save the latest state (path + log) to a pickle.

Requirements in repo:
  - Map PKL keys: start_state, end_state, bounds_min, bounds_max, point_cloud_path
  - Robot params in this script folder: A_matrix.npy, J_matrix.npy, mass.npy
  - Python modules available in path: EnvironmentHandler, Map, Robot, Obstacle, OptimizationState, GlobalOptimalPlanner
  - C++ pybind11 module: rrtcxx (with RRTPlanner3D, State, prunePath, etc.)
"""

import os
import sys
import time
import pickle
import argparse
from typing import List, Optional

import numpy as np
import pyvista as pv
import scipy.spatial.transform as trf

# Add this script's folder to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from Environment import EnvironmentHandler
from Map import Map
from RRTOptimization.Robot import Robot
from RRTOptimization.Obstacle import Obstacle
from RRTOptimization.OptimizationState import OptimizationState
from RRTOptimization.GlobalOptimalPlanner import GlobalOptimalPlanner
import rrtcxx  # pybind11 module exposing RRTPlanner3D


# ------------------------------- Utils -------------------------------

def find_repo_root(start: str) -> str:
    """Walk upward until we find a folder named 'src' and return its parent."""
    p = os.path.abspath(start)
    while True:
        if os.path.basename(p) == "src":
            return os.path.dirname(p)
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
    # Fallback: two levels above script
    return os.path.dirname(os.path.dirname(start))


def polyline(points: np.ndarray) -> pv.PolyData:
    """Create a PolyData polyline from (N,3) points for PyVista plotting."""
    n = points.shape[0]
    lines = np.hstack([[n], np.arange(n)]).astype(np.int64)
    pd = pv.PolyData(points)
    pd.lines = lines
    return pd


def visualize(environment,
              robot: Robot,
              base_path: Optional[List[OptimizationState]],
              path_after: List[OptimizationState],
              obstacles: Optional[List[Obstacle]] = None,
              collisions: Optional[List[Obstacle]] = None,
              title: str = "",
              boundsMax = None,
              boundsMin = None) -> None:
    """Per-iteration visualization (blocking window). Close it to proceed."""
    pl = pv.Plotter(title=title)

    # Map/Environment
    if hasattr(environment, "visualizeCoalMesh"):
        try:
            pl = environment.visualizeCoalMesh(pl)
        except Exception:
            pl = environment.visualizeMap(pl)
    else:
        pl = environment.visualizeMap(pl)

    # Base path (gray)
    if base_path:
        base_pts = np.asarray([p.x for p in base_path])
        if base_pts.shape[0] >= 2:
            pl.add_mesh(polyline(base_pts), line_width=2, color="gray")
        for p in base_path:
            pl.add_mesh(
                robot.getPVMesh(p.x, trf.Rotation.from_quat(p.q)),
                color="gray", show_edges=False, opacity=0.15
            )

    # Current path (blue)
    cur_pts = np.asarray([p.x for p in path_after])
    if cur_pts.shape[0] >= 2:
        pl.add_mesh(polyline(cur_pts), line_width=3, color="blue")
    for p in path_after:
        pl.add_mesh(
            robot.getPVMesh(p.x, trf.Rotation.from_quat(p.q)),
            color="blue", show_edges=True, opacity=0.45
        )

    # Start/End emphasized
    pl.add_mesh(
        robot.getPVMesh(path_after[0].x, trf.Rotation.from_quat(path_after[0].q)),
        color="green", show_edges=True, opacity=0.9
    )
    pl.add_mesh(
        robot.getPVMesh(path_after[-1].x, trf.Rotation.from_quat(path_after[-1].q)),
        color="red", show_edges=True, opacity=0.9
    )

    # Obstacles (yellow) and accumulated collisions (red)
    if obstacles:
        for o in obstacles:
            pl.add_mesh(pv.Plane(center=o.closestPointObstacle, direction=o.normal, i_size=0.5, j_size=0.5),
                        color="yellow", opacity=0.4)
            pl.add_mesh(pv.Arrow(start=o.closestPointObstacle, direction=o.normal, scale=0.1),
                        color="yellow")
    if collisions:
        for c in collisions:
            pl.add_mesh(pv.Plane(center=c.closestPointObstacle, direction=c.normal, i_size=0.5, j_size=0.5),
                        color="red", opacity=0.5)
            pl.add_mesh(pv.Arrow(start=c.closestPointObstacle, direction=c.normal, scale=0.1),
                        color="red")

    if not boundsMin is None and not boundsMax is None:
        box = pv.Box(bounds=(boundsMin[0], boundsMax[0],
                             boundsMin[1], boundsMax[1],
                             boundsMin[2], boundsMax[2]))
        pl.add_mesh(box, color="green", opacity=0.1)

    pl.show_axes()
    pl.show_grid()
    pl.add_axes_at_origin()
    pl.show()  # blocking window


# ------------------------------- Main -------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Bi-RRT (single, pruned) -> Indefinite Global Optimization with per-iteration visualization."
    )

    # Map / environment
    p.add_argument("--map", "-m", type=str, default="simpleMap.pkl",
                   help="Map PKL (resolved under <repo_root>/Maps if relative).")
    p.add_argument("--voxel-size", type=float, default=0.1,
                   help="Voxel size for environment.")

    # Bi-RRT parameters (single run)
    p.add_argument("--rrt-iterations", type=int, default=100000,
                   help="Bi-RRT iterations.")
    p.add_argument("--rrt-step", type=float, default=0.05,
                   help="Bi-RRT step size (max extension distance).")
    p.add_argument("--rrt-goal-bias", type=float, default=0.05,
                   help="Bi-RRT goal bias in [0,1].")

    # Payload options (passed to planner; keep parity with your C++ signature)
    p.add_argument("--use-payload", action="store_true", default=False)
    p.add_argument("--payload-translation", type=float, nargs=3, default=[-0.45, 0.0, 0.0])
    p.add_argument("--payload-size", type=float, nargs=3, default=[0.45, 0.45, 0.12])

    # Save latest on Ctrl+C
    p.add_argument("--save", type=str, default="birrt_opt_results.pkl",
                   help="Pickle file to save latest path + log on Ctrl+C.")

    args = p.parse_args()

    # Resolve repo root and Maps dir
    repo_root = find_repo_root(script_dir)
    maps_dir = os.path.join(repo_root, "Maps")

    # Resolve map PKL
    map_pkl = args.map if os.path.isabs(args.map) else os.path.join(maps_dir, args.map)
    if not os.path.exists(map_pkl):
        raise FileNotFoundError(f"Map PKL not found: {map_pkl}")
    with open(map_pkl, "rb") as f:
        meta = pickle.load(f)

    # Extract map data
    pcd_path = meta["point_cloud_path"]
    if not os.path.isabs(pcd_path):
        pcd_path = os.path.join(maps_dir, pcd_path)
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

    start_state = meta["start_state"]
    end_state   = meta["end_state"]
    bounds_min  = meta["bounds_min"]
    bounds_max  = meta["bounds_max"]

    print("Map metadata loaded successfully:")
    print(f" - Point Cloud Path: {pcd_path}")
    print(f" - Start State: {start_state}")
    print(f" - End State: {end_state}")
    print(f" - Bounds: {bounds_min} to {bounds_max}")

    start_pos = start_state[0:3].astype(float)
    start_q   = start_state[6:10].astype(float)  # [x y z w]
    goal_pos  = end_state[0:3].astype(float)
    goal_q    = end_state[6:10].astype(float)

    boundsX = np.array([bounds_min[0], bounds_max[0]], dtype=float)
    boundsY = np.array([bounds_min[1], bounds_max[1]], dtype=float)
    boundsZ = np.array([bounds_min[2], bounds_max[2]], dtype=float)

    # Build Map + Environment
    map_obj = Map(
        pointCloudPath=pcd_path,
        startState=start_state.copy(),
        endState=end_state.copy(),
        boundsMin=bounds_min.copy(),
        boundsMax=bounds_max.copy(),
    )
    environment = EnvironmentHandler.from_map(map_obj, voxel_size=args.voxel_size)

    # ------------------ Single-run Bi-RRT (with pruning) ------------------
    tri_vtx = environment.triangleVertex
    tri_idx = environment.triangleIndex

    planner = rrtcxx.RRTPlanner3D(
        tri_vtx,
        tri_idx,
        np.array(args.payload_translation, dtype=float),
        np.array(args.payload_size, dtype=float),
        bool(args.use_payload),
        int(args.rrt_iterations),
        float(args.rrt_step),
        float(args.rrt_goal_bias),
        float(boundsX[0]), float(boundsX[1]),
        float(boundsY[0]), float(boundsY[1]),
        float(boundsZ[0]), float(boundsZ[1]),
    )

    start = rrtcxx.State(start_pos, start_q)
    goal  = rrtcxx.State(goal_pos, goal_q)

    t0 = time.time()
    raw_path, cxx_time = planner.biRRT(start, goal)
    t1 = time.time()
    print(f"[Bi-RRT] wall={t1 - t0:.3f}s, cxx={cxx_time:.3f}s, states={len(raw_path) if raw_path else 0}")

    pruned = planner.prunePath(raw_path) if raw_path else []
    if not pruned or len(pruned) < 2:
        raise RuntimeError("Bi-RRT failed or produced insufficient path. Cannot start optimization.")

    # ------------------ Convert to OptimizationState list ------------------
    init_path: List[OptimizationState] = [
        OptimizationState(np.asarray(s.position, dtype=float),
                          np.asarray(s.q, dtype=float)) for s in pruned
    ]
    # Force endpoints to map start/end states (xyzw quats)
    init_path[0]  = OptimizationState(start_state[0:3], start_state[6:10])
    init_path[-1] = OptimizationState(end_state[0:3],  end_state[6:10])

    # ------------------ Robot & initial v/w ------------------
    A = np.load(os.path.join(script_dir, "A_matrix.npy"))
    J = np.load(os.path.join(script_dir, "J_matrix.npy"))
    m = np.load(os.path.join(script_dir, "mass.npy"))
    robot = Robot(J, A, m)

    minV = np.array([-5, -5, -5]);  maxV = -minV
    minW = np.array([-4, -4, -4]);  maxW = -minW
    dt   = 0.2
    for i in range(len(init_path) - 1):
        v = (init_path[i + 1].x - init_path[i].x) / dt
        init_path[i].v = np.clip(v, minV, maxV)
        w = (1.0 / dt) * (
            trf.Rotation.from_quat(init_path[i + 1].q) *
            trf.Rotation.from_quat(init_path[i].q).inv()
        ).as_rotvec()
        init_path[i].w = np.clip(w, minW, maxW)

    # ------------------ State bounds & planner ------------------
    state_lower = np.hstack([bounds_min, minV, np.array([-1, -1, -1, -1]), minW])
    state_upper = np.hstack([bounds_max, maxV, np.array([ 1,  1,  1,  1]), maxW])

    gplanner = GlobalOptimalPlanner(state_lower, state_upper, environment, robot)

    # ------------------ Indefinite Global Optimization Loop ------------------
    all_iters = []
    collisions_accum: List[Obstacle] = []
    base_path = init_path
    path = init_path

    print("Starting indefinite global optimization from Bi-RRT path.")
    print("Close the window each iteration to proceed. Press Ctrl+C to stop and save.")
    try:
        it = 0
        while True:
            it += 1
            obstacles, maxDistances = robot.getObstacles(environment, path)
            obstacles = list(obstacles) + list(collisions_accum)

            t0 = time.time()
            try:
                new_path, new_dt = gplanner.optimize(
                    path,
                    obstacles,
                    maxDistances,
                    dt,
                    path[0],
                    path[-1],
                )
            except Exception as e:
                print(f"[Iter {it}] Optimization error: {e}")
                break
            t1 = time.time()

            # Preserve endpoints and update dt
            new_path[0]  = path[0]
            new_path[-1] = path[-1]
            dt = new_dt

            ok = robot.collisionFree(new_path, environment)
            print(f"[Iter {it}] time={t1 - t0:.3f}s, dt={dt:.3f}, collision_free={ok}")

            all_iters.append({
                "iter": it,
                "time": float(t1 - t0),
                "dt": float(dt),
                "collision_free": bool(ok),
            })

            # Visualize EVERY iteration (blocking)
            visualize(environment, robot, base_path, new_path,
                      obstacles=obstacles, collisions=collisions_accum,
                      title=f"Iteration {it}: {'OK' if ok else 'COLLISIONS'}", 
                      boundsMin=bounds_min, boundsMax=bounds_max)

            # Accumulate collisions and continue regardless of 'ok'
            new_collisions = robot.getCollision(environment, new_path)
            collisions_accum.extend(new_collisions)

            path = new_path  # proceed from latest

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving latest results...")

    # ------------------ Save latest on exit ------------------
    out = {
        "latestPath": path,
        "iterations": all_iters,
        "collisions": collisions_accum,
        "dt": dt,
        "map": args.map,
    }
    with open(os.path.join(script_dir, args.save), "wb") as f:
        pickle.dump(out, f)
    print(f"[DONE] Saved results to {os.path.join(script_dir, args.save)}")


if __name__ == "__main__":
    main()
