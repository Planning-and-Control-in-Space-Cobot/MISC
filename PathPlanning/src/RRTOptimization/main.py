#!/usr/bin/env python3
"""
Iterative Global Optimization — runs indefinitely, visualizes EVERY iteration.
No multiprocessing/threads.

Flow:
  1) Load map metadata (PKL) + initial path (NPZ).
  2) Build Environment + Robot.
  3) Forever: GlobalOptimalPlanner.optimize() -> update path, accumulate collisions, visualize each step.
  4) On Ctrl+C, save the latest path + iteration log to disk.

Requires:
  - Map PKL keys: start_state, end_state, bounds_min, bounds_max, point_cloud_path
  - Path NPZ arrays: positions (Nx3), orientations (Nx4 [xyzw])
  - Robot params in this script folder: A_matrix.npy, J_matrix.npy, mass.npy
"""

import os
import sys
import time
import argparse
import pickle
from typing import List, Optional

import numpy as np
import pyvista as pv
import scipy.spatial.transform as trf

# Add this script's folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from Environment import EnvironmentHandler
from Map import Map
from Robot import Robot
from Obstacle import Obstacle
from OptimizationState import OptimizationState
from GlobalOptimalPlanner import GlobalOptimalPlanner


# ------------------------------- Utils -------------------------------

def find_repo_root(start: str) -> str:
    """Walk up until we find a folder named 'src' and return its parent (repo root)."""
    p = os.path.abspath(start)
    while True:
        if os.path.basename(p) == "src":
            return os.path.dirname(p)
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
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
              title: str = ""):
    """Per-iteration visualization (blocking window). Close it to proceed to next iteration."""
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
            pl.add_mesh(robot.getPVMesh(p.x, trf.Rotation.from_quat(p.q)),
                        color="gray", show_edges=False, opacity=0.15)

    # Current path (blue)
    cur_pts = np.asarray([p.x for p in path_after])
    if cur_pts.shape[0] >= 2:
        pl.add_mesh(polyline(cur_pts), line_width=3, color="blue")
    for p in path_after:
        pl.add_mesh(robot.getPVMesh(p.x, trf.Rotation.from_quat(p.q)),
                    color="blue", show_edges=True, opacity=0.45)

    # Start/End emphasized
    pl.add_mesh(robot.getPVMesh(path_after[0].x,  trf.Rotation.from_quat(path_after[0].q)),
                color="green", show_edges=True, opacity=0.9)
    pl.add_mesh(robot.getPVMesh(path_after[-1].x, trf.Rotation.from_quat(path_after[-1].q)),
                color="red",   show_edges=True, opacity=0.9)

    # Obstacles (yellow) & accumulated collisions (red)
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

    pl.show_axes()
    pl.show_grid()
    pl.add_axes_at_origin()
    pl.show()  # blocking


# ------------------------------- Main -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Iterative Global Optimization (visualize every step; runs indefinitely).")
    parser.add_argument("--path", "-p", type=str, default="path.npz",
                        help="Path to NPZ with arrays positions (Nx3) and orientations (Nx4 [xyzw]).")
    parser.add_argument("--map", "-m", type=str, default="simpleMap.pkl",
                        help="Map pickle (under <repo_root>/Maps if relative).")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size for environment.")
    parser.add_argument("--save", type=str, default="optimized_results.pkl",
                        help="Pickle to save latest path + history on Ctrl+C.")
    args = parser.parse_args()

    # Resolve repo root / maps
    repo_root = find_repo_root(script_dir)
    maps_dir = os.path.join(repo_root, "Maps")

    # Resolve files
    path_file = args.path if os.path.isabs(args.path) else os.path.join(script_dir, args.path)
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Path NPZ not found: {path_file}")

    map_pkl = args.map if os.path.isabs(args.map) else os.path.join(maps_dir, args.map)
    if not os.path.exists(map_pkl):
        raise FileNotFoundError(f"Map PKL not found: {map_pkl} (looked under {maps_dir}).")

    # Load map metadata
    with open(map_pkl, "rb") as f:
        meta = pickle.load(f)
    start_state = meta["start_state"]
    end_state   = meta["end_state"]
    bounds_min  = meta["bounds_min"]
    bounds_max  = meta["bounds_max"]
    pcd_path    = meta["point_cloud_path"]
    if not os.path.isabs(pcd_path):
        pcd_path = os.path.join(maps_dir, pcd_path)
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

    # Map + Environment
    map_obj = Map(
        pointCloudPath=pcd_path,
        startState=start_state.copy(),
        endState=end_state.copy(),
        boundsMin=bounds_min.copy(),
        boundsMax=bounds_max.copy(),
    )
    environment = EnvironmentHandler.from_map(map_obj, voxel_size=args.voxel_size)

    # Load initial path
    npz = np.load(path_file)
    pos = npz["positions"]
    orn = npz["orientations"]
    path: List[OptimizationState] = [OptimizationState(pos[i], orn[i]) for i in range(len(pos))]
    # Force endpoints to map start/end states
    path[0]  = OptimizationState(start_state[0:3], start_state[6:10])
    path[-1] = OptimizationState(end_state[0:3],  end_state[6:10])

    # Robot
    A = np.load(os.path.join(script_dir, "A_matrix.npy"))
    J = np.load(os.path.join(script_dir, "J_matrix.npy"))
    m = np.load(os.path.join(script_dir, "mass.npy"))
    robot = Robot(J, A, m)

    # Finite-diff velocities/omegas
    minV = np.array([-5, -5, -5]);  maxV = -minV
    minW = np.array([-4, -4, -4]);  maxW = -minW
    dt   = 0.2
    for i in range(len(path) - 1):
        v = (path[i + 1].x - path[i].x) / dt
        path[i].v = np.clip(v, minV, maxV)
        w = (1.0 / dt) * (
            trf.Rotation.from_quat(path[i + 1].q) *
            trf.Rotation.from_quat(path[i].q).inv()
        ).as_rotvec()
        path[i].w = np.clip(w, minW, maxW)

    # State bounds
    state_lower = np.hstack([bounds_min, minV, np.array([-1, -1, -1, -1]), minW])
    state_upper = np.hstack([bounds_max, maxV, np.array([ 1,  1,  1,  1]), maxW])

    # Planner
    planner = GlobalOptimalPlanner(state_lower, state_upper, environment, robot)

    # Indefinite optimization loop WITH visualization every iteration
    all_iters = []
    collisions_accum: List[Obstacle] = []
    base_path = path  # keep initial for comparison in viz

    print("Starting indefinite global optimization. Close the window each iteration to proceed. Press Ctrl+C to stop and save.")
    try:
        it = 0
        while True:
            it += 1
            obstacles, maxDistances = robot.getObstacles(environment, path)
            obstacles = list(obstacles) + list(collisions_accum)

            t0 = time.time()
            try:
                new_path, new_dt = planner.optimize(
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

            # preserve endpoints
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

            # ALWAYS visualize each iteration (blocking until window is closed)
            visualize(environment, robot, base_path, new_path,
                      obstacles=obstacles, collisions=collisions_accum,
                      title=f"Iteration {it}: {'OK' if ok else 'COLLISIONS'}")

            # accumulate collisions and continue
            new_collisions = robot.getCollision(environment, new_path)
            collisions_accum.extend(new_collisions)
            path = new_path  # continue from latest path

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving latest results...")

    # Save latest
    out = {
        "latestPath": path,
        "iterations": all_iters,
        "collisions": collisions_accum,
        "dt": dt,
        "map": args.map,
        "path_file": args.path,
    }
    with open(os.path.join(script_dir, args.save), "wb") as f:
        pickle.dump(out, f)
    print(f"[DONE] Saved results to {os.path.join(script_dir, args.save)}")


if __name__ == "__main__":
    main()
