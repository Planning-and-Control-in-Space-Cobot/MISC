#!/usr/bin/env python3
import os
import pickle
from typing import Tuple

import numpy as np
import open3d as o3d

MAPS_DIR = "/home/andret/MEEC/Thesis/Code/MISC/PathPlanning/Maps"
PCD_DIR  = os.path.join(MAPS_DIR, "pcds")

# ---------------------------
# Helpers
# ---------------------------
def make_state(pos: Tuple[float, float, float],
               quat_xyzw: Tuple[float, float, float, float]) -> np.ndarray:
    """13-dim state: [pos(3), vel(3)=0, quat(xyzw)(4), omega(3)=0].
    Quaternion normalization exactly as you do it."""
    px, py, pz = pos
    qx, qy, qz, qw = quat_xyzw
    state = np.array([px, py, pz, 0.0, 0.0, 0.0, qx, qy, qz, qw, 0.0, 0.0, 0.0], dtype=float)
    state[6:10] /= np.linalg.norm(state[6:10])  # your style
    return state

def save_map_meta(out_pkl_path: str,
                  pcd_path: str,
                  bounds_min: np.ndarray,
                  bounds_max: np.ndarray,
                  start_state: np.ndarray,
                  end_state: np.ndarray,
                  also_save_start_npy: bool = True) -> None:
    meta = {
        "point_cloud_path": os.path.abspath(pcd_path),
        "start_state": start_state,
        "end_state": end_state,
        "bounds_min": bounds_min.astype(float),
        "bounds_max": bounds_max.astype(float),
    }
    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Saved: {out_pkl_path}")

    if also_save_start_npy:
        npy_path = os.path.join(os.path.dirname(out_pkl_path),
                                f"{os.path.splitext(os.path.basename(out_pkl_path))[0]}_start_state.npy")
        np.save(npy_path, start_state)
        print(f"[OK] Saved start state .npy: {npy_path}")

def aabb_from_pcd(pcd_path: str) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {pcd_path}")
    aabb = pcd.get_axis_aligned_bounding_box()
    return np.asarray(aabb.get_min_bound()), np.asarray(aabb.get_max_bound())

# ---------------------------
# Known maps
# ---------------------------
def main():
    os.makedirs(MAPS_DIR, exist_ok=True)

    # ---------- simpleMap ----------
    simple_pcd = os.path.join(PCD_DIR, "simpleMap.pcd")
    simple_bounds_min = np.array([-5.0, 0.0, -2.5])
    simple_bounds_max = np.array([ 4.0, 8.0,  2.5])
    simple_start_pos  = np.array([3.5, 0.5, 2.0])
    simple_goal_pos   = np.array([3.0, 7.0, 0.0])
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    save_map_meta(
        out_pkl_path=os.path.join(MAPS_DIR, "simpleMap.pkl"),
        pcd_path=simple_pcd,
        bounds_min=simple_bounds_min,
        bounds_max=simple_bounds_max,
        start_state=make_state(tuple(simple_start_pos), tuple(q_id)),
        end_state=make_state(tuple(simple_goal_pos), tuple(q_id)),
    )

    # ---------- middleMap ----------
    middle_pcd = os.path.join(PCD_DIR, "middleMap.pcd")
    middle_bounds_min = np.array([1.5, 1.0, 0.0])
    middle_bounds_max = np.array([4.0, 6.0, 10.0])
    middle_start_pos  = np.array([2.75, 2.0, 1.0])
    middle_goal_pos   = np.array([2.75, 2.0, 7.0])
    middle_q          = np.array([0.0, 0.707, 0.0, 0.707])  # xyzw
    save_map_meta(
        out_pkl_path=os.path.join(MAPS_DIR, "middleMap.pkl"),
        pcd_path=middle_pcd,
        bounds_min=middle_bounds_min,
        bounds_max=middle_bounds_max,
        start_state=make_state(tuple(middle_start_pos), tuple(middle_q)),
        end_state=make_state(tuple(middle_goal_pos), tuple(middle_q)),
    )

    # ---------- complexMap ----------
    complex_pcd = os.path.join(PCD_DIR, "complexMap.pcd")
    complex_bounds_min = np.array([0.0, 3.0, 0.0])
    complex_bounds_max = np.array([3.0, 6.5, 7.0])
    complex_start_pos  = np.array([0.5, 3.5, 1.0])
    complex_goal_pos   = np.array([0.5, 5.0, 6.0])
    save_map_meta(
        out_pkl_path=os.path.join(MAPS_DIR, "complexMap.pkl"),
        pcd_path=complex_pcd,
        bounds_min=complex_bounds_min,
        bounds_max=complex_bounds_max,
        start_state=make_state(tuple(complex_start_pos), tuple(q_id)),
        end_state=make_state(tuple(complex_goal_pos), tuple(q_id)),
    )

    # ---------- DoubleSphere (using your params) ----------
    ds_pcd = os.path.join(PCD_DIR, "DoubleSphere.pcd")
    if os.path.exists(ds_pcd):
        ds_min, ds_max = aabb_from_pcd(ds_pcd)  # bounds from PCD AABB
        ds_min = np.array([-4, -3, 2])
        ds_max = np.array([14, 5, 7])
        ds_start_pos   = np.array([-3.0, 3.0, 5.0])   # from your MATLAB (center)
        ds_end_pos     = np.array([12.0, 0.0, 5.0])   # from your MATLAB (center)
        save_map_meta(
            out_pkl_path=os.path.join(MAPS_DIR, "DoubleSphere.pkl"),
            pcd_path=ds_pcd,
            bounds_min=ds_min,
            bounds_max=ds_max,
            start_state=make_state(tuple(ds_start_pos), tuple(q_id)),
            end_state=make_state(tuple(ds_end_pos),   tuple(q_id)),
        )
    else:
        print(f"[WARN] DoubleSphere PCD not found at {ds_pcd}; skipping.")

    print("\nAll done.")

if __name__ == "__main__":
    main()
