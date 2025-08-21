import os
import numpy as np
import open3d as o3d
import scipy.io as sio
import pyvista as pv

from scipy.spatial.transform import Rotation as R

def _extract_voxel_mesh_arrays(voxel_mesh):
    """
    Returns (points, quads, tris) from a pyvista mesh.
    quads/tris are integer (M,4)/(K,3) index arrays into points.
    """
    if voxel_mesh is None:
        return None, None, None

    pts = np.asarray(voxel_mesh.points, dtype=np.float64)

    # VTK face array format: [n, i0, i1, ..., n, j0, j1, ...]
    faces = np.asarray(voxel_mesh.faces, dtype=np.int64)
    quads = []
    tris = []
    i = 0
    while i < len(faces):
        n = faces[i]
        idx = faces[i+1 : i+1+n]
        if n == 4:
            quads.append(idx)
        elif n == 3:
            tris.append(idx)
        i += (1 + n)

    quads = np.asarray(quads, dtype=np.int64) if quads else np.empty((0,4), dtype=np.int64)
    tris  = np.asarray(tris,  dtype=np.int64) if tris  else np.empty((0,3), dtype=np.int64)
    return pts, quads, tris


def main():
    here = os.path.dirname(__file__)

    # --- Load environment (point cloud) ---
    pcd_path = os.path.join(here, 'complexMap.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)
    env_points = np.asarray(pcd.points, dtype=np.float64)

    # --- Build/voxelize environment with your handler (optional but nice to have) ---
    from Environment import EnvironmentHandler
    env = EnvironmentHandler(pcd)

    voxel_points, voxel_quads, voxel_tris = None, None, None
    if hasattr(env, 'voxel_mesh') and isinstance(env.voxel_mesh, (pv.PolyData, pv.UnstructuredGrid)):
        voxel_points, voxel_quads, voxel_tris = _extract_voxel_mesh_arrays(env.voxel_mesh)

    # --- Load paths (your original path.npz) ---
    path_npz = np.load(os.path.join(here, 'path.npz'))
    pruned_positions        = path_npz['positions'].astype(np.float64)              # (N,3)
    pruned_orientations     = path_npz['orientations'].astype(np.float64)           # (N,4) [x,y,z,w]
    unpruned_positions      = path_npz['unprunedPositions'].astype(np.float64)      # (M,3)
    unpruned_orientations   = path_npz['unprunedOrientations'].astype(np.float64)   # (M,4)

    # --- Robot size (full box dims, as in your plot) ---
    robot_size = np.array([0.45, 0.45, 0.12], dtype=np.float64)  # [RX, RY, RZ]

    # --- Bounds (optional; useful for consistent MATLAB framing) ---
    boundsX = np.array([0.0, 3.0], dtype=np.float64)
    boundsY = np.array([2.8, 5.5], dtype=np.float64)
    boundsZ = np.array([0.0, 7.0], dtype=np.float64)

    # --- Save NumPy bundle ---
    out_npz = os.path.join(here, 'scene_export.npz')
    np.savez_compressed(
        out_npz,
        env_points=env_points,
        voxel_points=voxel_points if voxel_points is not None else np.empty((0,3), np.float64),
        voxel_quads=voxel_quads if voxel_quads is not None else np.empty((0,4), np.int64),
        voxel_tris=voxel_tris   if voxel_tris   is not None else np.empty((0,3), np.int64),
        pruned_positions=pruned_positions,
        pruned_orientations=pruned_orientations,
        unpruned_positions=unpruned_positions,
        unpruned_orientations=unpruned_orientations,
        robot_size=robot_size,
        boundsX=boundsX, boundsY=boundsY, boundsZ=boundsZ
    )
    print(f"[OK] Wrote {out_npz}")

    # --- Also save a MATLAB .mat for easy loading in MATLAB ---
    out_mat = os.path.join(here, 'scene_export.mat')
    sio.savemat(out_mat, {
        'env_points': env_points,
        'voxel_points': voxel_points if voxel_points is not None else np.empty((0,3), np.float64),
        'voxel_quads': voxel_quads if voxel_quads is not None else np.empty((0,4), np.int64),
        'voxel_tris':  voxel_tris  if voxel_tris  is not None else np.empty((0,3), np.int64),
        'pruned_positions': pruned_positions,
        'pruned_orientations': pruned_orientations,
        'unpruned_positions': unpruned_positions,
        'unpruned_orientations': unpruned_orientations,
        'robot_size': robot_size,
        'boundsX': boundsX, 'boundsY': boundsY, 'boundsZ': boundsZ
    })
    print(f"[OK] Wrote {out_mat}")


if __name__ == "__main__":
    main()
