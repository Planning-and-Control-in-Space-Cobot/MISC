#!/usr/bin/env python3
"""
Single grouped plot:
  For each map (x-axis), show two violins side-by-side:
    - Bi-RRT (pruned)
    - Bi-RRT* (pruned)

- Success-only.
- One chart, no subplots. Matplotlib only. No explicit colors.
- Legend colors are taken from the plotted violins.
"""

import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# -------------------- Load --------------------
def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

# -------------------- Map name helpers --------------------
_MAP_NAME_MAP = {
    "simpleMap.pcd": "Simple Map",
    "middleMap.pcd": "Intermediary Map",
    "complexMap.pcd": "Complex Map",
}
_MAP_ORDER = ["simpleMap.pcd", "middleMap.pcd", "complexMap.pcd"]

def pretty_map_name(env_name: str) -> str:
    return _MAP_NAME_MAP.get(env_name, env_name)

def sort_maps_for_display(maps):
    known = [m for m in _MAP_ORDER if m in maps]
    others = sorted([m for m in maps if m not in _MAP_ORDER])
    return known + others

# -------------------- Helpers --------------------
def collect_by_map(runs):
    by_map = defaultdict(list)
    for r in runs:
        m = r.get("params", {}).get("env", "UNKNOWN")
        by_map[m].append(r)
    return by_map

def extract_pruned_lengths_by_map(by_map, success_only=True):
    out = {}
    for m, lst in by_map.items():
        bi_p, st_p = [], []
        for e in lst:
            bi   = e.get("BiRRT", {})
            star = e.get("BiRRTStar", {})
            if (not success_only) or bi.get("success", False):
                v = bi.get("prunedPathLength", None)
                if isinstance(v, (int, float)):
                    bi_p.append(float(v))
            if (not success_only) or star.get("success", False):
                v = star.get("prunedPathLength", None)
                if isinstance(v, (int, float)):
                    st_p.append(float(v))
        out[m] = {"BiRRT_pruned": bi_p, "BiRRTStar_pruned": st_p}
    return out

def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

# -------------------- Plot --------------------
def plot_grouped_pruned_by_map(pruned_by_map, outdir):
    maps_raw = sort_maps_for_display(list(pruned_by_map.keys()))
    if not maps_raw:
        print("No maps to plot.")
        return None

    centers = np.arange(len(maps_raw))
    offset = 0.18
    width = 0.30

    data_bi, data_st = [], []
    pos_bi, pos_st = [], []
    xticklabels = []

    for i, m in enumerate(maps_raw):
        xticklabels.append(pretty_map_name(m))
        bi = pruned_by_map[m]["BiRRT_pruned"]
        st = pruned_by_map[m]["BiRRTStar_pruned"]
        data_bi.append(bi if len(bi) > 0 else [np.nan])
        data_st.append(st if len(st) > 0 else [np.nan])
        pos_bi.append(centers[i] - offset)
        pos_st.append(centers[i] + offset)

    fig = plt.figure()
    ax = plt.gca()

    # Draw violins
    v1 = ax.violinplot(data_bi, positions=pos_bi, widths=width, showmeans=True, showmedians=False, showextrema=True)
    v2 = ax.violinplot(data_st, positions=pos_st, widths=width, showmeans=True, showmedians=False, showextrema=True)

    # Use the actual violin colors in the legend and make each group consistent
    bi_color = v1['bodies'][0].get_facecolor()[0]
    st_color = v2['bodies'][0].get_facecolor()[0]
    for pc in v1['bodies']:
        pc.set_facecolor(bi_color)
    for pc in v2['bodies']:
        pc.set_facecolor(st_color)

    ax.set_xticks(centers)
    ax.set_xticklabels(xticklabels, rotation=0, ha="center", fontsize=13)
    ax.tick_params(axis="y", which="major", labelsize=13)
    ax.set_ylabel("Pruned path length (nodes)", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Legend with proxy patches that match violin colors
    handles = [
        Patch(facecolor=bi_color, edgecolor='none', label="Bi-RRT (pruned)"),
        Patch(facecolor=st_color, edgecolor='none', label="Bi-RRT* (pruned)")
    ]
    ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    out_path = os.path.join(outdir, "grouped_pruned_path_lengths_all_maps.png")
    fig.savefig(out_path, dpi=200)
    plt.show()
    return out_path

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Grouped pruned path length per map: Bi-RRT vs Bi-RRT*")
    ap.add_argument("--results", type=str, default="rrt_results.json")
    ap.add_argument("--outdir", type=str, default="study_charts")
    args = ap.parse_args()

    runs = load_results(args.results)
    if not isinstance(runs, list) or len(runs) == 0:
        print("No runs found in results JSON.")
        return

    ensure_outdir(args.outdir)
    by_map = collect_by_map(runs)
    pruned_by_map = extract_pruned_lengths_by_map(by_map, success_only=True)

    fig_path = plot_grouped_pruned_by_map(pruned_by_map, args.outdir)
    print(f"Wrote grouped figure to: {fig_path}")

if __name__ == "__main__":
    main()
