"""
Aggregate best validation loss across seeds and plot mean +/- std for
baseline vs gated. Reads out_seed*/{variant}_history.json; best val per run is
the minimum val over its history (matches the early-stopped checkpoint).

Usage:
    python aggregate.py --seed_dirs out_seed0 out_seed1 out_seed2 \
        --fig_dir figures_enwik8
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def best_val(hist_path):
    hist = json.load(open(hist_path))
    return min(h["val"] for h in hist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", required=True)
    p.add_argument("--fig_dir", default="figures")
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    ln2 = math.log(2)

    stats = {}
    for variant in ["baseline", "gated"]:
        vals = []
        for d in args.seed_dirs:
            path = os.path.join(d, f"{variant}_history.json")
            if os.path.exists(path):
                vals.append(best_val(path))
        if vals:
            stats[variant] = np.array(vals)

    print("=== best val loss across seeds ===")
    for v, arr in stats.items():
        bpc = arr / ln2
        print(f"{v:8s} n={len(arr)} | val {arr.mean():.4f} +/- {arr.std():.4f} "
              f"| bpc {bpc.mean():.4f} +/- {bpc.std():.4f} | raw {np.round(arr,4)}")
    if {"baseline", "gated"} <= stats.keys():
        b, g = stats["baseline"].mean(), stats["gated"].mean()
        print(f"\nmean improvement: {(b-g):+.4f}  ({100*(b-g)/b:+.2f}%)")

    # bar chart with error bars (bits/char)
    plt.figure(figsize=(5.5, 5))
    names = [v for v in ["baseline", "gated"] if v in stats]
    means = [stats[v].mean() / ln2 for v in names]
    errs = [stats[v].std() / ln2 for v in names]
    colors = {"baseline": "tab:gray", "gated": "tab:red"}
    plt.bar(names, means, yerr=errs, capsize=8,
            color=[colors[n] for n in names])
    for i, v in enumerate(names):
        plt.text(i, means[i], f" {means[i]:.3f}", ha="center", va="bottom")
    plt.ylabel("best validation loss (bits/char)")
    n = len(next(iter(stats.values())))
    plt.title(f"enwik8 best val loss — mean ± std over {n} seeds")
    out = os.path.join(args.fig_dir, "loss_seeds.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
