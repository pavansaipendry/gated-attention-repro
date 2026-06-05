"""
Test the paper's third claim: the sigmoid gate induces *query-dependent
sparsity*. We load the gated model and inspect its gate values
g = sigmoid(W_g x) in [0, 1] for a batch of real validation sequences.

We quantify two things per layer:
  - SPARSITY:  how "closed" the gate is. Fraction of gate units < 0.1, and the
               mean gate value. A gate that is often near 0 is selectively
               suppressing the attention output (i.e. sparse contribution).
  - QUERY-DEPENDENCE: how much the gate's openness varies across query positions.
               For each position we take the mean gate over channels, giving a
               length-T vector; its std across positions measures how strongly
               the gate depends on the query. ~0 would mean "same gate
               everywhere" (not query-dependent); larger means position-specific.

Outputs (to --fig_dir): gate_sparsity.png (per-layer bars), gate_histogram.png,
gate_heatmap.png (positions x channels for one layer), plus a printed summary.

Usage:
    python sparsity.py --dataset enwik8 --out_dir out_big --fig_dir figures_enwik8
"""

import argparse
import os

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import prepare
from model import GPT, GPTConfig


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gated(out_dir, device):
    ckpt = torch.load(os.path.join(out_dir, "gated.pt"), map_location=device)
    config = GPTConfig(**ckpt["config"])
    assert config.gate_attn, "sparsity analysis needs the gated model"
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["shakespeare", "enwik8"], default="shakespeare")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--fig_dir", default="figures")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--closed_thresh", type=float, default=0.1)
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")
    model = load_gated(args.out_dir, device)
    block = model.config.block_size
    n_layer = model.config.n_layer

    # a batch of real validation sequences
    _, val_data, _ = prepare(args.dataset)
    torch.manual_seed(0)
    ix = torch.randint(len(val_data) - block, (args.batch,))
    xb = torch.stack(
        [torch.from_numpy(val_data[i : i + block]) for i in ix]
    ).to(device)

    gates = model.gate_activations(xb)  # list[n_layer] of (B, T, C)

    mean_per_layer = []      # average gate value
    frac_closed = []         # fraction of units below threshold (sparsity)
    query_dep = []           # std of per-position mean gate (query dependence)
    for g in gates:
        mean_per_layer.append(g.mean().item())
        frac_closed.append((g < args.closed_thresh).float().mean().item())
        per_pos = g.mean(dim=(0, 2))           # (T,) mean gate at each position
        query_dep.append(per_pos.std().item())

    # ---- plot 1: per-layer sparsity ----
    x = np.arange(n_layer)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - 0.2, frac_closed, 0.4, color="tab:purple",
            label=f"frac gate < {args.closed_thresh} (sparsity)")
    ax1.set_ylabel("fraction of gate units closed", color="tab:purple")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, mean_per_layer, 0.4, color="tab:orange", label="mean gate value")
    ax2.set_ylabel("mean gate value", color="tab:orange")
    ax2.set_ylim(0, 1)
    ax1.set_xticks(x); ax1.set_xticklabels([f"L{i}" for i in range(n_layer)])
    ax1.set_xlabel("transformer layer")
    plt.title("Gate sparsity per layer (closed = suppressing attention output)")
    out = os.path.join(args.fig_dir, "gate_sparsity.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"saved {out}")

    # ---- plot 2: histogram of all gate values ----
    allvals = torch.cat([g.flatten() for g in gates]).float().cpu().numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(allvals, bins=60, color="tab:purple", alpha=0.8)
    plt.axvline(args.closed_thresh, color="k", ls="--", alpha=0.5,
                label=f"closed threshold {args.closed_thresh}")
    plt.xlabel("gate value  sigmoid(W_g x)"); plt.ylabel("count")
    plt.title("Distribution of gate values (mass near 0 = sparsity)")
    plt.legend()
    out = os.path.join(args.fig_dir, "gate_histogram.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); print(f"saved {out}")

    # ---- plot 3: heatmap (positions x channels) for the most query-dependent layer ----
    L = int(np.argmax(query_dep))
    hm = gates[L][0].float().cpu().numpy()        # (T, C) for sample 0
    # show a channel slice so structure is visible
    plt.figure(figsize=(8, 5))
    plt.imshow(hm[:, :128].T, aspect="auto", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(label="gate value")
    plt.xlabel("query position"); plt.ylabel("channel (first 128)")
    plt.title(f"Gate values vary by position — layer L{L} (most query-dependent)")
    out = os.path.join(args.fig_dir, "gate_heatmap.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); print(f"saved {out}")

    # ---- summary ----
    print("\n=== gate sparsity / query-dependence per layer ===")
    for i in range(n_layer):
        print(f"L{i:<2d} mean {mean_per_layer[i]:.3f} | "
              f"closed<{args.closed_thresh} {frac_closed[i]:.2%} | "
              f"query-dep(std over pos) {query_dep[i]:.4f}")
    print(f"\noverall: mean gate {np.mean(mean_per_layer):.3f} | "
          f"mean frac closed {np.mean(frac_closed):.2%} | "
          f"mean query-dep {np.mean(query_dep):.4f}")
    print("query-dep ~0 would mean the gate ignores position; "
          ">0 means it is query-dependent.")


if __name__ == "__main__":
    main()
