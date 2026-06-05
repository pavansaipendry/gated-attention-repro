"""
Produce the two money plots:

  1. Loss curves: baseline vs gated validation loss over training.
  2. Attention sink: how much attention mass lands on the very first token.
     The paper's claim is that the sigmoid gate makes attention "sink-free",
     i.e. the gated model dumps far less mass onto token 0.

Also saves an attention heatmap for a sample sequence from each model.

Usage:
    python analyze.py
Outputs land in ./figures/.
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401  -- must precede optimizer/dynamo use (see file)
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import prepare
from model import GPT, GPTConfig

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATASET = "shakespeare"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(variant, device):
    ckpt = torch.load(os.path.join(OUT_DIR, f"{variant}.pt"), map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("best_iter")


def plot_loss_curves():
    plt.figure(figsize=(7, 5))
    for variant, color in [("baseline", "tab:gray"), ("gated", "tab:red")]:
        path = os.path.join(OUT_DIR, f"{variant}_history.json")
        if not os.path.exists(path):
            continue
        hist = json.load(open(path))
        iters = [h["iter"] for h in hist]
        plt.plot(iters, [h["val"] for h in hist], color=color, label=f"{variant} (val)")
        plt.plot(
            iters, [h["train"] for h in hist], color=color, linestyle="--", alpha=0.4,
            label=f"{variant} (train)",
        )
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title(f"Gated attention vs baseline — {DATASET}")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(FIG_DIR, "loss_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


def sink_score(att_maps):
    """Mean attention mass placed on token 0 by all query positions t>=1,
    averaged over heads, then reported per layer."""
    per_layer = []
    for att in att_maps:  # (B, H, T, T)
        # attention from queries t>=1 to key position 0
        mass_to_first = att[:, :, 1:, 0]  # (B, H, T-1)
        per_layer.append(mass_to_first.mean().item())
    return per_layer


def plot_attention_sink(device):
    _, val_data, _meta = prepare(DATASET)

    # use the trained context length so the sink is measured over the full
    # window (attention sinks emerge with longer context)
    first_ckpt = next(
        (os.path.join(OUT_DIR, f"{v}.pt") for v in ["baseline", "gated"]
         if os.path.exists(os.path.join(OUT_DIR, f"{v}.pt"))),
        None,
    )
    block = 256
    if first_ckpt:
        block = torch.load(first_ckpt, map_location="cpu")["config"]["block_size"]

    # a fixed sample sequence (same for both models -> fair comparison)
    torch.manual_seed(0)
    start = 1000
    seq = torch.from_numpy(val_data[start : start + block]).unsqueeze(0).to(device)

    results = {}
    heatmaps = {}
    for variant in ["baseline", "gated"]:
        path = os.path.join(OUT_DIR, f"{variant}.pt")
        if not os.path.exists(path):
            continue
        model, _ = load_model(variant, device)
        maps = model.attention_maps(seq)
        results[variant] = sink_score(maps)
        # keep last-layer, head-0 map for the heatmap panel
        heatmaps[variant] = maps[-1][0, 0].cpu().numpy()

    # --- bar chart: sink mass per layer ---
    plt.figure(figsize=(7, 5))
    n_layers = len(next(iter(results.values())))
    x = np.arange(n_layers)
    width = 0.38
    for i, (variant, color) in enumerate(
        [("baseline", "tab:gray"), ("gated", "tab:red")]
    ):
        if variant in results:
            plt.bar(x + (i - 0.5) * width, results[variant], width,
                    label=variant, color=color)
    plt.xlabel("transformer layer")
    plt.ylabel("avg attention mass on token 0  (the 'sink')")
    plt.title("Attention sink: lower = more sink-free")
    plt.xticks(x, [f"L{i}" for i in range(n_layers)])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    out = os.path.join(FIG_DIR, "attention_sink.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")

    # --- side-by-side heatmaps of the last layer, head 0 ---
    if heatmaps:
        fig, axes = plt.subplots(1, len(heatmaps), figsize=(6 * len(heatmaps), 5))
        if len(heatmaps) == 1:
            axes = [axes]
        for ax, (variant, hm) in zip(axes, heatmaps.items()):
            im = ax.imshow(hm, aspect="auto", cmap="viridis")
            ax.set_title(f"{variant}: last-layer head-0 attention")
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
            fig.colorbar(im, ax=ax, fraction=0.046)
        out = os.path.join(FIG_DIR, "attention_heatmaps.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"saved {out}")

    # --- print numeric summary ---
    print("\n=== attention-sink mass per layer (lower = better) ===")
    for variant, scores in results.items():
        pretty = ", ".join(f"L{i}:{s:.3f}" for i, s in enumerate(scores))
        print(f"{variant:8s} -> {pretty}  | mean {np.mean(scores):.3f}")


def main():
    global OUT_DIR, FIG_DIR, DATASET
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--fig_dir", default=FIG_DIR)
    p.add_argument("--dataset", choices=["shakespeare", "enwik8"], default=DATASET)
    args = p.parse_args()
    OUT_DIR = args.out_dir
    FIG_DIR = args.fig_dir
    DATASET = args.dataset

    os.makedirs(FIG_DIR, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")
    plot_loss_curves()
    plot_attention_sink(device)
    print("\nDone. See ./figures/")


if __name__ == "__main__":
    main()
