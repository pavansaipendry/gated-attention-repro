"""
Train a baseline GPT and a gated-attention GPT on TinyShakespeare, under
identical settings, and save each checkpoint + its loss history.

Usage:
    python train.py --variant baseline
    python train.py --variant gated
    python train.py --variant both      # default: trains both back to back

Everything runs on MPS (Apple Silicon), CUDA, or CPU automatically.
Designed to finish in minutes on an M-series Mac at the default size.
"""

import argparse
import json
import os
import time

import _bootstrap  # noqa: F401  -- must precede optimizer creation (see file)
import torch

from data import get_batch, prepare
from model import GPT, GPTConfig

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train_one(variant, cfg_kwargs, train_data, val_data, device, args):
    out_dir = args.out_dir
    torch.manual_seed(1337)  # same init/data order for a fair comparison
    config = GPTConfig(gate_attn=(variant == "gated"), **cfg_kwargs)
    model = GPT(config).to(device)
    print(
        f"\n=== {variant} === params: {model.num_params()/1e6:.2f}M  "
        f"gate_attn={config.gate_attn}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1
    )
    splits = {"train": train_data, "val": val_data}
    history = []
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{variant}.pt")

    best_val = float("inf")
    best_iter = 0
    bad_evals = 0  # consecutive evals without val improvement
    t0 = time.time()

    for it in range(args.iters + 1):
        if it % args.eval_interval == 0 or it == args.iters:
            losses = estimate_loss(
                model, splits, config.block_size, args.batch_size, device
            )
            dt = time.time() - t0
            improved = losses["val"] < best_val - 1e-4
            print(
                f"[{variant}] iter {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {dt:5.1f}s"
                + ("  * best -> saved" if improved else "")
            )
            history.append({"iter": it, **losses})

            if improved:
                best_val, best_iter, bad_evals = losses["val"], it, 0
                # checkpoint the BEST model (early-stopping snapshot), so analysis
                # runs on the best weights rather than an overfit final state
                torch.save(
                    {"model": model.state_dict(), "config": config.__dict__,
                     "best_val": best_val, "best_iter": best_iter},
                    ckpt_path,
                )
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"[{variant}] early stop at iter {it} "
                          f"(no val improvement for {args.patience} evals; "
                          f"best {best_val:.4f} @ iter {best_iter})")
                    break

        xb, yb = get_batch(train_data, config.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with open(os.path.join(out_dir, f"{variant}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{variant}] best val {best_val:.4f} @ iter {best_iter} -> {ckpt_path}")
    return {"history": history, "best_val": best_val, "best_iter": best_iter}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "gated", "both"], default="both")
    p.add_argument("--dataset", choices=["shakespeare", "enwik8"], default="shakespeare")
    p.add_argument("--patience", type=int, default=1000,
                   help="early stop after this many evals without val improvement")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    device = pick_device()
    print(f"device: {device}")
    # A100/Ampere: use TF32 tensor cores for fp32 matmuls (big speedup, applied
    # equally to both variants so the comparison stays fair)
    torch.set_float32_matmul_precision("high")

    train_data, val_data, meta = prepare(args.dataset)
    print(f"dataset {args.dataset} | vocab {meta['vocab_size']} | "
          f"train {len(train_data):,} | val {len(val_data):,}")

    cfg_kwargs = dict(
        block_size=args.block_size,
        vocab_size=meta["vocab_size"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    variants = ["baseline", "gated"] if args.variant == "both" else [args.variant]
    results = {}
    for v in variants:
        results[v] = train_one(v, cfg_kwargs, train_data, val_data, device, args)

    if len(results) == 2:
        import math as _m
        b = results["baseline"]["best_val"]
        g = results["gated"]["best_val"]
        ln2 = _m.log(2)
        print("\n================ SUMMARY (best / early-stopped) ================")
        print(f"baseline best val: {b:.4f}  ({b/ln2:.4f} bpc) @ iter {results['baseline']['best_iter']}")
        print(f"gated    best val: {g:.4f}  ({g/ln2:.4f} bpc) @ iter {results['gated']['best_iter']}")
        print(f"improvement: {(b - g):+.4f}  ({100*(b-g)/b:+.2f}%)")
        print("================================================================")


if __name__ == "__main__":
    main()
