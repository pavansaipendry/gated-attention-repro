#!/usr/bin/env bash
# Run the "go big" gated-attention experiment on a RunPod CUDA pod.
# Assumes a runpod/pytorch image (torch + numpy already present).
#
# Usage on the pod:
#   bash run_remote.sh
#
# Produces out_big/ checkpoints + figures_big/ plots, which you then pull back.
set -euo pipefail

cd "$(dirname "$0")"

echo "== GPU =="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "== deps =="
pip install -q matplotlib numpy   # torch ships in the runpod pytorch image

echo "== train on enwik8 (baseline + gated, early stopping) =="
python -u train.py --variant both --dataset enwik8 \
  --n_layer 12 --n_head 8 --n_embd 512 \
  --block_size 1024 --batch_size 32 \
  --iters 8000 --eval_interval 500 --patience 4 \
  --out_dir out_big

echo "== analyze (best/early-stopped checkpoints) =="
python analyze.py --dataset enwik8 --out_dir out_big --fig_dir figures_big

echo "== DONE -- results in out_big/ and figures_big/ =="
ls -la figures_big/
