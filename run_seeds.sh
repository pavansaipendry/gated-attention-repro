#!/usr/bin/env bash
# Multi-seed enwik8 run on a GPU pod: trains baseline+gated for several seeds,
# then produces error-bar loss, sink, and gate-sparsity figures.
# Configurable via env: SEEDS (default "0 1 2"), ITERS (default 8000).
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-0 1 2}"
ITERS="${ITERS:-8000}"
echo "== GPU =="; python -c "import torch;print(torch.cuda.get_device_name(0))"
pip install -q matplotlib numpy

for s in $SEEDS; do
  echo "== seed $s (iters=$ITERS) =="
  python -u train.py --variant both --dataset enwik8 --seed "$s" \
    --n_layer 12 --n_head 8 --n_embd 512 --block_size 1024 --batch_size 32 \
    --iters "$ITERS" --eval_interval 500 --patience 4 --out_dir "out_seed$s"
done

DIRS=""; for s in $SEEDS; do DIRS="$DIRS out_seed$s"; done
FIRST=$(echo $SEEDS | awk '{print $1}')

echo "== aggregate (error bars across seeds) =="
python aggregate.py --seed_dirs $DIRS --fig_dir figures_seeds

echo "== loss/sink + gate sparsity (on seed $FIRST) =="
python analyze.py  --dataset enwik8 --out_dir "out_seed$FIRST" --fig_dir figures_seeds
python sparsity.py --dataset enwik8 --out_dir "out_seed$FIRST" --fig_dir figures_seeds

echo "== DONE -- figures in figures_seeds/ =="
ls -la figures_seeds/
