"""
Datasets for the gated-attention experiment.

- "shakespeare": char-level TinyShakespeare (~1MB). Tiny; overfits fast.
- "enwik8":      byte-level first 100MB of Wikipedia (~36MB zipped). The
                 standard char/byte-level LM benchmark; big enough that a ~40M
                 model can't just memorize it, so val loss is meaningful.

Both expose the same interface via `prepare(dataset)`:
    train_data, val_data, meta = prepare("enwik8")
where meta has: vocab_size, encode, decode.
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
ENWIK8_URLS = [
    "http://mattmahoney.net/dc/enwik8.zip",
    "https://data.deepai.org/enwik8.zip",  # fallback mirror
]


# --------------------------------------------------------------------------
# TinyShakespeare (char-level)
# --------------------------------------------------------------------------
def _prepare_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print(f"Downloading TinyShakespeare -> {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)

    data = np.array(encode(text), dtype=np.int64)
    n = int(len(data) * 0.9)
    meta = dict(vocab_size=len(chars), encode=encode, decode=decode)
    return data[:n], data[n:], meta


# --------------------------------------------------------------------------
# enwik8 (byte-level)
# --------------------------------------------------------------------------
def _prepare_enwik8():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "enwik8")
    if not os.path.exists(raw_path):
        zip_path = os.path.join(DATA_DIR, "enwik8.zip")
        if not os.path.exists(zip_path):
            last_err = None
            for url in ENWIK8_URLS:
                try:
                    print(f"Downloading enwik8 (~36MB) from {url}")
                    urllib.request.urlretrieve(url, zip_path)
                    last_err = None
                    break
                except Exception as e:  # try the next mirror
                    print(f"  failed: {e}")
                    last_err = e
            if last_err is not None:
                raise last_err
        print("Extracting enwik8 ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("enwik8", DATA_DIR)

    with open(raw_path, "rb") as f:
        raw = f.read()  # bytes
    data = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)

    # standard enwik8 split: 90M train / 5M val / 5M test (we use train+val)
    n_train, n_val = 90_000_000, 5_000_000
    train = data[:n_train]
    val = data[n_train : n_train + n_val]

    encode = lambda s: list(s.encode("utf-8"))
    decode = lambda l: bytes(l).decode("utf-8", errors="replace")
    meta = dict(vocab_size=256, encode=encode, decode=decode)
    return train, val, meta


def prepare(dataset="shakespeare"):
    if dataset == "shakespeare":
        return _prepare_shakespeare()
    if dataset == "enwik8":
        return _prepare_enwik8()
    raise ValueError(f"unknown dataset: {dataset}")


def get_batch(split_data, block_size, batch_size, device):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(split_data[i : i + block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(split_data[i + 1 : i + 1 + block_size]) for i in ix])
    return x.to(device), y.to(device)
