import math

import numpy as np
import torch

DTYPE = np.uint16


def full_split_loss(model, bin_path, block_size, device, max_blocks=None, seed=1337):
    data = np.memmap(bin_path, dtype=DTYPE, mode="r")
    n_blocks = (len(data) - 1) // block_size
    block_ids = np.arange(n_blocks)
    if max_blocks is not None and max_blocks < n_blocks:
        rng = np.random.RandomState(seed)
        block_ids = rng.choice(n_blocks, size=max_blocks, replace=False)

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in block_ids:
            start = int(i) * block_size
            x = torch.from_numpy(
                data[start:start + block_size].astype(np.int64)
            ).unsqueeze(0).to(device)
            y = torch.from_numpy(
                data[start + 1:start + block_size + 1].astype(np.int64)
            ).unsqueeze(0).to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * block_size
            total_tokens += block_size

    return total_loss / total_tokens


def random_baseline_loss(vocab_size):
    return math.log(vocab_size)


def bigram_baseline_loss(train_bin_path, val_bin_path, vocab_size):
    train_data = np.memmap(train_bin_path, dtype=DTYPE, mode="r")
    counts = np.ones((vocab_size, vocab_size), dtype=np.float32)

    prev = train_data[:-1].astype(np.int64)
    nxt = train_data[1:].astype(np.int64)
    np.add.at(counts, (prev, nxt), 1)
    probs = counts / counts.sum(axis=1, keepdims=True)

    val_data = np.memmap(val_bin_path, dtype=DTYPE, mode="r")
    prev_v = val_data[:-1].astype(np.int64)
    nxt_v = val_data[1:].astype(np.int64)
    token_probs = probs[prev_v, nxt_v]

    return float(-np.log(token_probs).mean())


def perplexity(loss_nats):
    return math.exp(loss_nats)


def bits_per_byte(loss_nats, n_tokens, n_bytes):
    total_bits = (loss_nats * n_tokens) / math.log(2)
    return total_bits / n_bytes