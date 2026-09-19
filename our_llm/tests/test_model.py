import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.model_config import load_config
from src.model import OurLLM

CONFIG_FILE = Path("../configs/run_01.yaml")


def shape_test(model, config):
    B, T = 4, 32
    batch = torch.randint(0, config.vocab_size, (B, T))
    logits, _ = model(batch)

    expected = (B, T, config.vocab_size)
    actual = tuple(logits.shape)
    assert actual == expected, "expected logits shape {}, got {}".format(
        expected, actual
    )


def random_init_loss_test(model, config):
    B, T = 8, config.block_size
    batch = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    _, loss = model(batch, targets)
    expected = math.log(config.vocab_size)

    print("  random-init loss = {:.3f}   (expected ~= {:.3f})".format(
        loss.item(), expected))

    assert abs(loss.item() - expected) < 1.0, (
        "loss is too far from ln(vocab_size) - check the loss, the "
        "input/target shift, or the causal mask"
    )


def overfit_one_batch_test(model, config):
    torch.manual_seed(0)
    model.eval()

    B, T = 4, 32
    batch = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    loss = None
    for _ in range(200):
        _, loss = model(batch, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("  final loss after 200 steps on one batch = {:.4f}".format(loss.item()))
    assert loss.item() < 0.1, (
        "loss did not collapse toward zero - the model or optimiser is broken"
    )


def causality_test(model, config):
    B, T = 1, 16
    batch = torch.randint(0, config.vocab_size, (B, T))

    model.eval()
    with torch.no_grad():
        logits_before, _ = model(batch)

        changed = batch.clone()
        t = T // 2
        changed[0, t] = (changed[0, t] + 1) % config.vocab_size

        logits_after, _ = model(changed)

    unaffected_before = torch.allclose(
        logits_before[:, :t], logits_after[:, :t], atol=1e-5
    )
    assert unaffected_before, (
        "logits before the changed position moved - the causal mask is leaking"
    )


def main():
    config = load_config(CONFIG_FILE)

    checks = [
        ("Shape test", shape_test),
        ("Random-init loss test", random_init_loss_test),
        ("Overfit-one-batch test", overfit_one_batch_test),
        ("Causality test", causality_test),
    ]

    print()

    failures = 0
    for name, check in checks:
        print(name + "...")
        try:
            check(OurLLM(config), config)
            print("  PASS")
        except AssertionError as e:
            failures += 1
            print("  FAIL:", e)
        print()

    if failures:
        print("{} check(s) failed - fix the model".format(failures))
        sys.exit(1)
    else:
        print("All checks passed")


if __name__ == "__main__":
    main()