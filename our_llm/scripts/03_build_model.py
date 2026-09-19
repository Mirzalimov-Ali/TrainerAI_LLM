import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.model_config import load_config
from src.model import OurLLM

CONFIG_FILE = Path("configs/run_01.yaml")

config = load_config(CONFIG_FILE)
model = OurLLM(config)

print("=== Our LLM ===")
print(model)
print()
print("Parameters (real count)        : {:,}".format(model.num_parameters()))

batch = torch.randint(0, config.vocab_size, (2, config.block_size))
logits, _ = model(batch)
print("Smoke test forward pass -> logits shape:", tuple(logits.shape))
print()
print("Next: run 'python tests/test_model.py'")