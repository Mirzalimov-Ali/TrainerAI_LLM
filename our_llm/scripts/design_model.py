import json
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_config import load_config

CONFIG_FILE = PROJECT_ROOT / "configs" / "run_01.yaml"
META_FILE = PROJECT_ROOT / "dataset" / "meta.json"

config = load_config(CONFIG_FILE)
params = config.estimate_parameters()

print("=== Our LLM Model Design ===")
print()
print("Vocabulary size    :", config.vocab_size)
print("Context length     :", config.block_size)
print("Embedding dimension:", config.n_embd)
print("Layers             :", config.n_layer)
print("Attention heads    :", config.n_head, "(head dim", config.head_dim(), ")")
print("Dropout            :", config.dropout)
print("FFN multiplier     :", config.ffn_mult)
print("Tied weights       :", config.tie_weights)
print()

print("Estimated parameters")
print("  non-embedding :  {:>12,}".format(params["non_embedding"]))
print("  embedding     :  {:>12,}".format(params["embedding"]))
print("  position      :  {:>12,}".format(params["position"]))
print("  TOTAL         :  {:>12,}   (~{:.1f}M)".format(
    params["total"], params["total"] / 1e6))
print()

print("Rough training memory for weights + gradients + AdamW: {:.0f} MB".format(
    config.estimate_training_memory_mb()))
print("(activations add more on top, and they grow with batch size)")
print()
 
if META_FILE.exists():
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    n_tokens = meta["n_tokens_train"]
    ratio = n_tokens / params["total"]

    print("Training tokens available: {:,}".format(n_tokens))
    print("Tokens per parameter     : {:.1f}   (about 20 is the target)".format(ratio))

    if ratio < 10:
        print("-> The model is large for this corpus. Shrink it, or get more data.")
    elif ratio > 40:
        print("-> Plenty of data for this model. We could afford to go bigger.")
    else:
        print("-> Reasonably balanced.")
else:
    print("No dataset/meta.json yet - run scripts/build_dataset.py to check")
    print("the tokens-per-parameter ratio against a real corpus.")
