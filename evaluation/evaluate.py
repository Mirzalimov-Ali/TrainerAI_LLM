import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.jsonl_io import read_jsonl
from src.dataset_builder import DatasetBuilder
from evaluation.perplexity import (
    full_split_loss, random_baseline_loss, bigram_baseline_loss,
    perplexity, bits_per_byte,
)
from inference.engine import InferenceEngine, resolve_checkpoint

CONFIG_PATH = Path("configs/run_01.yaml")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
TRAIN_BIN = Path("dataset/train.bin")
VAL_BIN = Path("dataset/val.bin")
META_FILE = Path("dataset/meta.json")
FINAL_CORPUS = Path("data/final/tinystories.jsonl")

DEMO_PROMPTS = ["Artificial intelligence will change the world because", "Once upon a time"]
DEMO_TEMPERATURES = [0.2, 0.8, 1.5]


def val_byte_count(meta, seed):
    if not FINAL_CORPUS.exists():
        return None
    docs = read_jsonl(FINAL_CORPUS)
    _, val_docs = DatasetBuilder(tokenizer=None, val_ratio=meta["val_ratio"],
                                  seed=seed).split_documents(docs)
    return sum(len(d["text"].encode("utf-8")) for d in val_docs)


def main():
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    engine = InferenceEngine(CONFIG_PATH, resolve_checkpoint(), TOKENIZER_PATH)
    model, device, config = engine.model, engine.device, engine.config

    print("=== Loss ===")
    val_loss = full_split_loss(model, VAL_BIN, config.block_size, device)
    train_loss = full_split_loss(model, TRAIN_BIN, config.block_size, device,
                                  max_blocks=2000)
    print("Train loss (sampled): {:.4f}".format(train_loss))
    print("Val loss (exact)    : {:.4f}".format(val_loss))

    print()
    print("=== Baselines ===")
    random_loss = random_baseline_loss(config.vocab_size)
    print("Computing bigram baseline (one-off, may take a minute)...")
    bigram_loss = bigram_baseline_loss(TRAIN_BIN, VAL_BIN, config.vocab_size)
    print("Random baseline loss: {:.4f}  (ln(vocab_size))".format(random_loss))
    print("Bigram baseline loss: {:.4f}".format(bigram_loss))
    print("Our val loss        : {:.4f}".format(val_loss))

    n_bytes = val_byte_count(meta, meta["seed"])
    bpb = None
    if n_bytes:
        bpb = bits_per_byte(val_loss, meta["n_tokens_val"], n_bytes)
        print("Bits-per-byte (val) : {:.4f}".format(bpb))
    else:
        print("data/final/ not present here - skipping bits-per-byte "
              "(re-run the data pipeline in this environment to get it)")

    print()
    print("=== Generation samples ===")
    samples = []
    for prompt in DEMO_PROMPTS:
        for temp in DEMO_TEMPERATURES:
            text, n_tokens, latency_ms = engine.generate(prompt, temperature=temp)
            print("[T={}] {} -> {}".format(temp, repr(prompt), text))
            samples.append({
                "prompt": prompt, "temperature": temp, "text": text,
                "tokens_generated": n_tokens, "latency_ms": latency_ms,
            })

    gap = val_loss - train_loss
    verdict = (
        "train/val gap is small - not clearly overfitting yet"
        if gap < 0.3 else
        "train/val gap is meaningful - some overfitting, expected on a "
        "~9-epoch run over a small corpus"
    )
    print()
    print("=== Verdict ===")
    print("train/val gap: {:.4f} -> {}".format(gap, verdict))

    results = {
        "parameters": {
            "total": model.num_parameters(),
        },
        "dataset": {
            "train_tokens": meta["n_tokens_train"],
            "val_tokens": meta["n_tokens_val"],
            "vocab_size": meta["vocab_size"],
        },
        "loss": {
            "train_sampled": train_loss,
            "val_exact": val_loss,
            "train_val_gap": gap,
        },
        "perplexity": {
            "val": perplexity(val_loss),
            "random_baseline": perplexity(random_loss),
            "bigram_baseline": perplexity(bigram_loss),
            "bits_per_byte_val": bpb,
        },
        "generation_samples": samples,
        "verdict": verdict,
    }

    out_path = Path("evaluation/results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print("Saved", out_path)


if __name__ == "__main__":
    main()