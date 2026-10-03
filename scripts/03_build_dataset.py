import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import DatasetBuilder
from src.tokenizer import Tokenizer

INPUT = Path("data/final/mixture_final.jsonl")
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")

TRAIN_BIN = Path("dataset/train.bin")
VAL_BIN = Path("dataset/val.bin")
META = Path("dataset/meta.json")

BLOCK_SIZE = 256
VAL_RATIO = 0.01          

tokenizer = Tokenizer().load(TOKENIZER_FILE)

builder = DatasetBuilder(tokenizer, val_ratio=VAL_RATIO)
builder.run(INPUT, TRAIN_BIN, VAL_BIN, META, block_size=BLOCK_SIZE)

x, y = builder.get_batch(TRAIN_BIN, batch_size=1, block_size=8)

print()
print("x (input) :", x[0].tolist())
print("y (target):", y[0].tolist())
print("Shift correct:", (x[0][1:] == y[0][:-1]).all())