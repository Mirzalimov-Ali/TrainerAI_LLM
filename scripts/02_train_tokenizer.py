import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jsonl_io import read_jsonl
from src.tokenizer import Tokenizer

INPUT = Path("data/final/mixture_final.jsonl")
OUTPUT = Path("tokenizer/tokenizer.json")

VOCAB_SIZE = 8000

MAX_DOCUMENTS = 400000

documents = read_jsonl(INPUT, limit=MAX_DOCUMENTS)
texts = [doc["text"] for doc in documents]
print("Training on", len(texts), "documents")

tokenizer = Tokenizer()
tokenizer.train(texts, vocab_size=VOCAB_SIZE)
tokenizer.save(OUTPUT)

sample = "Do hummer curl for brachialis"
ids = tokenizer.encode(sample)
back = tokenizer.decode(ids)

print()
print("Text   :", sample)
print("IDs    :", ids)
print("Decoded:", back)
print("Round-trip OK:", back == sample)