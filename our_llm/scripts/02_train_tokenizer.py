# to run this file: python train_tokenizer.py --input ../data/final/fitness_final.jsonl --output tokenizer/tokenizer.json --vocab-size 1000 --max-docs 5000

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

SPECIAL_TOKENS = ["<eos>"]


def iter_texts(path, limit=None):
    """JSONL fayldan matnlarni oqim sifatida o'qiydi (RAM ga sig'dirmasdan)."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)["text"]


def build_tokenizer(vocab_size, min_frequency):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_frequency,
        show_progress=True,
    )
    return tok, trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-docs", type=int, default=200_000,
                   help="BPE statistikasi tez to'yinadi; None uchun 0 bering")
    args = p.parse_args()

    if args.vocab_size >= 65_536:
        raise SystemExit("vocab_size < 65536 bo'lishi kerak (uint16 dataset uchun)")

    limit = args.max_docs or None
    tok, trainer = build_tokenizer(args.vocab_size, args.min_frequency)
    tok.train_from_iterator(iter_texts(args.input, limit), trainer=trainer)

    # Round-trip
    for probe in ["Once upon a time.", "  bo'sh   joylar ", "qator\nqator", "salom!"]:
        back = tok.decode(tok.encode(probe).ids)
        if back != probe:
            raise SystemExit("Round-trip xato: %r -> %r" % (probe, back))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.output))
    print("Saqlandi: %s  (vocab_size=%d)" % (args.output, tok.get_vocab_size()))


if __name__ == "__main__":
    main()
