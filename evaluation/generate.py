import argparse
import sys
from pathlib import Path

from inference.engine import InferenceEngine, resolve_checkpoint

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_TEMPERATURES = [0.2, 0.8, 1.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/run_01.yaml"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer.json"))
    args = parser.parse_args()

    engine = InferenceEngine(args.config, resolve_checkpoint(args.checkpoint), args.tokenizer)

    print("Prompt:", repr(args.prompt))
    print()
    for temp in args.temperatures:
        text, n_tokens, latency_ms = engine.generate(
            args.prompt, max_tokens=args.max_tokens, temperature=temp
        )
        print("--- temperature = {} ---".format(temp))
        print(text)
        print("({} tokens, {:.0f} ms)".format(n_tokens, latency_ms))
        print()


if __name__ == "__main__":
    main()