#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from jwall_gpt.data.prepare import split_tokens, tokenize_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize a text corpus into train/val .bin shards"
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Input text file")
    parser.add_argument("--out", type=Path, required=True, help="Output train .bin path")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of tokens held out for validation (0 disables the val split)",
    )
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    tokens = tokenize_corpus(text)
    train_tokens, val_tokens = split_tokens(tokens, args.val_frac)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_tokens.tofile(args.out)
    print(f"Wrote {len(train_tokens):,} train tokens to {args.out}")

    if len(val_tokens) > 0:
        val_path = args.out.parent / "val.bin"
        val_tokens.tofile(val_path)
        print(f"Wrote {len(val_tokens):,} val tokens to {val_path}")


if __name__ == "__main__":
    main()
