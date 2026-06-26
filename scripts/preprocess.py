#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken


def tokenize_corpus(text: str) -> np.ndarray:
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    return np.array(tokens, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a text corpus into a .bin shard")
    parser.add_argument("--corpus", type=Path, required=True, help="Input text file")
    parser.add_argument("--out", type=Path, required=True, help="Output .bin path")
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    tokens = tokenize_corpus(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(args.out)
    print(f"Wrote {len(tokens):,} tokens to {args.out}")


if __name__ == "__main__":
    main()
