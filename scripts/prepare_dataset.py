#!/usr/bin/env python3
"""Download a Hugging Face text dataset, tokenize it, and write train/val .bin shards.

Optionally uploads the shards to the datasets S3 bucket under
``<dataset>/<tokenizer>/{train,val}.bin``.

Requires the ``data`` dependency group:

    uv run --group data python scripts/prepare_dataset.py --dataset tinystories --upload
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import tiktoken

from jwall_gpt.data.datasets import dataset_s3_uri, get_dataset_spec
from jwall_gpt.data.prepare import split_tokens

WRITE_BATCHES = 1024


def tokenize_dataset(dset, text_column: str, enc: tiktoken.Encoding, num_proc: int):
    eot = enc.eot_token

    def process(example: dict) -> dict:
        ids = enc.encode_ordinary(example[text_column])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    return dset.map(
        process,
        remove_columns=dset.column_names,
        num_proc=num_proc,
        desc="tokenizing",
    )


def write_bin(tokenized, path: Path) -> int:
    """Stream tokenized ``ids`` into a uint16 memmap to avoid holding all tokens in RAM."""
    arr_len = int(np.sum(tokenized["len"], dtype=np.int64))
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
    total_batches = max(1, min(WRITE_BATCHES, len(tokenized)))
    idx = 0
    for batch_idx in range(total_batches):
        batch = tokenized.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
    return arr_len


def upload(path: Path, uri: str) -> None:
    print(f"Uploading {path} -> {uri}")
    subprocess.run(["aws", "s3", "cp", str(path), uri], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Registered dataset key, e.g. tinystories")
    parser.add_argument("--tokenizer", default="gpt2", help="tiktoken encoding name")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Held-out fraction when the dataset has no validation split",
    )
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--bucket", default="jwall-gpt-datasets")
    parser.add_argument("--upload", action="store_true", help="Upload shards to S3 after writing")
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    enc = tiktoken.get_encoding(args.tokenizer)

    from datasets import load_dataset  # heavy import; only needed at run time

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    if spec.val_split is not None:
        train_raw = load_dataset(spec.hf_path, spec.hf_name, split=spec.train_split)
        val_raw = load_dataset(spec.hf_path, spec.hf_name, split=spec.val_split)
        train_tok = tokenize_dataset(train_raw, spec.text_column, enc, args.num_proc)
        val_tok = tokenize_dataset(val_raw, spec.text_column, enc, args.num_proc)
        n_train = write_bin(train_tok, train_path)
        n_val = write_bin(val_tok, val_path)
    else:
        train_raw = load_dataset(spec.hf_path, spec.hf_name, split=spec.train_split)
        tokenized = tokenize_dataset(train_raw, spec.text_column, enc, args.num_proc)
        all_tokens = np.concatenate(tokenized.with_format("numpy")["ids"]).astype(np.uint16)
        train_tokens, val_tokens = split_tokens(all_tokens, args.val_frac)
        train_tokens.tofile(train_path)
        val_tokens.tofile(val_path)
        n_train, n_val = len(train_tokens), len(val_tokens)

    print(f"Wrote {n_train:,} train tokens to {train_path}")
    print(f"Wrote {n_val:,} val tokens to {val_path}")

    if args.upload:
        upload(train_path, dataset_s3_uri(args.bucket, spec.key, args.tokenizer, "train.bin"))
        upload(val_path, dataset_s3_uri(args.bucket, spec.key, args.tokenizer, "val.bin"))
        print(f"Uploaded to s3://{args.bucket}/{spec.key}/{args.tokenizer}/ (train.bin, val.bin)")


if __name__ == "__main__":
    main()
