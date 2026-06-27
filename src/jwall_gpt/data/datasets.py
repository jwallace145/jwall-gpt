from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Describes how to load and tokenize a Hugging Face text dataset.

    ``val_split`` of ``None`` means the dataset ships no validation split, so a
    contiguous tail of the training tokens is held out instead (see
    ``jwall_gpt.data.prepare.split_tokens``).
    """

    key: str
    hf_path: str
    hf_name: str | None
    text_column: str
    train_split: str
    val_split: str | None


DATASETS: dict[str, DatasetSpec] = {
    "tinystories": DatasetSpec(
        key="tinystories",
        hf_path="roneneldan/TinyStories",
        hf_name=None,
        text_column="text",
        train_split="train",
        val_split="validation",
    ),
    "wikitext103": DatasetSpec(
        key="wikitext103",
        hf_path="Salesforce/wikitext",
        hf_name="wikitext-103-raw-v1",
        text_column="text",
        train_split="train",
        val_split="validation",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        available = ", ".join(sorted(DATASETS))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise KeyError(msg) from None


def dataset_s3_key(dataset: str, tokenizer: str, filename: str) -> str:
    """S3 object key for a tokenized shard, e.g. ``tinystories/gpt2/train.bin``."""
    return f"{dataset}/{tokenizer}/{filename}"


def dataset_s3_uri(bucket: str, dataset: str, tokenizer: str, filename: str) -> str:
    return f"s3://{bucket}/{dataset_s3_key(dataset, tokenizer, filename)}"
