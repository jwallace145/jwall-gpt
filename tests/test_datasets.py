from __future__ import annotations

import pytest

from jwall_gpt.data.datasets import (
    DATASETS,
    dataset_s3_key,
    dataset_s3_uri,
    get_dataset_spec,
)


def test_get_dataset_spec_known() -> None:
    spec = get_dataset_spec("tinystories")
    assert spec.hf_path == "roneneldan/TinyStories"
    assert spec.text_column == "text"
    assert spec.val_split == "validation"


def test_get_dataset_spec_unknown_lists_available() -> None:
    with pytest.raises(KeyError, match="Unknown dataset 'nope'"):
        get_dataset_spec("nope")


def test_registry_keys_match_spec_keys() -> None:
    for key, spec in DATASETS.items():
        assert key == spec.key


def test_dataset_s3_key_layout() -> None:
    assert dataset_s3_key("tinystories", "gpt2", "train.bin") == "tinystories/gpt2/train.bin"


def test_dataset_s3_uri_layout() -> None:
    uri = dataset_s3_uri("jwall-gpt-datasets", "tinystories", "gpt2", "val.bin")
    assert uri == "s3://jwall-gpt-datasets/tinystories/gpt2/val.bin"
