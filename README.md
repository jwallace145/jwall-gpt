# jwall-gpt

[![CI](https://github.com/jwallace145/jwall-gpt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jwallace145/jwall-gpt/actions/workflows/ci.yml)
[![Release](https://github.com/jwallace145/jwall-gpt/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/jwallace145/jwall-gpt/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/jwallace145/jwall-gpt?display_name=tag&sort=semver)](https://github.com/jwallace145/jwall-gpt/releases/latest)
[![License](https://img.shields.io/github/license/jwallace145/jwall-gpt)](https://github.com/jwallace145/jwall-gpt/blob/main/LICENSE)

Build, train, and scale a GPT from scratch in PyTorch — an educational hobby project.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/preprocess.py --corpus scripts/corpora/tiny_shakespeare.txt --out data/train.bin  # also writes data/val.bin
uv run jwall-gpt-train --config configs/tiny.py
uv run jwall-gpt-sample --checkpoint checkpoints/latest.pt --prompt "ROMEO:"
```

## Development

```bash
uv sync
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
uv run pre-commit install --hook-type commit-msg
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions, quality gates, and the release process.

## Datasets

Tiny Shakespeare (`scripts/preprocess.py`) is the built-in smoke-test corpus. For larger
training runs, `scripts/prepare_dataset.py` downloads a Hugging Face dataset, tokenizes it
with GPT-2 BPE, and writes `train.bin` / `val.bin`. It needs the optional `data` dependency
group:

```bash
# Tokenize TinyStories and upload to s3://jwall-gpt-datasets/tinystories/gpt2/
uv run --group data python scripts/prepare_dataset.py --dataset tinystories --upload
```

Registered datasets live in [`src/jwall_gpt/data/datasets.py`](src/jwall_gpt/data/datasets.py)
(currently `tinystories`, `wikitext103`). Tokenized shards are stored durably in the
`jwall-gpt-datasets` S3 bucket under `<dataset>/<tokenizer>/`, and GPU trainers read them
back at run time. Tokenization runs locally for now; a dedicated AWS job is planned for very
large datasets.

## AWS training pipeline

GPU training workers are provisioned with Terraform under [`infra/`](infra/). Launch a run from
**Actions → Train** (requires **`training` environment approval** before AWS spend). Pick the
`dataset` (e.g. `tinystories`), `training_config` (e.g. `configs/small.py`), and an optional
`max_steps` override. The worker pulls the tokenized dataset from the `jwall-gpt-datasets`
bucket, trains, and uploads the result.

The trained model lands at `s3://<training-bucket>/checkpoints/<dataset>/<release-tag>/latest.pt`.
Download it and prompt the model locally:

```bash
aws s3 cp s3://jwall-gpt-training/checkpoints/tinystories/v0.7.0/latest.pt checkpoints/latest.pt
uv run jwall-gpt-sample --checkpoint checkpoints/latest.pt --prompt "Once upon a time"
```

See [infra/README.md](infra/README.md) for bootstrap, OIDC setup, and `terraform.tfvars` options.

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 James Wallace.
