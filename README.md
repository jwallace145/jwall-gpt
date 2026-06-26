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
uv run python scripts/preprocess.py --corpus scripts/corpora/tiny_shakespeare.txt --out data/train.bin
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

## AWS training pipeline

GPU training workers are provisioned with Terraform under [`infra/`](infra/). Launch a run from **Actions → Train** (requires **`training` environment approval** before AWS spend).

See [infra/README.md](infra/README.md) for bootstrap, OIDC setup, and `terraform.tfvars` options.

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 James Wallace.
