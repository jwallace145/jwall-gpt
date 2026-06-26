# jwall-gpt

[![Release](https://github.com/jimmymwallace/jwall-gpt/actions/workflows/release.yml/badge.svg)](https://github.com/jimmymwallace/jwall-gpt/actions/workflows/release.yml)
[![GitHub release](https://img.shields.io/github/v/release/jimmymwallace/jwall-gpt)](https://github.com/jimmymwallace/jwall-gpt/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 James Wallace.
