# Contributing to jwall-gpt

Thank you for contributing. This guide covers setup for human and AI contributors.

## Development environment

```bash
uv sync
```

Run commands inside the project environment with `uv run`, for example:

```bash
uv run pytest
uv run ruff check
uv run jwall-gpt-train --config configs/tiny.py
```

## Pre-commit hooks

Install all hook types once:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
uv run pre-commit install --hook-type commit-msg
```

### Tier 1 — every commit (fast)

- File hygiene (trailing whitespace, EOF, YAML/TOML validity, merge conflicts, private keys)
- `ruff format` and safe `ruff check --fix` on staged Python
- Conventional commit message validation via Commitizen
- **Single-line subject only** — no commit body; use the PR description for context

### Tier 2 — every push (mirrors CI)

- `uv lock --check`
- `uv run ruff check`
- `uv run ruff format --check`
- `uv run pytest --cov=jwall_gpt --cov-fail-under=80`
- `uv build`

Run the full push suite manually:

```bash
uv run pre-commit run --all-files --hook-stage push
```

### Tier 3 — GitHub CI

Pull requests and pushes to `main` run the same checks as Tier 2 in `.github/workflows/ci.yml`.

## Conventional Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages and PR titles (squash merge uses the PR title).

**Keep each commit to one line** — subject only, no body. The git log should stay scannable; put design notes and rationale in the PR description.

| Prefix | Release impact | Example |
|--------|----------------|---------|
| `feat:` | Minor version bump | `feat: add cosine LR schedule` |
| `fix:` | Patch version bump | `fix: correct causal attention mask` |
| `perf:` | Patch version bump | `perf: fuse QKV projection` |
| `feat!:` or `BREAKING CHANGE:` | Major version bump | `feat!: rename GPTConfig fields` |
| `chore:`, `ci:`, `docs:`, `test:`, `refactor:` | No release | `chore: update uv lock` |

Examples:

```
feat: implement causal self-attention block
fix: handle empty corpus in preprocess script
docs: expand training loop README section
```

## Pull request process

1. Create a branch from `main`
2. Make changes and ensure `uv run pre-commit run --all-files --hook-stage push` passes
3. Open a PR with a conventional commit title
4. Wait for CI to pass
5. Squash merge to `main`

Pull requests that change `infra/**` also trigger a **Terraform Plan** workflow that comments the plan on the PR. Review infrastructure changes before merging.

## AWS infrastructure

Terraform lives in [`infra/`](../infra/). See [`infra/README.md`](../infra/README.md) for bootstrap and repository variable setup.

- Use `terraform.tfvars` locally (never commit secrets; `terraform.tfvars` is gitignored)
- Prefer changing trainer sizing via `instance_type`, `use_spot_instances`, and related variables in `terraform.tfvars`
- GitHub Actions uses OIDC (`AWS_ROLE_ARN`) — do not add long-lived AWS access keys to the repository
- AWS resource IDs are stored in SSM Parameter Store after `terraform apply`, not in the repo

## Releases

Do **not** hand-edit `project.version` in `pyproject.toml` or `CHANGELOG.md` for releases.

[python-semantic-release](https://python-semantic-release.readthedocs.io/) runs on push to `main`, bumps the version from commit history, updates the changelog, tags, and publishes a GitHub Release.

## Agent contributor notes

- Run the full push-tier pre-commit suite before opening a PR
- Use conventional commit format in the PR title
- Do not commit `data/`, `checkpoints/`, `.venv/`, or secrets
- Prefer focused diffs that match existing code style
- Add or update tests when changing `jwall_gpt` package code

## What not to commit

- Training data shards (`data/`, `*.bin`)
- Model checkpoints (`checkpoints/`)
- Virtual environments (`.venv/`)
- API keys, credentials, or `.env` files
