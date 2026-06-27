# jwall-gpt Design

Educational GPT built from scratch in PyTorch. See the project plan for the full roadmap.

## Architecture

Decoder-only transformer (GPT-2 style):

- **Tokenizer**: GPT-2 BPE via `tiktoken`
- **Embeddings**: token + learned positional embeddings
- **Attention**: multi-head causal self-attention
- **FFN**: 2-layer MLP with GELU (`4 * n_embd` hidden dim)
- **Block**: pre-norm residual (`LN → Attn → + → LN → FFN → +`)
- **Head**: linear projection to vocabulary (weight-tied with token embeddings)

## Phase 1 config (tiny, ~10M params)

| Parameter | Value |
|-----------|-------|
| `n_layer` | 6 |
| `n_head` | 6 |
| `n_embd` | 384 |
| `block_size` | 256 |
| `vocab_size` | 50257 |

## Training

- Optimizer: AdamW (β1=0.9, β2=0.95, weight decay=0.1)
- LR schedule: cosine with warmup
- Loss: cross-entropy on next-token prediction
- Gradient clipping: max norm 1.0

## Data and evaluation

- `scripts/preprocess.py` tokenizes the corpus (GPT-2 BPE) and writes a contiguous
  train/val split (`data/train.bin`, `data/val.bin`; default `--val-frac 0.1`)
- Every `eval_interval` steps, training reports held-out **validation loss** averaged
  over `eval_iters` batches alongside train loss — the gap between the two is the
  signal for under/overfitting

## Scaling path

Tiny (~10M) → Nano (~25–50M) → Small (~124M) → beyond as hardware and budget allow.

## AWS training

Terraform in [`infra/`](../infra/) provisions:

- VPC (public or private subnet for trainers)
- S3 buckets (with cost-saving lifecycle rules):
  - `jwall-gpt-datasets` — tokenized datasets (durable inputs; expire only old versions)
  - `jwall-gpt-training` — run outputs (checkpoints expire after 90d, logs after 30d)
  - `jwall-gpt-terraform-state` — Terraform state with S3 lockfiles (trim old versions)
  - All buckets abort incomplete multipart uploads after 7 days
- GPU trainer launch template (on-demand by default; optional Spot)
- GitHub OIDC IAM role for Actions
- SSM Parameter Store entries for runtime IDs (not committed to git)

GitHub workflows:

- **Terraform Plan** — posts `terraform plan` on PRs touching `infra/`
- **Terraform Apply** — runs `terraform apply` on push to `main` when `infra/` changes
- **Train** — `workflow_dispatch` to launch a worker; pulls the chosen tokenized dataset from the datasets bucket, trains, and writes checkpoints to `checkpoints/<tag>/<config>/<dataset>/<run-id>/` (run-id is the GitHub Actions `run_id-run_attempt`, so each run is isolated and traceable). Each run also drops a `run.json` metadata/metrics file alongside the checkpoint for post-run analysis. Requires **`training` environment approval** before AWS resources are created
