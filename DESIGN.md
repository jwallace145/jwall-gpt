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

## Scaling path

Tiny (~10M) → Nano (~25–50M) → Small (~124M) → beyond as hardware and budget allow.

## AWS training

Terraform in [`infra/`](../infra/) provisions:

- VPC (public or private subnet for trainers)
- S3 buckets (training data, checkpoints, Terraform state with S3 lockfiles)
- GPU trainer launch template (on-demand by default; optional Spot)
- GitHub OIDC IAM role for Actions
- SSM Parameter Store entries for runtime IDs (not committed to git)

GitHub workflows:

- **Terraform Plan** — posts `terraform plan` on PRs touching `infra/`
- **Terraform Apply** — runs `terraform apply` on push to `main` when `infra/` changes
- **Train** — `workflow_dispatch` to launch a worker; requires **`training` environment approval** before AWS resources are created
