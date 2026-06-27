# jwall-gpt AWS Infrastructure

Terraform-managed AWS resources for launching on-demand GPU training workers from GitHub Actions.

## Architecture

```text
GitHub Actions (OIDC)
        │
        ├─ PR: terraform plan → PR comment
        └─ workflow_dispatch: ec2 run-instances
                │
                ▼
        EC2 trainer (launch template)
        clone release → uv sync → train → S3 checkpoints → shutdown
```

### Modules

| Module | Purpose |
|--------|---------|
| [`modules/vpc`](modules/vpc) | VPC, public/private subnets, IGW, optional NAT + S3 endpoint |
| [`modules/storage`](modules/storage) | Training, datasets, and Terraform state S3 buckets |
| [`modules/trainer`](modules/trainer) | GPU launch template, instance profile, security group, SSM parameters |
| [`modules/github_oidc`](modules/github_oidc) | GitHub OIDC provider + IAM role for Actions |

### Runtime config (SSM Parameter Store)

Terraform writes AWS resource IDs to **SSM Parameter Store** under `/${project_name}/` (e.g. `/jwall-gpt/launch-template-id`). The **Train** workflow reads these at runtime — nothing sensitive is committed to the public repo.

| Parameter | Purpose |
|-----------|---------|
| `/jwall-gpt/launch-template-id` | EC2 launch template for trainers |
| `/jwall-gpt/trainer-subnet-id` | Subnet for worker instances |
| `/jwall-gpt/trainer-instance-profile` | IAM instance profile name |
| `/jwall-gpt/trainer-assign-public-ip` | Whether to assign a public IP |
| `/jwall-gpt/training-bucket` | S3 bucket for run outputs (checkpoints, logs) |
| `/jwall-gpt/datasets-bucket` | S3 bucket for tokenized datasets (read-only inputs) |
| `/jwall-gpt/github-actions-role-arn` | OIDC role ARN (reference) |
| `/jwall-gpt/aws-region` | Deployed AWS region |
| `/jwall-gpt/terraform-state-bucket` | Terraform state bucket name |

## Bootstrap (one-time)

### 1. Configure variables

Edit [`terraform.tfvars`](terraform.tfvars) (committed) or copy from [`terraform.tfvars.example`](terraform.tfvars.example) if starting fresh.

### 2. Initial apply (local credentials)

Requires **Terraform 1.11.0** — pinned in [`.terraform-version`](../.terraform-version) at the repo root (used by CI, [tfenv](https://github.com/tfutils/tfenv), and [asdf](https://asdf-vm.com/)). `infra/main.tf` enforces `>= 1.11.0` for S3 native lockfiles (`use_lockfile`).

```bash
cd infra
terraform init
terraform apply
```

This creates infrastructure and populates SSM parameters.

Note the outputs:

- `github_actions_role_arn`
- `terraform_state_bucket_name`

### 3. Enable remote state

[`backend.tf`](backend.tf) is committed with the S3 backend config. If migrating from local state for the first time:

```bash
terraform init -migrate-state
```

State locking uses **S3 lockfiles** (`use_lockfile = true`) — no DynamoDB table.

**GitHub OIDC provider:** Most AWS accounts already have `https://token.actions.githubusercontent.com` registered (one per account). The default `create_github_oidc_provider = false` reuses it. Set `true` only on a fresh account that has never configured GitHub Actions OIDC.

### 4. Configure GitHub repository variables

In **Settings → Secrets and variables → Actions → Variables**, set:

| Variable | Example |
|----------|---------|
| `AWS_ROLE_ARN` | `arn:aws:iam::123456789012:role/jwall-gpt-github-actions` (from `terraform output github_actions_role_arn`) |
| `AWS_REGION` | `us-east-1` |
| `TF_STATE_BUCKET` | `jwall-gpt-terraform-state` |

After this, pull requests that touch `infra/**` will run `terraform plan` and post the result as a PR comment. When those changes merge to `main`, the **Terraform Apply** workflow runs `terraform apply` automatically (path-filtered — only when `infra/**` changes).

## Launch a training run

### One-time: create the `training` environment

1. Open **Settings → Environments → New environment**
2. Name it **`training`** (must match the workflow)
3. Enable **Required reviewers** and add yourself
4. Save

The `launch-trainer` job waits for approval before assuming AWS credentials or starting EC2.

### Run training

1. Open **Actions → Train → Run workflow**
2. Review the **prepare** job summary (release tag, dataset, config)
3. Approve the pending **training** environment deployment
4. Leave **release_tag** empty to use the latest GitHub Release (default)
5. Set inputs as needed:
   - `dataset` — registered dataset key to train on (default `tinystories`)
   - `tokenizer` — tokenizer the dataset was prepared with (default `gpt2`)
   - `training_config` — model/training config (default `configs/tiny.py`; use `configs/nano.py` or `configs/small.py` for larger runs)
   - `max_steps` — optional override of the config's step count for longer runs

The worker pulls the tokenized dataset from
`s3://<datasets-bucket>/<dataset>/<tokenizer>/{train,val}.bin`, trains, uploads checkpoints to
S3, and shuts down. Datasets must be prepared and uploaded first with
`scripts/prepare_dataset.py` (see the repo README).

Checkpoints path: `s3://<training-bucket>/checkpoints/<tag>/<config>/<dataset>/<run-id>/` (the
trained model is `latest.pt`, run metadata/metrics are in `run.json`; `<run-id>` is the GitHub
Actions `run_id-run_attempt`).

## Key variables (`terraform.tfvars`)

| Block | Field | Description |
|-------|-------|-------------|
| `github_details` | `github_org`, `github_repo` | OIDC trust policy and tagging |
| `aws_account.network` | `use_private_subnet` | `false` = public subnet + public IP; `true` = private + NAT |
| `aws_account.training_compute` | `instance_type` | GPU instance size (e.g. `g4dn.xlarge`) |
| `aws_account.training_compute` | `use_spot_instances` | Optional; default `false` (on-demand). Set `true` for Spot when scaling |
| `aws_account.training_compute` | `spot_max_price` | Optional spot bid cap when `use_spot_instances = true` |
| `aws_account.training_compute` | `root_volume_size_gb` | EBS root volume size |
| `aws_account.storage` | `checkpoint_retention_days` | Expire checkpoints after N days (default 90) |
| `aws_account.storage` | `log_retention_days` | Expire training logs after N days (default 30) |
| `aws_account.storage` | `dataset_noncurrent_days` | Expire superseded dataset versions (default 30) |
| `aws_account.storage` | `state_noncurrent_days` | Expire superseded Terraform state versions (default 90) |
| `aws_account.storage` | `abort_multipart_days` | Abort incomplete multipart uploads (default 7) |

## Networking modes

**Public subnet (default)** — trainers get a public IP for egress. Simplest setup.

**Private subnet** — trainers run without public IPs. NAT gateway provides egress; S3 gateway endpoint reduces data transfer costs.

## Security

- GitHub Actions assumes an IAM role via OIDC (no long-lived AWS keys in GitHub)
- The **Train** workflow requires manual approval via the **`training`** environment before any AWS spend
- Trainer instances use a dedicated instance profile (S3 + CloudWatch only)
- State and training buckets block public access and use SSE-S3

## Local commands

```bash
cd infra
terraform fmt -recursive
terraform validate
terraform plan
```

After infrastructure changes, re-apply so SSM parameters stay in sync.
