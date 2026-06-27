#!/usr/bin/env bash
# Bootstrap script run on EC2 training workers via user-data.
set -euo pipefail

RELEASE_TAG="${RELEASE_TAG}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/tiny.py}"
TRAINING_BUCKET="${TRAINING_BUCKET}"
DATASETS_BUCKET="${DATASETS_BUCKET}"
DATASET="${DATASET:-tinystories}"
TOKENIZER="${TOKENIZER:-gpt2}"
GITHUB_REPO="${GITHUB_REPO:-jwallace145/jwall-gpt}"
PROJECT_NAME="${PROJECT_NAME:-jwall-gpt}"
MAX_STEPS="${MAX_STEPS:-}"

if [[ -z "${RELEASE_TAG}" || -z "${TRAINING_BUCKET}" || -z "${DATASETS_BUCKET}" ]]; then
  echo "RELEASE_TAG, TRAINING_BUCKET, and DATASETS_BUCKET are required"
  exit 1
fi

exec > >(tee "/var/log/${PROJECT_NAME}-training.log") 2>&1
echo "Starting ${PROJECT_NAME} training worker"
echo "Release: ${RELEASE_TAG}"
echo "Dataset: ${DATASET} (${TOKENIZER})"
echo "Config: ${TRAINING_CONFIG}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git curl awscli

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:${PATH}"

WORK_DIR="/opt/${PROJECT_NAME}"
rm -rf "${WORK_DIR}"
git clone --depth 1 --branch "${RELEASE_TAG}" "https://github.com/${GITHUB_REPO}.git" "${WORK_DIR}"
cd "${WORK_DIR}"

uv sync --frozen

mkdir -p data checkpoints
DATA_PREFIX="s3://${DATASETS_BUCKET}/${DATASET}/${TOKENIZER}"
echo "Downloading tokenized dataset from ${DATA_PREFIX}"
aws s3 cp "${DATA_PREFIX}/train.bin" data/train.bin
aws s3 cp "${DATA_PREFIX}/val.bin" data/val.bin

TRAIN_ARGS=(--config "${TRAINING_CONFIG}")
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_ARGS+=(--max-steps "${MAX_STEPS}")
fi
uv run jwall-gpt-train "${TRAIN_ARGS[@]}"

CHECKPOINT_PREFIX="s3://${TRAINING_BUCKET}/checkpoints/${DATASET}/${RELEASE_TAG}"
aws s3 sync checkpoints/ "${CHECKPOINT_PREFIX}/"
aws s3 cp "/var/log/${PROJECT_NAME}-training.log" \
  "s3://${TRAINING_BUCKET}/logs/${DATASET}/${RELEASE_TAG}-$(date -u +%Y%m%dT%H%M%SZ).log"

echo "Training complete. Checkpoints at ${CHECKPOINT_PREFIX}/. Shutting down."
shutdown -h now
