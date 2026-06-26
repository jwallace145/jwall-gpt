#!/usr/bin/env bash
# Bootstrap script run on EC2 training workers via user-data.
set -euo pipefail

RELEASE_TAG="${RELEASE_TAG}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/tiny.py}"
TRAINING_BUCKET="${TRAINING_BUCKET}"
GITHUB_REPO="${GITHUB_REPO:-jwallace145/jwall-gpt}"
PROJECT_NAME="${PROJECT_NAME:-jwall-gpt}"
MAX_STEPS="${MAX_STEPS:-}"

if [[ -z "${RELEASE_TAG}" || -z "${TRAINING_BUCKET}" ]]; then
  echo "RELEASE_TAG and TRAINING_BUCKET are required"
  exit 1
fi

exec > >(tee "/var/log/${PROJECT_NAME}-training.log") 2>&1
echo "Starting ${PROJECT_NAME} training worker"
echo "Release: ${RELEASE_TAG}"
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
aws s3 sync "s3://${TRAINING_BUCKET}/data/" data/ || true

if [[ -f data/train.bin ]]; then
  echo "Using existing data/train.bin from S3"
else
  uv run python scripts/preprocess.py \
    --corpus scripts/corpora/tiny_shakespeare.txt \
    --out data/train.bin
  aws s3 cp data/train.bin "s3://${TRAINING_BUCKET}/data/train.bin"
fi

if [[ -n "${MAX_STEPS}" ]]; then
  echo "max_steps override not yet implemented in train.py; ignoring MAX_STEPS=${MAX_STEPS}"
fi

uv run jwall-gpt-train --config "${TRAINING_CONFIG}"

aws s3 sync checkpoints/ "s3://${TRAINING_BUCKET}/checkpoints/${RELEASE_TAG}/"
aws s3 cp "/var/log/${PROJECT_NAME}-training.log" \
  "s3://${TRAINING_BUCKET}/logs/${RELEASE_TAG}-$(date -u +%Y%m%dT%H%M%SZ).log"

echo "Training complete. Shutting down."
shutdown -h now
