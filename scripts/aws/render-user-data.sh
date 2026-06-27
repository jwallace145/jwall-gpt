#!/usr/bin/env bash
# Render EC2 user-data from the training worker bootstrap script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="${ROOT_DIR}/scripts/aws/train-worker.sh"

: "${RELEASE_TAG:?RELEASE_TAG is required}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/tiny.py}"
TRAINING_BUCKET="${TRAINING_BUCKET:?TRAINING_BUCKET is required}"
DATASETS_BUCKET="${DATASETS_BUCKET:?DATASETS_BUCKET is required}"
DATASET="${DATASET:-tinystories}"
TOKENIZER="${TOKENIZER:-gpt2}"
GITHUB_REPO="${GITHUB_REPO:-jwallace145/jwall-gpt}"
PROJECT_NAME="${PROJECT_NAME:-jwall-gpt}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_ID="${RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"

export RELEASE_TAG TRAINING_CONFIG TRAINING_BUCKET DATASETS_BUCKET DATASET TOKENIZER
export GITHUB_REPO PROJECT_NAME MAX_STEPS RUN_ID

envsubst '$RELEASE_TAG $TRAINING_CONFIG $TRAINING_BUCKET $DATASETS_BUCKET $DATASET $TOKENIZER $GITHUB_REPO $PROJECT_NAME $MAX_STEPS $RUN_ID' < "${TEMPLATE}"
