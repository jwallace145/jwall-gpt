#!/usr/bin/env bash
# Render EC2 user-data from the training worker bootstrap script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="${ROOT_DIR}/scripts/aws/train-worker.sh"

: "${RELEASE_TAG:?RELEASE_TAG is required}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/tiny.py}"
TRAINING_BUCKET="${TRAINING_BUCKET:?TRAINING_BUCKET is required}"
GITHUB_REPO="${GITHUB_REPO:-jwallace145/jwall-gpt}"
PROJECT_NAME="${PROJECT_NAME:-jwall-gpt}"
MAX_STEPS="${MAX_STEPS:-}"

export RELEASE_TAG TRAINING_CONFIG TRAINING_BUCKET GITHUB_REPO PROJECT_NAME MAX_STEPS

envsubst '$RELEASE_TAG $TRAINING_CONFIG $TRAINING_BUCKET $GITHUB_REPO $PROJECT_NAME $MAX_STEPS' < "${TEMPLATE}"
