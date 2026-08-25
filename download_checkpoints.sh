#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
RELEASE_URL="https://github.com/Jiang-Muyun/LEAF/releases/download/v1.0"

mkdir -p "${CHECKPOINT_DIR}"

wget -c -P "${CHECKPOINT_DIR}" "${RELEASE_URL}/leaf-v1.0-pretrain.ckpt"
wget -c -P "${CHECKPOINT_DIR}" "${RELEASE_URL}/leaf-v1.0-instruct-mpnet-base.ckpt"

echo "Checkpoints downloaded to ${CHECKPOINT_DIR}"
