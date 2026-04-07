#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="${1:-JAmanOG/social-media-optimizer}"
EXCLUDE_FILE="${SCRIPT_DIR}/.openenvignore"

cd "${SCRIPT_DIR}"
./venv/bin/openenv push . --repo-id "${REPO_ID}" --exclude "${EXCLUDE_FILE}"