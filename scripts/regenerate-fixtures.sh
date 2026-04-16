#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

python3 "${SCRIPT_DIR}/generate-modern-fixtures.py"
