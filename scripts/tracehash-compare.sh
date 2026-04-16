#!/usr/bin/env sh
set -eu

RUST_TRACE="${1:-/tmp/rust.tsv}"
C_TRACE="${2:-/tmp/c.tsv}"
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TRACEHASH_DIR="${TRACEHASH_DIR:-${SCRIPT_DIR}/../tools/tracehash}"

cargo run --manifest-path "${TRACEHASH_DIR}/Cargo.toml" --bin tracehash-compare -- \
  "${RUST_TRACE}" "${C_TRACE}"
