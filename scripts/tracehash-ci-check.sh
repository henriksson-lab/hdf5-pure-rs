#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

TRACEHASH_WORKDIR="${TRACEHASH_WORKDIR:-}"
if [ -z "${TRACEHASH_WORKDIR}" ]; then
  TRACEHASH_WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/hdf5-tracehash-ci.XXXXXX")"
  TRACEHASH_CLEANUP=1
else
  mkdir -p "${TRACEHASH_WORKDIR}"
  TRACEHASH_CLEANUP=0
fi

cleanup() {
  if [ "${TRACEHASH_CLEANUP}" = 1 ]; then
    rm -rf "${TRACEHASH_WORKDIR}"
  fi
}
trap cleanup EXIT INT TERM

RUST_TRACE="${TRACEHASH_WORKDIR}/rust.tsv"
C_TRACE="${TRACEHASH_WORKDIR}/c.tsv"

cd "${REPO_DIR}"

echo "tracehash-ci: writing Rust trace to ${RUST_TRACE}"
TRACEHASH_OUT="${RUST_TRACE}" "${SCRIPT_DIR}/tracehash-rust-corpus.sh"

echo "tracehash-ci: writing C trace to ${C_TRACE}"
TRACEHASH_OUT="${C_TRACE}" "${SCRIPT_DIR}/tracehash-c-corpus.sh"

echo "tracehash-ci: comparing supported corpus"
"${SCRIPT_DIR}/tracehash-compare.sh" "${RUST_TRACE}" "${C_TRACE}"
