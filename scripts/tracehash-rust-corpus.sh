#!/usr/bin/env sh
set -eu

TRACEHASH_OUT="${TRACEHASH_OUT:-/tmp/rust.tsv}"
TRACEHASH_SIDE="${TRACEHASH_SIDE:-rust}"
TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID:-hdf5-corpus}"
TRACEHASH_CORPUS_FILES="${TRACEHASH_CORPUS_FILES:-tests/data/simple_v0.h5 tests/data/simple_v2.h5 tests/data/datasets_v3.h5 tests/data/compound.h5 tests/data/dense_links.h5 tests/data/dense_attrs.h5 tests/data/strings.h5 tests/data/hdf5_ref/all_dtypes.h5 tests/data/hdf5_ref/fractal_heap_modern.h5 tests/data/hdf5_ref/v4_fixed_array_chunks.h5 tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5 tests/data/hdf5_ref/v4_extensible_array_chunks.h5 tests/data/hdf5_ref/v4_extensible_array_spillover.h5 tests/data/hdf5_ref/v4_btree2_chunks.h5 tests/data/hdf5_ref/v4_btree2_internal_chunks.h5 tests/data/hdf5_ref/nbit_filter_i32.h5 tests/data/hdf5_ref/scaleoffset_filter_i32.h5}"

export TRACEHASH_SIDE TRACEHASH_RUN_ID

# `tracehash-rs` 0.1 opens `TRACEHASH_OUT` with `File::create`, which truncates
# the file on every process start. Our corpus walker runs as a separate
# `cargo run` per fixture, so we stage each fixture's rows in its own file
# and concatenate them into the final merged trace at the end.
SHARD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hdf5-tracehash.XXXXXX")"
trap 'rm -rf "${SHARD_DIR}"' EXIT INT TERM

TRACEHASH_RUN_ID_BASE="${TRACEHASH_RUN_ID}"
i=0
for f in ${TRACEHASH_CORPUS_FILES}; do
  i=$((i + 1))
  shard="${SHARD_DIR}/shard-$(printf '%04d' "$i").tsv"
  TRACEHASH_OUT="${shard}" \
    TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID_BASE}:${f}" \
    cargo run --features tracehash --bin tracehash_corpus -- "$f"
done

: > "${TRACEHASH_OUT}"
for shard in "${SHARD_DIR}"/shard-*.tsv; do
  [ -f "${shard}" ] || continue
  cat "${shard}" >> "${TRACEHASH_OUT}"
done

echo "wrote Rust trace to ${TRACEHASH_OUT}"
