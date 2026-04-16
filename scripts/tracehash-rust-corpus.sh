#!/usr/bin/env sh
set -eu

TRACEHASH_OUT="${TRACEHASH_OUT:-/tmp/rust.tsv}"
TRACEHASH_SIDE="${TRACEHASH_SIDE:-rust}"
TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID:-hdf5-corpus}"
TRACEHASH_CORPUS_FILES="${TRACEHASH_CORPUS_FILES:-tests/data/simple_v0.h5 tests/data/simple_v2.h5 tests/data/datasets_v3.h5 tests/data/compound.h5 tests/data/dense_links.h5 tests/data/dense_attrs.h5 tests/data/strings.h5 tests/data/hdf5_ref/all_dtypes.h5 tests/data/hdf5_ref/fractal_heap_modern.h5 tests/data/hdf5_ref/v4_fixed_array_chunks.h5 tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5 tests/data/hdf5_ref/v4_extensible_array_chunks.h5 tests/data/hdf5_ref/v4_extensible_array_spillover.h5 tests/data/hdf5_ref/v4_btree2_chunks.h5 tests/data/hdf5_ref/v4_btree2_internal_chunks.h5 tests/data/hdf5_ref/nbit_filter_i32.h5 tests/data/hdf5_ref/scaleoffset_filter_i32.h5}"

export TRACEHASH_OUT TRACEHASH_SIDE TRACEHASH_RUN_ID

: > "${TRACEHASH_OUT}"

TRACEHASH_RUN_ID_BASE="${TRACEHASH_RUN_ID}"
for f in ${TRACEHASH_CORPUS_FILES}; do
  TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID_BASE}:${f}" \
    cargo run --features tracehash --bin tracehash_corpus -- "$f"
done

echo "wrote Rust trace to ${TRACEHASH_OUT}"
