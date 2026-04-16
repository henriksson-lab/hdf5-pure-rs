#!/usr/bin/env sh
set -eu

TRACEHASH_OUT="${TRACEHASH_OUT:-/tmp/c.tsv}"
TRACEHASH_SIDE="${TRACEHASH_SIDE:-c}"
TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID:-hdf5-corpus}"
TRACEHASH_CORPUS_FILES="${TRACEHASH_CORPUS_FILES:-tests/data/simple_v0.h5 tests/data/simple_v2.h5 tests/data/datasets_v3.h5 tests/data/compound.h5 tests/data/dense_links.h5 tests/data/dense_attrs.h5 tests/data/strings.h5 tests/data/hdf5_ref/all_dtypes.h5 tests/data/hdf5_ref/fractal_heap_modern.h5 tests/data/hdf5_ref/v4_fixed_array_chunks.h5 tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5 tests/data/hdf5_ref/v4_extensible_array_chunks.h5 tests/data/hdf5_ref/v4_extensible_array_spillover.h5 tests/data/hdf5_ref/v4_btree2_chunks.h5 tests/data/hdf5_ref/v4_btree2_internal_chunks.h5 tests/data/hdf5_ref/nbit_filter_i32.h5 tests/data/hdf5_ref/scaleoffset_filter_i32.h5}"
HDF5_TRACEHASH_H5DUMP="${HDF5_TRACEHASH_H5DUMP:-/tmp/hdf5-trace-build/bin/h5dump}"
HDF5_TRACEHASH_DRIVER="${HDF5_TRACEHASH_DRIVER:-/tmp/hdf5-trace-build/bin/hdf5_trace_corpus}"

if [ -z "${HDF5_TRACEHASH_CMD:-}" ]; then
  if [ -x "${HDF5_TRACEHASH_DRIVER}" ]; then
    HDF5_TRACEHASH_CMD='TRACEHASH_RUN_ID_BASE="${TRACEHASH_RUN_ID}"; for f in ${TRACEHASH_CORPUS_FILES}; do TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID_BASE}:$f" "${HDF5_TRACEHASH_DRIVER}" "$f" || exit 1; done'
  elif [ -x "${HDF5_TRACEHASH_H5DUMP}" ]; then
    HDF5_TRACEHASH_CMD='TRACEHASH_RUN_ID_BASE="${TRACEHASH_RUN_ID}"; for f in ${TRACEHASH_CORPUS_FILES}; do TRACEHASH_RUN_ID="${TRACEHASH_RUN_ID_BASE}:$f" "${HDF5_TRACEHASH_H5DUMP}" "$f" >/dev/null || exit 1; done'
  else
  cat >&2 <<'EOF'
Set HDF5_TRACEHASH_CMD to a patched HDF5 C corpus command.

The patched C build should include tools/tracehash/c/tracehash_c.c and probes
from analysis/tracehash_divergence.md. This runner exports:
  TRACEHASH_OUT=/tmp/c.tsv
  TRACEHASH_SIDE=c
  TRACEHASH_RUN_ID=hdf5-corpus

Alternatively set HDF5_TRACEHASH_H5DUMP to a patched h5dump binary. The
default is /tmp/hdf5-trace-build/bin/h5dump.
EOF
  exit 2
  fi
fi

export TRACEHASH_OUT TRACEHASH_SIDE TRACEHASH_RUN_ID TRACEHASH_CORPUS_FILES HDF5_TRACEHASH_H5DUMP HDF5_TRACEHASH_DRIVER

: > "${TRACEHASH_OUT}"

sh -c "${HDF5_TRACEHASH_CMD}"

echo "wrote C trace to ${TRACEHASH_OUT}"
