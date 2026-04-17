# Tracehash Divergence Tracking

The Rust side pulls the tracehash library from crates.io
([`tracehash-rs`](https://crates.io/crates/tracehash-rs)). The C-side
helpers (`tracehash_c.c`, `tracehash_c.h`) that a patched libhdf5 build
links against ship inside the same crate under `c/`; for local use the
crate can be checked out at `tools/tracehash` (gitignored) so the
patched HDF5 sources can `#include "../../tools/tracehash/c/tracehash_c.h"`.

The optional Cargo feature is disabled by default:

```sh
cargo test --features tracehash
```

When enabled, Rust probes emit rows if `TRACEHASH_OUT` is set:

```sh
TRACEHASH_OUT=/tmp/rust.tsv TRACEHASH_SIDE=rust TRACEHASH_RUN_ID=hdf5-corpus \
  cargo test --features tracehash
```

Compare Rust and C traces with:

```sh
scripts/tracehash-compare.sh /tmp/rust.tsv /tmp/c.tsv
```

Current local run status is recorded in:

```text
analysis/tracehash_divergence_report.md
```

## Rust Probe Names

- `hdf5.datatype.decode`: `DatatypeMessage::decode`
- `hdf5.datatype.properties`: `DatatypeMessage::decode`
- `hdf5.dataspace.extent`: `DataspaceMessage::decode`
- `hdf5.selection.deserialize`: VDS `Dataset::decode_virtual_selection`
- `hdf5.fill_value.decode`: `FillValueMessage::decode` and
  `FillValueMessage::decode_old`
- `hdf5.chunk_index.btree1.lookup`: v1 B-tree chunk lookup in `Dataset`
- `hdf5.chunk_index.btree2.lookup`: v2 B-tree chunk lookup in `Dataset`
- `hdf5.chunk_index.btree2.record_decode`: v2 B-tree chunk record decode in
  `Dataset`
- `hdf5.chunk_index.btree2.internal_traverse`: v2 B-tree chunk internal child
  pointer decode in `btree_v2`
- `hdf5.chunk_index.fixed_array.lookup`: fixed-array chunk address lookup in
  `Dataset`
- `hdf5.chunk_index.extensible_array.lookup`: extensible-array chunk address
  lookup in `Dataset`
- `hdf5.fractal_heap.managed_object`: managed fractal-heap object block
  resolution in `FractalHeapHeader`
- `hdf5.fractal_heap.huge_object`: huge fractal-heap object address/size
  resolution in `FractalHeapHeader`
- `hdf5.fractal_heap.tiny_object`: tiny fractal-heap object decode in
  `FractalHeapHeader`
- `hdf5.global_heap.deref`: global heap object dereference in
  `read_global_heap_object`
- `hdf5.vlen.read`: variable-length string heap payload reads in `Dataset` and
  the tracehash corpus walker
- `hdf5.external_link.resolve`: external-link filename/object-path target
  decode in `LinkMessage`
- `hdf5.vds.source.resolve`: virtual dataset mapping source filename/dataset
  target decode in `Dataset`
- `hdf5.data_layout.decode`: `DataLayoutMessage::decode`
- `hdf5.filter_pipeline.decode`: `FilterPipelineMessage::decode`
- `hdf5.object_header.read`: `ObjectHeader::read_at`

## C-Side Probe Targets

Matching probes are checked into the vendored HDF5 C tree behind
`H5_HAVE_TRACEHASH`:

- `hdf5.datatype.decode`: `hdf5/src/H5Odtype.c`
- `hdf5.datatype.properties`: `hdf5/src/H5Odtype.c`
- `hdf5.dataspace.extent`: `hdf5/src/H5Osdspace.c`
- `hdf5.selection.deserialize`: `hdf5/src/H5Sselect.c`
- `hdf5.fill_value.decode`: `hdf5/src/H5Ofill.c`
- `hdf5.chunk_index.btree1.lookup`: `hdf5/src/H5Dbtree.c`
- `hdf5.data_layout.decode`: `hdf5/src/H5Olayout.c`
- `hdf5.filter_pipeline.decode`: `hdf5/src/H5Opline.c`
- `hdf5.filter_pipeline.apply`: `hdf5/src/H5Z.c`
- `hdf5.object_header.read`: `hdf5/src/H5Ocache.c`
- `hdf5.chunk_index.fixed_array.lookup`: `hdf5/src/H5Dfarray.c`
- `hdf5.chunk_index.extensible_array.lookup`: `hdf5/src/H5Dearray.c`
- `hdf5.chunk_index.btree2.lookup`: `hdf5/src/H5Dbtree2.c`
- `hdf5.chunk_index.btree2.record_decode`: `hdf5/src/H5Dbtree2.c`
- `hdf5.chunk_index.btree2.internal_traverse`: `hdf5/src/H5B2cache.c`
- `hdf5.fractal_heap.read`: `hdf5/src/H5HF.c`
- `hdf5.fractal_heap.managed_object`: `hdf5/src/H5HFman.c`
- `hdf5.fractal_heap.huge_object`: `hdf5/src/H5HFhuge.c`
- `hdf5.fractal_heap.tiny_object`: `hdf5/src/H5HFtiny.c`
- `hdf5.global_heap.deref`: `hdf5/src/H5HG.c`
- `hdf5.vlen.read`: `hdf5/src/H5Tvlen.c`
- `hdf5.external_link.resolve`: `tools/tracehash/c/hdf5_trace_corpus.c`
- `hdf5.vds.source.resolve`: `tools/tracehash/c/hdf5_trace_corpus.c`

Use the same probe name and the same canonical input/output field order. The
comparison key is:

```text
run_id + function + input_hash -> output_hash
```

`scripts/tracehash-rust-corpus.sh` emits `/tmp/rust.tsv`.
`scripts/tracehash-c-corpus.sh` emits `/tmp/c.tsv`. The C runner prefers the
patched public-API corpus driver at `/tmp/hdf5-trace-build/bin/hdf5_trace_corpus`
when it exists, and falls back to patched `h5dump` traversal otherwise.

## Priority Divergence Areas

- The default parity corpus remains read-side only and excludes virtual
  datasets for now. Adding `hdf5/tools/test/testfiles/vds/1_vds.h5` currently
  causes Rust to trace external source-file reads under the VDS run id while the
  patched C walker emits no matching VDS/source-file probe rows.
- Add dedicated virtual dataset mapping probes before enabling VDS fixtures in
  the default tracehash compare.
- `hdf5.selection.deserialize` is validated with a focused VDS-only filtered
  trace comparison because the default corpus intentionally excludes VDS source
  traversal.
- `hdf5.external_link.resolve` is validated with a focused filtered trace
  comparison over `tests/data/hdf5_ref/groups_and_links.h5`.
- `hdf5.vds.source.resolve` is validated with a focused filtered trace
  comparison over `tests/data/hdf5_ref/vds_same_file.h5`.
- Add writer-side probes for mutable resize and chunk-index updates before
  adding writer-mutated files to the default tracehash compare.
- SZip and third-party filters remain outside the pure-Rust supported surface.
