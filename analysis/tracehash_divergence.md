# Tracehash Divergence Tracking

This project vendors the tracehash toolkit at:

```text
tools/tracehash
```

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
- `hdf5.data_layout.decode`: `DataLayoutMessage::decode`
- `hdf5.filter_pipeline.decode`: `FilterPipelineMessage::decode`
- `hdf5.object_header.read`: `ObjectHeader::read_at`

## C-Side Probe Targets

Matching probes are checked into the vendored HDF5 C tree behind
`H5_HAVE_TRACEHASH`:

- `hdf5.datatype.decode`: `hdf5/src/H5Odtype.c`
- `hdf5.data_layout.decode`: `hdf5/src/H5Olayout.c`
- `hdf5.filter_pipeline.decode`: `hdf5/src/H5Opline.c`
- `hdf5.filter_pipeline.apply`: `hdf5/src/H5Z.c`
- `hdf5.object_header.read`: `hdf5/src/H5Ocache.c`
- `hdf5.chunk_index.fixed_array.lookup`: `hdf5/src/H5Dfarray.c`
- `hdf5.chunk_index.extensible_array.lookup`: `hdf5/src/H5Dearray.c`
- `hdf5.chunk_index.btree2.lookup`: `hdf5/src/H5Dbtree2.c`
- `hdf5.fractal_heap.read`: `hdf5/src/H5HF.c`

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
- Add writer-side probes for mutable resize and chunk-index updates before
  adding writer-mutated files to the default tracehash compare.
- SZip and third-party filters remain outside the pure-Rust supported surface.
