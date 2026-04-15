# Tracehash Divergence Report Status

Date: 2026-04-15

## Local Runs

- Rust corpus runner: `scripts/tracehash-rust-corpus.sh`
- Rust trace output: `/tmp/rust.tsv`
- Rust trace rows observed locally: 15261
- C corpus runner: `scripts/tracehash-c-corpus.sh`
- C trace output: `/tmp/c.tsv`
- C trace rows observed locally: 15261

## Result

A patched vendored HDF5 C build was configured with CMake 4.3.1 and built in:

```text
/tmp/hdf5-trace-build
```

The build and comparison required these tracehash fixes:

- Move the `hdf5.fractal_heap.read` tracehash call into `H5HF_read`, where its
  input/output events are emitted.
- Add `tools/tracehash/c/tracehash_c.c` to the HDF5 static library sources so
  C corpus executables link the tracehash runtime symbols.
- Open C trace files in append mode and truncate them in
  `scripts/tracehash-c-corpus.sh`, because the C corpus runs one `h5dump`
  process per fixture.
- Add a small patched-C corpus driver, `tools/tracehash/c/hdf5_trace_corpus.c`,
  so C traversal uses object/attribute iteration and dataset reads instead of
  `h5dump` formatting behavior.
- Add a Rust `tracehash_corpus` binary so Rust and C use the same default
  fixture list instead of comparing the full Rust test suite against `h5dump`.
- Route the Rust `tracehash_corpus` walker through library internals so each
  object header is decoded once during corpus traversal, instead of through
  public API calls that reopen objects for type checks and dataset reads.
- Fix Rust v2 B-tree node sizing and fractal-heap direct-block offset handling
  so dense-link heaps with multiple direct blocks are fully traversed.
- Traverse dense attributes in the Rust corpus walker and tag trace rows with
  per-fixture run IDs to avoid cross-file address collisions.
- Add Rust probes for chunk-index lookups and filter application.
- Align Rust/C decode probe outputs for datatype, data layout, and filter
  pipeline and object-header messages.
- Avoid hashing the C layout message chunk-index union field for non-chunk
  layouts.

The local C and Rust corpus runners used this fixture set:

```text
tests/data/simple_v0.h5
tests/data/simple_v2.h5
tests/data/datasets_v3.h5
tests/data/compound.h5
tests/data/dense_links.h5
tests/data/dense_attrs.h5
tests/data/hdf5_ref/all_dtypes.h5
tests/data/hdf5_ref/fractal_heap_modern.h5
tests/data/hdf5_ref/v4_fixed_array_chunks.h5
tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5
tests/data/hdf5_ref/v4_extensible_array_chunks.h5
tests/data/hdf5_ref/v4_extensible_array_spillover.h5
tests/data/hdf5_ref/v4_btree2_chunks.h5
tests/data/hdf5_ref/v4_btree2_internal_chunks.h5
tests/data/hdf5_ref/nbit_filter_i32.h5
tests/data/hdf5_ref/scaleoffset_filter_i32.h5
```

`scripts/tracehash-compare.sh /tmp/rust.tsv /tmp/c.tsv` currently matches:

```text
tracehash: traces match for 15261 left rows and 15261 right rows
```

The current tracehash corpus has no count differences, missing inputs, or
matched-output divergences across the instrumented object-header, message
decode, chunk lookup, fractal-heap, and filter probes.

## Next Command When C Trace Exists

```sh
scripts/tracehash-compare.sh /tmp/rust.tsv /tmp/c.tsv
```
