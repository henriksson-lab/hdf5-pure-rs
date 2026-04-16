# Tracehash Divergence Report Status

Date: 2026-04-15

## Local Runs

- Rust corpus runner: `scripts/tracehash-rust-corpus.sh`
- Rust trace output: `/tmp/rust.tsv`
- Rust trace rows observed locally: 22839
- C corpus runner: `scripts/tracehash-c-corpus.sh`
- C trace output: `/tmp/c.tsv`
- C trace rows observed locally: 22839
- Latest revalidation: after adding global-heap dereference and
  variable-length string read tracehash coverage,
  `scripts/tracehash-compare.sh /tmp/rust.tsv /tmp/c.tsv` matches 22839 Rust
  rows to 22839 C rows.
- Focused VDS selection revalidation:
  `scripts/tracehash-compare.sh /tmp/rust-vds-selection.tsv /tmp/c-vds-selection.tsv`
  matches 6 Rust rows to 6 C rows for `hdf5.selection.deserialize`.
- Focused external-link revalidation:
  `scripts/tracehash-compare.sh /tmp/rust-ext-only.tsv /tmp/c-ext-only.tsv`
  matches 1 Rust row to 1 C row for `hdf5.external_link.resolve`.
- Focused same-file VDS source revalidation:
  `scripts/tracehash-compare.sh /tmp/rust-vds-source-only.tsv /tmp/c-vds-source-only.tsv`
  matches 1 Rust row to 1 C row for `hdf5.vds.source.resolve`.

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
- Add datatype property tracehash coverage for version, class bits, declared
  size, and raw class-specific property bytes.
- Add dataspace extent tracehash coverage for version, rank, flags, extent
  class, dimensions, and maximum dimensions.
- Add serialized selection tracehash coverage for VDS selection buffers.
- Add fill-value tracehash coverage for message version, allocation time,
  write time, defined state, and raw fill bytes.
- Add v1 B-tree chunk lookup tracehash coverage keyed by chunk index address
  and scaled chunk coordinates.
- Add v2 B-tree chunk record decode tracehash coverage keyed by raw chunk
  record bytes, with decoded address, byte length, filter mask, and scaled
  coordinates.
- Add v2 B-tree chunk internal-node child traversal coverage for decoded child
  pointer address, child record count, and cumulative descendant record count.
- Add fractal-heap managed-object tracehash coverage for direct/indirect block
  resolution, selected block address/size, object offset/length, and filtered
  block state.
- Add fractal-heap huge-object and tiny-object tracehash probes. A generated
  dense-attribute huge-object fixture was spot-checked by filtering traces to
  `hdf5.fractal_heap.huge_object`, matching 20 Rust rows to 20 C rows.
- Add global-heap dereference and variable-length string read probes, and add
  `tests/data/strings.h5` to the default Rust/C tracehash corpus. The corpus
  walker now exercises variable-length string datasets and attributes without
  using public API metadata reopens.
- Add focused external-link target and same-file VDS source mapping probes.
  These remain outside the default corpus to avoid unrelated full-link/VDS
  traversal differences, but filtered trace comparisons match the intended
  probe rows.
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
tests/data/strings.h5
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
tracehash: traces match for 22839 left rows and 22839 right rows
```

Focused VDS selection rows currently match:

```text
tracehash: traces match for 6 left rows and 6 right rows
```

Focused external-link target rows currently match:

```text
tracehash: traces match for 1 left rows and 1 right rows
```

Focused same-file VDS source rows currently match:

```text
tracehash: traces match for 1 left rows and 1 right rows
```

The current tracehash corpus has no count differences, missing inputs, or
matched-output divergences across the instrumented object-header, message
decode, chunk lookup, fractal-heap, global-heap, vlen-read, and filter probes;
focused external-link and VDS source-resolution probes also match.

## Next Command When C Trace Exists

```sh
scripts/tracehash-compare.sh /tmp/rust.tsv /tmp/c.tsv
```
