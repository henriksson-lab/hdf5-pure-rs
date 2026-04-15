# Unsupported HDF5 Features

This crate intentionally implements a subset of HDF5. The features below are
recognized as outside the supported surface until dedicated implementations and
compatibility tests are added.

## Dataset Storage

- Virtual dataset reads support serialized all-selection and regular hyperslab
  mappings, including relative external source files. Point selections,
  irregular hyperslabs, printf-gap expansion, and full VDS access-property behavior remain
  unsupported.
- v4 chunk indexes return `Unsupported` except for single-chunk datasets,
  unfiltered implicit chunk indexes, fixed-array chunk indexes, extensible-array
  chunk indexes including data/super-block spillover, and v2-B-tree chunk
  indexes.
- Filtered v4 implicit chunk indexes remain explicitly unsupported; HDF5 does
  not normally select implicit chunk indexes for filtered datasets.
- `MutableFile::write_chunk` appends and replaces full chunks in v1 chunk
  B-trees, including append-only root-leaf rebuilds for full leaves. It can
  replace existing chunks in v4 fixed-array indexes. Growing v4 fixed arrays
  and updating extensible-array or v2-B-tree chunk indexes remain unsupported.

## Filters

- NBit decodes datatype-aware HDF5 set-local filter parameters.
- ScaleOffset decodes datatype-aware integer and decimal floating-point
  set-local filter parameters.
- SZip returns `Unsupported` permanently unless a pure-Rust implementation is
  added later.
- Unknown filters return `Unsupported`.

## Compound Members

Compound field metadata preserves each embedded member datatype, including byte
order and nested datatype information. Direct `read_field::<T>()` supports
fixed-size fields where the requested Rust type has the same byte size.
`read_field_raw()` returns raw per-record member bytes for non-primitive
members so callers can handle nested, array, variable-length, or reference
payloads explicitly.

Compound field byte-swapping is applied for big-endian primitive numeric
members. Recursive value conversion is available for nested compound, array,
variable-length, and reference members through `read_field_values()`.

## Fractal Heaps

- Directly addressed huge fractal heap objects and v2-B-tree tracked huge
  objects are readable, including filtered records when the heap filter pipeline
  is supported. Unsupported record variants fail explicitly.
- Filtered direct managed fractal heap blocks are readable with supported
  filter pipelines. Unsupported filtered variants fail explicitly.

## Tracehash

Rust-side probes exist for decode checkpoints, chunk-index lookup, filter
application, and fractal-heap reads. The vendored HDF5 tree has matching probe
calls plus a small public-API corpus driver. The current local comparison has
Rust emitting `/tmp/rust.tsv` and patched HDF5 C emitting `/tmp/c.tsv`; the
current report shows an exact 15261-row match with no count differences, missing
inputs, or matched-output divergences. See
`analysis/tracehash_divergence_report.md`.
