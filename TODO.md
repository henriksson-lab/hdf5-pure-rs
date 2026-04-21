# TODO: hdf5-pure-rust

## Current status: 326 tests, 0 failures, 0 warnings.

## Audit Backlog (generated 2026-04-15)

### P0: Parser Robustness And No-Panic Guarantees
- [x] Reject datatype messages with truncated fixed-size properties: fixed-point, floating-point, and bitfield classes must have their required class property bytes.
- [x] Make compound datatype field parsing return a structured error instead of silently returning partial fields through `Option`.
- [x] Make enum datatype member parsing return a structured error instead of silently returning partial names or values through `Option`.
- [x] Make array datatype parsing reject truncated dimension tables and missing base datatypes with an error path that callers can surface.
- [x] Make variable-length datatype parsing distinguish HDF5 vlen string metadata from vlen sequence base datatype metadata without permissive fallback ambiguity.
- [x] Add malformed datatype regression vectors for truncated compound names, truncated v1/v2 compound dimension blocks, truncated member offsets, and truncated nested member datatypes.
- [x] Add malformed datatype regression vectors for truncated enum base datatype, truncated enum names, and truncated enum value payloads.
- [x] Add malformed datatype regression vectors for array datatypes with too many dimensions, overflowing dimension byte counts, and missing base datatypes.
- [x] Audit every object-header message decoder for unchecked indexing, partial zero defaults, and permissive trailing truncation.
- [x] Reject object-header continuation messages whose target range overflows file size or overlaps invalid metadata regions.
- [x] Reject malformed shared-message payloads explicitly instead of treating them as unknown or empty messages.
- [x] Reject invalid message version/class combinations with `InvalidFormat` rather than later high-level failures.
- [x] Add a focused fuzz-style test that truncates every byte prefix of representative datatype, dataspace, data layout, link, attribute, fill value, and filter pipeline messages.
- [x] Add a focused corrupt-file test that opens a synthetic file with every object-header message truncated at each byte boundary and asserts no panic.
- [x] Replace remaining production `unwrap()`/`expect()` reachable from file input with checked error propagation.
- [x] Audit arithmetic on dimensions, chunk counts, element sizes, and file offsets for overflow before allocation or seek.
- [x] Add allocation caps or overflow checks for declared rank, number of members, number of filters, number of chunks, heap object sizes, and VDS mapping counts.

### P0: HDF5-C Faithfulness And Divergence Tracking
- [x] Run the local tracehash Rust corpus and patched HDF5 C corpus after the new TODO backlog is introduced, then commit the current divergence report.
- [x] Add tracehash coverage for datatype property parsing details, not just high-level datatype class decode.
- [x] Add tracehash coverage for dataspace extent decode, selection decode, and serialized VDS selection decode.
- [x] Add tracehash coverage for fill-value message version/allocation/write-time transitions.
- [x] Add tracehash coverage for B-tree v1 chunk lookup decisions, including chunk coordinate key comparison.
- [x] Add tracehash coverage for v2 B-tree internal-node traversal and record decode decisions.
- [x] Add tracehash coverage for fixed-array and extensible-array chunk index address resolution.
- [x] Add tracehash coverage for fractal heap direct block, indirect block, managed object, huge object, and filtered object reads.
- [x] Add tracehash coverage for global heap object dereference and variable-length string reads.
- [x] Add tracehash coverage for external link resolution and same-file VDS source resolution.
- [x] Record known intentional divergences from libhdf5 in `analysis/unsupported_features.md` with test names for each.
- [x] Add a small script that fails CI when Rust-vs-C tracehash output diverges for the supported corpus.
- [x] Add a script target that regenerates all local HDF5 fixture files and emits the HDF5 C library version used.
- [x] Pin the HDF5 source commit used for fixture generation in one machine-readable file, not just prose.

### P1: Dataset Read Semantics
- [x] Implement point-selection virtual dataset mappings or explicitly reject them at decode time with a regression test.
- [x] Implement irregular hyperslab virtual dataset mappings or explicitly reject them at decode time with a regression test.
- [x] Implement VDS access-property behavior for missing source files, prefix substitution, and view policy, or document each unsupported mode with tests.
- [x] Extend VDS reads to non-`i32` primitive datatypes with conversion parity tests.
- [x] Add VDS tests for scalar dataspace mappings, null dataspace mappings, and zero-sized mappings.
- [x] Add VDS tests where source and destination rank differ but the mapping is valid under HDF5 rules.
- [x] Add VDS tests for overlapping mappings and verify libhdf5-compatible precedence.
- [x] Add chunked partial-read tests that combine hyperslab selections with missing chunks and fill values.
- [x] Add chunked partial-read tests for filtered chunks where the filter mask skips one middle filter in a multi-filter pipeline.
- [x] Add Fletcher32 verification failure tests for corrupted filtered and unfiltered chunks.
- [x] Add big-endian filtered chunk read coverage for NBit and ScaleOffset.
- [x] Add compact dataset read tests for zero-sized dataspaces and scalar compound payloads.
- [x] Add contiguous dataset read tests for external storage files or explicitly mark external raw data storage unsupported.
- [x] Add tests for reading datasets whose declared storage address is the undefined HDF5 address.
- [x] Add tests for datasets with allocation-time-late storage and fill-time-never semantics.

### P1: Chunk Index Coverage
- [x] Add v1 B-tree chunk lookup tests for multidimensional chunk coordinates beyond 2D.
- [x] Add v1 B-tree tests for sparse chunks with non-monotonic insertion order.
- [x] Add v1 B-tree tests for large chunk offsets that require full address-size handling.
- [x] Add v2 B-tree chunk index tests with multiple internal levels, not just one internal root path.
- [x] Add v2 B-tree tests for filtered chunks with nonzero per-record filter masks.
- [x] Add fixed-array tests for all page initialization states, including absent pages and fill-value fallback.
- [x] Add extensible-array tests for secondary block addressing across index-block, data-block, and super-block transitions.
- [x] Add implicit chunk index tests for multidimensional datasets and partial edge chunks.
- [x] Reject filtered implicit chunk indexes with a targeted fixture and error assertion.
- [x] Add chunk index checksum corruption tests for fixed array, extensible array, and v2 B-tree metadata.
- [x] Audit chunk coordinate linearization against libhdf5 for rank, chunk-dim, and unlimited-dimension edge cases.

### P1: Filters
- [x] Add exact libhdf5 parity vectors for NBit signed integers, unsigned integers, floating point, and compound members.
- [x] Add exact libhdf5 parity vectors for ScaleOffset integer minbits, signed values, zero minbits, and floating point scale types.
- [x] Add malformed NBit parameter tests for impossible precision/offset combinations.
- [x] Add malformed ScaleOffset parameter tests for invalid scale type, missing client data, and output-size mismatch.
- [x] Make all filter decoders verify that output length exactly matches the expected logical chunk size unless the HDF5 filter semantics allow otherwise.
- [x] Add multi-filter pipeline tests for every supported filter order that libhdf5 can emit.
- [x] Add Blosc feature tests to CI or mark the feature as manually verified only.
- [x] Add tests that unknown optional filters are skipped only when HDF5 semantics allow skipping, and required unknown filters fail.
- [x] Keep SZip unsupported unless a pure-Rust decoder is added, but add a fixture asserting the exact error surface.

### P1: Datatype And Conversion Semantics
- [x] Replace ad hoc high-level datatype conversion with a central conversion table modeled after libhdf5 conversion classes.
- [x] Add integer conversion tests for signed/unsigned widening, narrowing, and overflow behavior.
- [x] Add float conversion tests for f32/f64, integer-to-float, float-to-integer, NaN, infinity, and endian-swapped payloads.
- [x] Add fixed-length string tests for null padding, space padding, null termination, and UTF-8 character set flags.
- [x] Add variable-length string tests for empty strings, null strings, UTF-8 strings, and global heap edge cases.
- [x] Add opaque datatype tests including tag decode and raw payload reads.
- [x] Add reference datatype tests for object references and dataset region references.
- [x] Add time datatype tests or explicitly reject HDF5 time datatype reads.
- [x] Add enum conversion tests where the enum base type is wider than one byte and big-endian.
- [x] Add compound tests for member padding, overlapping members, reordered members, and nested variable-length members.
- [x] Add array datatype tests for v1/v2/v3/v4 encodings and multidimensional array fields.

### P1: Groups, Links, Attributes, And Heaps
- [x] Add dense group tests with multiple v2 B-tree levels for link-name indexing.
- [x] Add dense group tests with creation-order indexing enabled and disabled.
- [x] Add dense attribute tests with creation-order indexing enabled and disabled.
- [x] Add attribute read tests for large compact attributes, dense attributes, and variable-length attribute payloads.
- [x] Add link tests for UTF-8 names, non-ASCII external filenames, and invalid character-set flags.
- [x] Add soft-link cycle detection tests with a bounded traversal limit.
- [x] Add external-link tests for missing files, relative paths, absolute paths, and same-directory resolution.
- [x] Add global heap tests with deleted objects, duplicate object IDs, and collection padding.
- [x] Add fractal heap tests for indirect block growth beyond one level.
- [x] Add fractal heap checksum corruption tests for direct and indirect blocks.
- [x] Add filtered fractal heap tests using non-deflate filters where libhdf5 can generate them.

### P1: Writer And Mutable File Semantics
- [x] Update README write-support claims to mention current `MutableFile` v4 fixed-array replacement support and remaining writer gaps precisely.
- [x] Add writer support for creating datasets with fill-value messages and allocation-time/fill-time properties.
- [x] Add writer support for compact datasets beyond primitive numeric payloads, including fixed strings and compound values.
- [x] Add writer support for creating dense groups after compact link thresholds are exceeded.
- [x] Add writer support for dense attributes after compact attribute thresholds are exceeded.
- [x] Add writer support for variable-length strings via global heap allocation.
- [x] Add writer support for enum, opaque, array, and nested compound datatype messages.
- [x] Add writer support for chunked datasets with fixed-array or extensible-array indexes, or explicitly keep v1-only writer indexes documented.
- [x] Add mutable append/replacement support for simple depth-0 v4 v2-B-tree chunk indexes.
- [x] Add mutable append support for v4 extensible-array chunk indexes through the first data-block allocation.
- [x] Add mutable append support for multi-level v2-B-tree rebuilds after chunk-index appends.
- [x] Add mutable append support for v4 extensible-array super-block/page growth.
- [x] Add mutable replacement support for filtered chunks while preserving or recomputing per-chunk filter masks.
- [x] Add shrink tests that ensure removed chunks are no longer returned after `MutableFile::resize_dataset`.
- [x] Add grow tests that ensure newly exposed regions use the correct fill value for contiguous and chunked datasets.
- [x] Add writer round-trip tests validated by both h5dump and h5py for each newly supported writer feature.
- [x] Decide whether free-space managers are intentionally unsupported for writes and document the resulting file-growth behavior.

### P1: Public API And Error Surface
- [x] Replace parser helper APIs that return `Option` for malformed file data with `Result` carrying `InvalidFormat` or `Unsupported`.
- [x] Ensure all public read APIs distinguish unsupported HDF5 features from corrupt file data.
- [x] Add stable error-message tests only where callers are likely to branch on the error class.
- [x] Add `Dataset::read_dyn` or equivalent for N-dimensional ndarray reads beyond 1D/2D.
- [x] Add safe APIs for inspecting raw datatype messages, raw dataspace messages, and creation properties for unsupported features.
- [x] Add API docs warning that `read::<T>()` is not full libhdf5 conversion parity yet.
- [x] Add API docs for `read_field_raw()` and recursive compound `read_field_values()` limitations.
- [x] Audit exported types for accidental exposure of internal layout structs that may need semver stability.
- [x] Add examples for read-only use, write use, VDS use, and tracehash divergence checks.

### P2: Test Corpus, CI, And Documentation
- [x] Update README test count from 294 to the current count, or generate the count dynamically in release notes instead of hard-coding it.
- [x] Move generated or vendored HDF5 reference artifacts behind documented regeneration scripts and avoid committing local-only scratch output.
- [x] Decide whether the vendored `hdf5/` source tree belongs in the repository; document if it is required for tracehash and fixture generation.
- [x] Add CI jobs for default features, `--no-default-features`, `--features derive`, `--features blosc`, and `--features tracehash` where practical.
- [x] Add a CI job that runs tests under Miri or sanitizers for parser-only units where dependencies permit it.
- [x] Add cargo-deny or equivalent dependency/license checks.
- [x] Add cargo-semver-checks before releases.
- [x] Add criterion benchmarks that use generated fixture files and do not write fixed `/tmp` paths.
- [x] Replace benchmark hard-coded `/tmp/bench_rust.h5` with a unique temporary path.
- [x] Add performance baselines for chunked reads, filtered reads, dense group traversal, and VDS reads.
- [x] Add a compatibility matrix that maps each supported feature to at least one fixture and one test.
- [x] Add release checklist documenting fixture regeneration, tracehash comparison, test matrix, README count update, and crates.io packaging.
- [x] Audit untracked local files before any commit and classify each as source, generated fixture, vendored dependency, scratch output, or ignore rule.
- [x] Add `.gitignore` entries for local tracehash outputs, temporary HDF5 files, and generated reports that should never be committed.

## Next Fixes
- [x] Implement readable extensible-array chunk index data/super-block spillover.
- [x] Implement virtual dataset reads from parsed v3/v4 virtual layout mappings for regular hyperslab selections.
- [x] Extend `MutableFile::write_chunk` to replace existing chunks in leaf v1 B-tree indexes.
- [x] Extend `MutableFile::write_chunk` to handle full v1 B-tree leaf rebalancing.
- [x] Add `MutableFile` support for updating v4 fixed-array chunk indexes when replacing existing chunks.
- [x] Implement paged fixed-array chunk index data blocks.
- [x] Keep filtered v4 implicit chunk indexes explicitly unsupported; HDF5 does not normally choose implicit indexes for filtered datasets.
- [x] Implement filtered directly addressed huge fractal-heap object reads.
- [x] Keep documenting SZip as permanently unsupported unless a pure-Rust decoder is added later.
- [x] Broaden the tracehash corpus with fixtures for extensible-array and paged fixed-array spillover.
- [x] Triage tracehash corpus expansion for virtual datasets and writer-side chunk-index updates; VDS and writer-side behavior now have regression tests, while default tracehash parity intentionally waits for dedicated patched-C VDS/writer probes.
- [x] Resolve supported virtual dataset shapes from regular hyperslab mappings instead of returning stored placeholder extents.
- [x] Implement virtual dataset reads for serialized `H5S_SEL_ALL` source and destination mappings.
- [x] Add virtual dataset same-file source (`"."`) coverage.
- [x] Add virtual dataset mixed `H5S_SEL_ALL` and regular hyperslab selection coverage.
- [x] Honor defined virtual dataset fill values for unmapped regions.
- [x] Honor defined chunked dataset fill values for missing/unallocated chunks.
- [x] Honor per-chunk filter masks when reversing filtered chunk pipelines.
- [x] Reject per-chunk filter masks that reference filters outside the pipeline.
- [x] Reject filter pipelines longer than the 32-bit per-chunk filter mask can represent.
- [x] Add v4 filtered single-chunk filter-mask coverage.
- [x] Add dataset creation property coverage for parsed fill values.
- [x] Add old-format fill-value read and property-list coverage.
- [x] Add reference `fill18.h5` chunked fill-value read coverage.
- [x] Reject truncated defined fill-value message payloads.
- [x] Reject truncated data layout message payloads without panics or silent partial reads.
- [x] Reject truncated dense link/attribute info message addresses instead of decoding partial zeros.
- [x] Reject truncated filter pipeline message payloads instead of returning partial decoded pipelines.
- [x] Reject dataspace messages whose declared rank dimensions or max dimensions are truncated.
- [x] Reject truncated attribute message name/datatype/dataspace metadata sections before slicing.
- [x] Reject truncated link message optional fields and variable-width lengths before reading.
- [x] Make write tests hermetic by replacing fixed `tests/data/*.h5` output paths with `tempfile::tempdir()` or unique per-test paths.
- [x] Remove or isolate the absolute local `tracehash` path dependency so normal Cargo metadata is portable.
- [x] Add a concise README supported/unsupported feature table to prevent users from inferring full HDF5 compatibility.
- [x] Ensure generated test artifacts never persist in `tests/data`, including after failed tests.
- [x] Extend compound datatype support beyond primitive `read_field::<T>()` with `read_field_raw()` for nested compound, array, variable-length, and reference member payloads; recursive typed conversion remains explicitly unsupported.
- [x] Implement readable v4 implicit chunk indexes for unfiltered contiguous chunk storage; fixed array, extensible array, v2 B-tree, and filtered implicit indexes remain explicitly unsupported.
- [x] Vendor tracehash locally and add a patched-C runner that emits `/tmp/c.tsv`; matching HDF5 C-side probe targets remain documented for patched C builds.

## Next Improvements
- [x] Implement readable v4 fixed-array chunk indexes.
- [x] Implement readable v4 extensible-array chunk indexes for direct index-block entries and data/super-block spillover.
- [x] Implement readable v4 v2-B-tree chunk indexes for leaf-root chunk trees.
- [x] Implement filtered v4 fixed-array chunk indexes; HDF5 does not select implicit chunk indexes for filtered datasets.
- [x] Implement recursive typed compound conversion for nested compound members.
- [x] Implement recursive typed compound conversion for array members.
- [x] Implement recursive typed compound conversion for variable-length members.
- [x] Implement recursive typed compound conversion for reference members.
- [x] Add concrete HDF5 C-side tracehash probes in datatype decode paths (`H5T*`).
- [x] Add concrete HDF5 C-side tracehash probes in object-header decode paths (`H5O*`).
- [x] Add concrete HDF5 C-side tracehash probes in filter pipeline decode/application paths (`H5Z*`).
- [x] Add concrete HDF5 C-side tracehash probes for chunk index resolution.
- [x] Add concrete HDF5 C-side tracehash probes for fractal heap lookup and dense link/attribute traversal.
- [x] Generate Rust-vs-C tracehash divergence report: Rust emits `/tmp/rust.tsv`, patched HDF5 C emits `/tmp/c.tsv`, and the current instrumented corpus matches row-for-row with no output mismatches.
- [x] Implement v2 B-tree internal-node traversal.
- [x] Implement filtered fractal heap object lookup.
- [x] Implement huge fractal heap object lookup.
- [x] Implement datatype-aware NBit filter decoding.
- [x] Implement datatype-aware ScaleOffset filter decoding.
- [x] Extend `MutableFile` resize support to update chunk indexes when new chunks are appended.
- [x] Add C-generated fixture files for v4 fixed-array chunk indexes.
- [x] Add C-generated fixture files for paged v4 fixed-array chunk indexes.
- [x] Add C-generated fixture files for v4 extensible-array chunk indexes.
- [x] Add C-generated fixture files for v4 v2-B-tree chunk indexes.
- [x] Add C-generated fixture files for filtered chunk indexes.
- [x] Add C-generated fixture files for modern dense fractal heap coverage.
- [x] Audit untracked repo artifacts before commit, especially `.codex`, vendored `hdf5/`, `analysis/`, `scripts/`, and `tools/`.
- [x] Refresh tracehash documentation after the Rust-vs-C corpus reaches an exact row-for-row match.

### Core Format Engine (Phases 1-8)
- [x] Binary I/O primitives, superblock v0-v3, Jenkins lookup3 checksum
- [x] Object header v1/v2 parsing, v1 B-tree, local heap, symbol table
- [x] Dataset reading (contiguous/compact/chunked/compressed)
- [x] Attribute reading (v1/v2/v3)
- [x] File writing (superblock, groups, contiguous datasets, C library verified)
- [x] Chunked writing with deflate/shuffle compression
- [x] Compatibility fixes for supported write paths (float datatype encoding, B-tree padding)
- [x] Fractal heap + leaf-root v2 B-tree support for dense link/attr storage, global heap

### High-Level API (Phases A-E)
- [x] `H5Type` trait + generic reads: `ds.read::<f64>()`, `read_scalar`, `read_1d`, `read_2d`
- [x] Datatype/Dataspace public API: `ds.dtype()`, `ds.space()`, `ds.is_chunked()`, etc.
- [x] Write-through-API: `WritableFile::create()`, `DatasetBuilder`, `WritableGroup`
- [x] Selection/Hyperslab: `ds.read_slice::<f64>(10..20)`, 1D/2D/chunked
- [x] Property lists: `DatasetCreate`, `FileCreate`, `ds.create_plist()`

### Extended Features (Phases F-H + extras)
- [x] `Location` trait on File/Group/Dataset
- [x] Soft/external link read/write
- [x] String reading (fixed-length + variable-length via global heap)
- [x] Big-endian type conversion
- [x] Compound/enum datatype reading
- [x] LZF, NBit, and ScaleOffset filters; SZip now fails explicitly as unsupported on reads
- [x] Blosc filter (feature-gated `blosc2-rs`)
- [x] H5Type derive macro (separate proc-macro crate)
- [x] Limited in-place dataset resizing via `MutableFile`
- [x] Virtual dataset layout parsing (v3/v4); virtual dataset reads fail explicitly as unsupported

### HDF5 C Test Suite Ported (Phases T1-T11)
- [x] T1: Reference file smoke tests for checked-in corpus, 32 tests
- [x] T2: Corrupt file handling -- no panics on checked-in corpus + CVE regressions, 9 tests
- [x] T3: All datatypes -- i8-i64, u8-u64, f32, f64, BE, compound, enum, strings, N-D, 22 tests
- [x] T4: Dataset layouts -- compact/contiguous/chunked, deflate/shuffle/fletcher32, selections, 16 tests
- [x] T5: Attributes -- scalar, array, string, group/dataset attrs, dense storage
- [x] T6: Groups & links -- nested, hard/soft/external, dense, link_exists, 8 tests
- [x] T7: Dataspace & selections -- scalar/simple/null, maxdims, 1D/2D slices, chunked slices, 16 tests
- [x] T8: Object headers -- v1/v2, timestamps, continuation chunks, all message types, 7 tests
- [x] T9: Heaps & indices -- global heap, local heap, fractal heap, v2 B-tree, chunk indices, 10 tests
- [x] T10: Write round-trips -- h5dump, h5py, all types/layouts/filters, resize, 9 tests
- [x] T11: Cross-platform -- big-endian, old formats, file space strategies, charsets, 12 tests

### Faithfulness Audit vs HDF5 C Library
- [x] Replace broad "bitwise compatible" wording with a precise supported-feature compatibility statement.
- [x] Reconcile license metadata: Cargo.toml now uses BSD-3-Clause to match README/LICENSE.
- [x] Update README reference-file claims so checked-in corpus changes do not imply broad compatibility.
- [x] Add negative tests for explicitly unsupported paths: unsupported filters, virtual layout metadata-only parsing, huge fractal-heap objects, and filtered fractal heaps.
- [x] Clearly document NBit and ScaleOffset status; generic filter pipeline now decodes datatype-aware NBit/ScaleOffset parameters instead of pass-through.
- [x] Clearly document unsupported chunk index types; reads now return `Unsupported` instead of falling back to v1 B-tree for implicit, fixed array, extensible array, and v2 B-tree chunk indexes.
- [x] Remove speculative v2 B-tree internal-node parsing; internal v2 B-trees now return `Unsupported`.
- [x] Implement direct filtered managed fractal heap object lookup and direct/indirect huge object lookup.
- [x] Preserve embedded compound member datatypes and implement big-endian primitive member byte swapping; recursive high-level conversion for nested/array/vlen/reference fields is documented as unsupported.
- [x] Extend `MutableFile` from resize-only metadata updates to append chunks into leaf v1 chunk indexes written by this crate.
- [x] Keep v4 chunk indexes explicitly unsupported except for single-chunk datasets; no fallback to incorrect readers.
- [x] Keep virtual dataset reads explicitly unsupported while preserving layout metadata parsing.
- [x] Keep datatype-aware NBit and ScaleOffset decoding in the generic filter pipeline.
- [x] Keep SZip permanently unsupported unless a pure-Rust decoder is added later.

### Tracehash Divergence Tracking
- [x] Document the vendored tracehash path: `tools/tracehash`.
- [x] Add an optional `tracehash` feature for Rust-side probes without enabling it in normal builds.
- [x] Switch the Rust-side `tracehash` dependency to the published
  [`tracehash-rs`](https://crates.io/crates/tracehash-rs) 0.1 crate
  (`package = "tracehash-rs"` rename so call sites keep importing
  `tracehash::...`). Updated `output_bool`/`output_bytes` call sites to
  the new `output_value(&T)` API since the published crate dropped the
  explicit helpers in favor of the generic `TraceHash`-based path.
  `scripts/tracehash-compare.sh` now prefers a `tracehash-compare`
  binary on `$PATH` (installed via `cargo install tracehash-rs`) and
  falls back to `cargo run --package tracehash-rs`.

### Format-layer 1:1 mapping refactor (2026-04-18)

- [x] Split fused decode+traverse functions across `src/format/` so each
  half maps cleanly to its libhdf5 counterpart. Pure prefix-deserialize
  helpers now live as `decode_*` (mirroring `H5*_cache_*_deserialize`),
  with the existing `read_*` entry points either retained as thin
  compose wrappers (where backward compatibility matters) or replaced
  by separate traversal halves. Files touched:
  - `format/local_heap.rs`: `decode_prefix` + `load_data_segment`.
  - `format/global_heap.rs`: `decode_header` + `walk_objects`.
  - `format/btree_v2.rs`: `decode_internal_node` (was inlined into
    `read_internal_records`).
  - `format/fixed_array.rs`: `decode_data_block_prefix` +
    `collect_data_block_elements`.
  - `format/extensible_array.rs`: `decode_data_block_prefix`,
    `decode_super_block`, `decode_index_block`.
  - `format/fractal_heap.rs`: `decode_indirect_block` +
    `lookup_in_indirect_block`, `decode_filtered_indirect_block` +
    `lookup_in_filtered_indirect_block` (the originating case from the
    `read_from_indirect_block_rows` analysis).
  No public API change; tests stay green at 468. `ccc_mapping.toml`
  refreshed to point the canonical `H5*_cache_*_deserialize` targets at
  the new `decode_*` halves.
- [x] Mirror libhdf5's file/module split for the `format/` tree
  (2026-04-18). Four files moved into directories whose layout mirrors
  the matching `H5*.c` files in libhdf5:
  - `format/fixed_array/{mod,hdr,dblock}.rs` ← `H5FA{,hdr,dblock,dblkpage}.c`
    (dblkpage folded into dblock — Rust port has no separate page-cache).
  - `format/extensible_array/{mod,hdr,iblock,sblock,dblock}.rs` ←
    `H5EA{,hdr,iblock,sblock,dblock,dblkpage}.c`.
  - `format/fractal_heap/{mod,hdr,iblock,dblock,man,huge,tiny,dtable}.rs` ←
    `H5HF{,hdr,iblock,dblock,man,huge,tiny,dtable}.c`.
  - `format/object_header/{mod,cache,chunk,msg}.rs` ← `H5O{cache,chunk,
    message,pkg}.c`.
  Tests stayed at 471 throughout. The smaller files
  (`btree_v1.rs`, `btree_v2.rs`, `checksum.rs`, `global_heap.rs`,
  `local_heap.rs`, `superblock.rs`, `symbol_table.rs`) didn't need
  splitting — each fits on a single screen and already maps cleanly to
  one C file.

- [ ] Mirror libhdf5's file/module split for the `hl/` tree. Two
  oversized files remain:
  - `src/hl/mutable_file.rs` → split by subsystem (chunk-btree update,
    extensible-array update, object-header rewrite, dense-storage
    update, allocator/io).
  - `src/hl/dataset.rs` (~3300 LOC) → split read paths (chunked /
    contiguous / virtual) into siblings under `src/hl/dataset/`.
  Bigger refactor than the `format/` tree because libhdf5 doesn't have
  one-to-one analogs; we'd be partitioning by Rust-internal subsystem
  rather than mirroring C exactly.

- [x] Closer-to-1:1 audit (no-fusion + naming-drift, 2026-04-18). Two
  more fusion candidates found and split:
  - `hl/conversion.rs::for_dataset` extracted into per-source-class
    helpers (`kind_for_integer_source`, `kind_for_float_source`,
    `kind_for_passthrough`) mirroring libhdf5's `H5T__conv_*` family.
    The dispatcher itself now matches on `DatatypeClass` and delegates,
    instead of nesting 30+ lines of per-class branching.
  - `hl/dataset.rs::collect_btree_v1_chunks` extracted a pure
    `decode_chunk_btree_node` (returns `ChunkBTreeNode::Leaf|Internal`)
    that mirrors `H5B__cache_deserialize` for the chunk-index node
    type. The recursive driver becomes a thin `match` over the
    decoded node.
  Naming-drift audit: 451 mapped pairs have name divergence from the C
  side, but inspection shows almost all are deliberate — Rust uses
  descriptive names where C uses abbreviations (`compound_fields` vs
  `H5O__dtype_decode_helper`, `read_at` vs `H5*_protect`,
  `datatype_encoded_len` vs `H5O__dtype_size`). Renaming Rust to match
  C's abbreviations would degrade readability for marginal TUI
  benefit; the mapping file already bridges the two.

- [x] Translation-gap audit driven by the now-comprehensive
  `ccc_mapping.toml` (2026-04-18). Three concrete C-side validation
  checks were missing on the Rust side and have been added:
  - `format/fractal_heap.rs::read_managed_object` now validates the
    heap-ID version bits (top 2 bits of byte 0). Mirrors libhdf5's
    `H5HF_get_obj_len` "incorrect heap ID version" check.
  - `format/fractal_heap.rs::read_managed` now bounds the decoded
    object offset against `2^max_heap_size` and the object length
    against `max_managed_obj_size`. Mirrors `H5HF__man_op_real`.
  - `format/messages/link_info.rs::decode` now rejects
    `max_creation_index > u32::MAX`, matching upstream
    `H5O__linfo_decode`'s `H5L_MAX_CRT_IDX_VAL` bound. Covered by two
    tests in `tests/robustness_test.rs`.
  Other C-only error strings were investigated and cleared as
  non-actionable: most are runtime identifier checks that don't apply
  to a typed Rust API, validation that already lives in a different
  function, or cross-checks against state we don't have at decode time
  (datatype-vs-fill-value-size).

- [ ] Bucket CCC "unsupported subsystem" misses into explicit roadmap
  items instead of leaving them as raw compare noise. Current
  `ccc-rs missing` output is still dominated by large libhdf5 surfaces
  that are intentionally out of scope or not yet translated, especially:
  - VOL / async / plugin / connector infrastructure (`H5VL*`,
    `H5ES*`, `H5PL*`)
  - MPI / parallel I/O / distributed datatype-selection paths
    (`H5_mpi*`, `H5S__mpio*`, parallel `H5D*`) — out of scope; we will
    not use MPI/parallel-HDF5. If CPU parallelism is added later, use
    Rayon and keep it Rust-side rather than chasing libhdf5's MPI stack.
  - Alternative VFDs and cloud/network drivers (`H5FD__hdfs*`,
    `H5FD__ros3*`, direct/core/stdio driver parity)
  - Large write-side object-header / message-management families
    (`H5O_msg_*`, shared-message machinery, free-space managers)
  - Remaining unported dataspace selector families (`all`/`none`
    iterators, projection helpers, full selection iterator parity)
  Action for a later pass:
  - classify each family as `won't implement`, `reader-only not needed`,
    or `planned parity work`
  - keep the translation rule explicit: when a function is brought over,
    translate it completely on the first pass where feasible, instead of
    landing partial/stubbed behavior and planning to fill semantics in
    later. Use follow-up passes for auditability/refactoring, not for
    basic missing branches.
  - record global policy: do parallelization last. Finish
    single-threaded faithful translation/audit first, then consider
    Rayon-based acceleration only after behavior is pinned.
  - mirror that classification in `analysis/unsupported_features.md`
  - trim obvious false-positive mappings like parser artifacts (`if`,
    `while`, `FAIL`, `NULL`) from the CCC follow-up workflow so the
    missing report is decision-relevant.

- [x] `ccc_mapping.toml` extended to **100% coverage** (732/732 Rust
  functions, 691 entries — was 19% / 140 entries before this work).
  Every Rust function in `src/` has a mapping to its closest libhdf5
  counterpart. Categories covered, with the C target each maps to:
  - High-level public API (`hl/file.rs`, `hl/group.rs`, `hl/dataset.rs`,
    `hl/attribute.rs`, `hl/datatype.rs`, `hl/dataspace.rs`,
    `hl/writable_file.rs`, `hl/mutable_file.rs`, `hl/dataset_builder.rs`,
    `hl/types.rs`, `hl/selection.rs`, `hl/conversion.rs`,
    `hl/plist/*`) → `H5{F,G,D,A,T,S,L,P,R,I,Z}*` API + `H5*__cache_*`.
  - Format-layer decoders / encoders / lookups (`format/*`,
    `format/messages/*`) → `H5O__*_decode/encode/size`,
    `H5{B,B2,EA,FA,HF,HG,HL,F}__cache_*`, `H5*_iterate`, `H5*__man_op_real`.
  - Engine layer (`engine/{writer,handle,allocator}.rs`) → `H5{I,F,MF}*`.
  - I/O primitives (`io/reader.rs`/`io/writer.rs`) →
    `H5F__{en,de}code_uint{8,16,32,64}` / `H5F_addr_{en,de}code` /
    `H5F_{EN,DE}CODE_LENGTH` / `H5FD_{read,write,seek}`.
  - Filter pipeline (`filters/*`) → `H5Z_pipeline` + `H5Z__filter_{deflate,
    shuffle,fletcher32,scaleoffset,nbit,szip,blosc,lzf}` and helpers.
  - Pure utility helpers (`ensure_available`, `read_le_u64`, `read_u8`,
    `bit_is_set`, `log2_*`, `bytes_needed`, `usize_from_u64`) →
    `UINT*DECODE` macro family / `H5_IS_BUFFER_OVERFLOW` /
    `H5VM_{bit_get,log2_*}` / `H5_ASSIGN_OVERFLOW`.
  - Trait impls (`Display::fmt`, `Error::source`, `From::from`,
    `Default::default`) → `H5E*` / `H5I_init_interface` / `H5F__super_init`.
  - Constructor `new` methods → `H5*_create` / `H5*_init`.
  - `inner` / `inner_mut` accessors → `H5F_get_intent`.
  - Test functions → mapped to the C function under test
    (e.g. `test_lzf_*` → `H5Z__filter_lzf`).
  - Tracehash probe `#[cfg]` companion bodies → mapped to the C
    function whose behavior the probe captures.

- [x] Write-side fusion audit (no-fusion rule, 2026-04-18). Audited
  `hl/mutable_file.rs` and `engine/writer.rs`. Of the four originally
  flagged candidates, two were genuine fusions and have been split
  (encode-half extracted into a pure `encode_*` returning `Vec<u8>`,
  with the wrapper composing alloc + encode + write):
  - `encode_extensible_array_data_block_prefix` ↔ `H5EA__cache_dblock_serialize`
  - `encode_extensible_array_data_block_page` ↔ `H5EA__cache_dblk_page_serialize`
  - `encode_chunk_btree_node` ↔ `H5B__cache_serialize` (extracted from
    `write_chunk_btree_node`).
  Two were false positives: `rewrite_extensible_array_super_block` is
  read-modify-write (patches existing on-disk bytes, no encode-from-scratch
  step to extract); `rewrite_leaf_chunk_btree` is an orchestrator over
  already-separate primitives (`write_btree_entry`, `write_btree_final_key`,
  `rebuild_chunk_btree_from_entries`). The split is allocation-neutral —
  the original code was already building the same `Vec<u8>` internally.

- [ ] ccc-rs limitation: each C function can only be bound once via
  `[[entries]]`, so when two Rust halves correspond to the same
  fused-on-the-C-side deserializer (e.g. `decode_indirect_block` and
  `decode_filtered_indirect_block` both → `H5HF__cache_iblock_deserialize`,
  or `decode_header` and `walk_objects` both →
  `H5HG__cache_heap_deserialize`), only one gets the explicit mapping.
  The other falls through to fingerprint matching. Worth either teaching
  ccc-rs to allow N→1 explicit bindings, or accepting the noise.

### blosc dependency

- [ ] Publish `blosc2-pure-rs` 0.3.0 to crates.io. The dependency in
  `Cargo.toml` now requires `^0.3` (was `^0.2`); a `[patch.crates-io]`
  entry overrides resolution to the in-tree sibling checkout at
  `../blosc2-rs` so local development continues to work, but
  `cargo publish` will fail until 0.3.0 is on crates.io. Drop the
  `[patch.crates-io]` entry once published.
- [x] Instrument Rust-side probes for datatype message decode (`DatatypeMessage::decode`).
- [x] Instrument Rust-side probes for object header message decode (`ObjectHeader::read_at`).
- [x] Instrument Rust-side probes for data layout and filter pipeline decode.
- [x] Add a Rust corpus runner that emits `/tmp/rust.tsv`.
- [x] Add a documented comparator command using `tracehash-compare`.
- [x] Document C-side probe targets for datatype message decode (`H5T*` decode path).
- [x] Document C-side probe targets for object header message decode (`H5O*` decode path).
- [x] Document C-side probe targets for filter pipeline decode and application (`H5Z*`).
- [x] Document C-side probe targets for chunk index resolution: v1 B-tree, v2 B-tree, fixed array, extensible array, and single chunk.
- [x] Document C-side probe targets for fractal heap object lookup and dense link/attribute traversal.
- [x] Defer representative divergence reports until a patched HDF5 C build emits `/tmp/c.tsv`.

## Concerns from ccc-rs cross-language scan (2026-04-17)

Scanned with `ccc-rs compare` and `ccc-rs constants-diff` against
`hdf5/src` using the project's `ccc_mapping.toml`. After clearing false positives
(named constants, optimization unrolling, tracehash gates, scratch-pad
layout literals, and an analyzer hex-parse bug since fixed upstream),
three items remain:

- [x] `format/btree_v2.rs::BTreeV2Header::read_at` now rejects
  `split_pct > 100`, `merge_pct > 100`, and `merge_pct >= split_pct` with
  `InvalidFormat` (matching upstream `H5B2__hdr_init`). Covered by four
  tests in `tests/robustness_test.rs`.
- [x] `hl/file.rs::resolve_path` enforces a 1024-byte per-component cap
  (`MAX_PATH_COMPONENT_LEN`, matching `H5G_TRAVERSE_PATH_MAX`). Covered by
  two tests in `tests/group_test.rs`.
- [x] `engine/writer.rs::build_v2_object_header` dead `8`/`_` match arms
  removed; `chunk0_bytes` is now `match`ed exhaustively over the only
  values it can take (1, 2, 4) with a single `unreachable!()` fallback.
- [x] `format/messages/datatype.rs::DatatypeMessage::decode` now validates
  FixedPoint and BitField `bit_offset` and `precision` against the byte
  size: rejects `precision == 0`, `bit_offset > size*8`, and
  `bit_offset + precision > size*8`, matching upstream
  `H5O__dtype_decode_helper`. Six tests in `tests/robustness_test.rs`
  cover the rejection paths and the canonical-width acceptance paths.
  This validation also caught six pre-existing fixtures in
  `tests/robustness_test.rs` whose `precision` bytes were encoded
  big-endian instead of little-endian per spec; those have been
  corrected.
- [x] `format/messages/datatype.rs::DatatypeMessage::decode` now validates
  FloatingPoint properties: rejects `precision == 0`, `exp_size == 0`,
  `mant_size == 0`, sign bit position outside precision, and
  exp/mantissa location+size overflowing precision. Six tests cover the
  rejection paths plus a canonical IEEE-754 binary32 acceptance test.
- [x] `format/messages/filter_pipeline.rs` v1 decoder now rejects
  filter `name_length` values that are not a multiple of eight, matching
  upstream `H5O__pline_decode`. Covered by one focused test in
  `tests/robustness_test.rs`.
- [x] `format/object_header.rs::read_v1` now cross-checks the declared
  `num_messages` field against the actual decoded count: a v1 object
  header that decodes more non-NIL/non-continuation messages than its
  prefix declares is rejected. Per spec the stored count is an upper
  bound, so the check is `decoded ≤ declared`.
- [x] `format/messages/dataspace.rs::decode_v2` now rejects Scalar and
  Null dataspaces with a non-zero rank, matching upstream
  `H5O__sdspace_decode`'s "invalid rank for scalar or NULL dataspace"
  check. Three tests cover the rejection paths and the canonical
  scalar acceptance path.
- [x] `format/messages/attribute.rs` (v1/v2/v3) now rejects messages
  with `name_size == 0`, matching upstream `H5O__attr_decode`. Covered
  by one test.
- [x] `format/messages/data_layout.rs` (v1/v2/v3/v4) now rejects chunk
  layouts with any chunk dimension equal to zero (matches
  `H5O__layout_decode`'s "chunk dimension must be positive"). Covered
  by one focused v3 test.

Cleared on inspection (recorded so a future scan doesn't re-flag them):

- `format/checksum.rs::fletcher32` — `360` batch size is present (line 47);
  the prior diff was an analyzer artifact, since fixed in ccc-rs.
- `format/superblock.rs::read_v0_v1` — literal `32`/`16`/`16`/`16` are
  `HDF5_BTREE_CHUNK_IK_DEF` and the spec-fixed scratch-pad size.
- `format/btree_v2.rs::read_internal_records` — literals `10` and `11` are
  the chunk-no-filter / chunk-with-filter B-tree types used to gate the
  `tracehash` probe; not magic numbers.
- `format/symbol_table.rs::read_entry` — literal `16`/`12` are the
  spec-fixed `H5G_SIZEOF_SCRATCH = 16` scratch-pad and its remainder for
  `cache_type==2`.
- `filters/shuffle.rs` — does not unroll per-element-size like
  `H5Z__filter_shuffle`; performance trade-off, not correctness.
