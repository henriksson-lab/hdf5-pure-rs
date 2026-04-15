# hdf5-pure-rust

[![Crates.io](https://img.shields.io/crates/v/hdf5-pure-rust.svg)](https://crates.io/crates/hdf5-pure-rust)
[![License](https://img.shields.io/crates/l/hdf5-pure-rust.svg)](https://github.com/henriksson-lab/hdf5-pure-rs)

Pure Rust implementation of the HDF5 file format. 

Based on HDF5 C library commit [`62701c4`](https://github.com/HDFGroup/hdf5/commit/62701c4c79775d267deedd15ed14d4c09571e792) (2026-04-10, v1.14.x branch).

**This crate is still under construction, with more testing and benchmarking needed. Be careful in using it for production**



## This is an LLM-mediated faithful (hopefully) translation, not the original code!

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this page](https://rewrites.bio/)
but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back to the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information



## Installation

```toml
[dependencies]
hdf5-pure-rust = "0.2.2"
```

## Quick Start

```rust
use hdf5_pure_rust::{File, WritableFile};

// Write
let mut wf = WritableFile::create("data.h5")?;
wf.new_dataset_builder("temperatures")
    .shape(&[1000])
    .chunk(&[100])
    .deflate(4)
    .write::<f64>(&values)?;
wf.flush()?;

// Read
let f = File::open("data.h5")?;
let ds = f.dataset("temperatures")?;
let values: Vec<f64> = ds.read::<f64>()?;

// Typed reads with ndarray
let arr = ds.read_1d::<f64>()?;        // Array1<f64>
let mat = ds.read_2d::<i32>()?;        // Array2<i32>

// Slicing
let subset: Vec<f64> = ds.read_slice::<f64, _>(10..20)?;

// Strings
let strings = ds.read_strings()?;       // Vec<String>

// Compound types
let x_vals: Vec<f64> = ds.read_field::<f64>("x")?;
```

## Features

| Area | Supported | Explicitly Unsupported |
|------|-----------|------------------------|
| Superblocks and object headers | Superblock v0-v3; object header v1/v2 with checksums | Full C-library metadata-cache behavior |
| Dataset storage | Compact, contiguous, chunked with v1 B-tree, v4 single-chunk datasets, unfiltered v4 implicit chunk indexes, v4 fixed-array chunk indexes, v4 extensible-array chunk indexes including data/super-block spillover, v4 v2-B-tree chunk indexes, and virtual datasets with all-selection or regular hyperslab mappings | Virtual dataset point/irregular selections and full VDS access-property behavior |
| Filters | Deflate, shuffle, fletcher32, LZF, NBit, ScaleOffset, optional Blosc | SZip, unknown filters |
| Datatypes | Primitive numeric types, enum metadata, fixed/vlen strings, compound metadata, primitive compound field reads, raw compound member extraction, recursive compound field values for nested compound/array/vlen/reference members | Full HDF5 datatype conversion parity |
| Groups and links | v1 symbol tables, v2 link messages, dense link/attribute storage, filtered direct fractal heap reads, filtered and unfiltered huge direct/indirect fractal heap reads, soft/external links | Full coverage of every HDF5 index/storage variant |
| Writing | v2 superblock, groups, datasets, attributes, compact/contiguous/chunked storage, deflate/shuffle, soft/external links, limited `MutableFile::write_chunk` append/replace/rebuild for v1 chunk B-trees, and replacement of existing v4 fixed-array chunks | General-purpose HDF5 writer parity with the C library |

**Reading:**
- Superblock v0-v3
- Object header v1 and v2 (with checksums)
- All storage layouts: compact, contiguous, chunked
- Chunk indices: v1 B-tree, single chunk, unfiltered v4 implicit, v4 fixed array, v4 extensible array including data/super-block spillover, and v4 v2-B-tree including internal nodes.
- Virtual datasets with serialized all-selection or regular hyperslab source and destination selections.
- Filters: deflate, shuffle, fletcher32, LZF, NBit, ScaleOffset, and optional Blosc. SZip and unknown filters return `Unsupported` for reads.
- All primitive types (i8-i64, u8-u64, f32, f64) with automatic big-endian byte-swap
- Compound and enum datatypes
- Raw compound field extraction and recursive compound field values for non-primitive member payloads
- Fixed-length and variable-length strings (via global heap)
- Groups with v1 symbol tables and v2 link messages
- Dense link/attribute storage (fractal heap + v2 B-tree)
- Soft and external links
- Hyperslab selections: `ds.read_slice::<f64>(10..20)`
- ndarray integration: `ds.read_1d()`, `ds.read_2d()`

**Writing:**
- v2 superblock with Jenkins lookup3 checksums
- Groups, nested groups, datasets, attributes
- Contiguous, compact, and chunked storage
- Deflate and shuffle compression
- Soft and external links
- Verified readable by h5dump and h5py

**Other:**
- `#[derive(H5Type)]` for user-defined structs and enums
- `MutableFile::open_rw()` for limited in-place dataset resizing, v1 chunk B-tree append/replace/rebuild, and existing v4 fixed-array chunk replacement.
- Property list queries (`ds.create_plist()`)
- Most checked-in C-library reference files parse successfully; the exact count is enforced by tests rather than treated as a general compatibility guarantee.
- Zero panics on corrupt/malformed files (CVE regression tested)

## Benchmark

1M f64 elements, chunked (50K), deflate level 1:

| Operation | h5py/C (v1.14.5) | hdf5-pure-rust | Speedup |
|-----------|------------------:|---------------:|--------:|
| Write     | 68.7 ms           | 42.4 ms        | 1.6x    |
| Read      | 17.3 ms           | 20.4 ms        | 0.85x   |

Write is faster due to less overhead (no C FFI, no HDF5 metadata cache management). Read is slightly slower due to the pure-Rust deflate implementation vs C zlib.

**These benchmarks must be taken with a huge grain of salt. HDF5 is a large complex library with many features, so these benchmarks are primarily intentended to guide further development and track regression**

## Derive Macro

```rust
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Measurement {
    time: f64,
    value: f32,
    #[hdf5(rename = "error_margin")]
    error: f32,
}
```

## Optional Features

| Feature | Default | Description |
|---------|---------|-------------|
| `derive` | yes | `#[derive(H5Type)]` proc macro |
| `blosc`  | no  | Blosc decompression via [`blosc2-pure-rs`](https://crates.io/crates/blosc2-pure-rs) |
| `tracehash` | no | Local development probes for Rust-vs-HDF5-C parity tracing. See `analysis/tracehash_divergence.md`. |

## Test Suite

294 tests covering:
- Selected C library reference files and generated fixtures
- All primitive types, compound, enum, strings
- All storage layouts and filter combinations
- Corrupt file handling (zero panics, CVE regressions)
- Write round-trips verified by h5dump and h5py
- Cross-platform: big-endian, old formats, various file space strategies

**This test suite needs to be expanded before any claims of general compatibility.**

Unsupported HDF5 features are tracked in `analysis/unsupported_features.md`.


## How to Cite HDF5

If you use HDF5 in your research, please cite it. See the original [original code](https://github.com/HDFGroup/hdf5) for details

**Quick DOI:** [10.5281/zenodo.17808558](https://doi.org/10.5281/zenodo.17808558)


## License

This is [derived work](https://github.com/HDFGroup/hdf5) and the license follows from the original HDF5 (BSD-3-Clause).
See the LICENSE file
