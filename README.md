# hdf5-pure-rust

[![Crates.io](https://img.shields.io/crates/v/hdf5-pure-rust.svg)](https://crates.io/crates/hdf5-pure-rust)
[![License](https://img.shields.io/crates/l/hdf5-pure-rust.svg)](https://github.com/henriksson-lab/hdf5-pure-rs)

Pure Rust implementation of the HDF5 file format. **No C dependencies.**

Read and write HDF5 files without linking to the C HDF5 library. Files produced by this crate are verified compatible with the C library (h5dump, h5py). Can also be compiled for WebAssembly.

Based on HDF5 C library commit [`62701c4`](https://github.com/HDFGroup/hdf5/commit/62701c4c79775d267deedd15ed14d4c09571e792) (2026-04-10, v1.14.x branch).

This is a reimplementation of the HDF5 format, not a wrapper around the C library. Output should be bitwise compatible with the original. Please report any deviations.

## Installation

```toml
[dependencies]
hdf5-pure-rust = "0.1"
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

**Reading:**
- Superblock v0-v3
- Object header v1 and v2 (with checksums)
- All storage layouts: compact, contiguous, chunked
- Chunk indices: v1 B-tree, v2 B-tree, fixed array, extensible array, single chunk
- Filters: deflate, shuffle, fletcher32, NBit, ScaleOffset, LZF, SZip (stub), Blosc (optional)
- All primitive types (i8-i64, u8-u64, f32, f64) with automatic big-endian byte-swap
- Compound and enum datatypes
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
- `MutableFile::open_rw()` for in-place dataset resizing
- Property list queries (`ds.create_plist()`)
- 93% of C library reference test files parse successfully
- Zero panics on corrupt/malformed files (CVE regression tested)

## Benchmark

1M f64 elements, chunked (50K), deflate level 1:

| Operation | h5py/C (v1.14.5) | hdf5-pure-rust | Speedup |
|-----------|------------------:|---------------:|--------:|
| Write     | 68.7 ms           | 42.4 ms        | 1.6x    |
| Read      | 17.3 ms           | 20.4 ms        | 0.85x   |

Write is faster due to less overhead (no C FFI, no HDF5 metadata cache management). Read is slightly slower due to the pure-Rust deflate implementation vs C zlib.

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

## Test Suite

269 tests covering:
- 53/57 C library reference files (93% compatibility)
- All primitive types, compound, enum, strings
- All storage layouts and filter combinations
- Corrupt file handling (zero panics, CVE regressions)
- Write round-trips verified by h5dump and h5py
- Cross-platform: big-endian, old formats, various file space strategies

## License

MIT OR Apache-2.0
