# Public API Audit

The root crate exports the stable high-level handles:

- `File`
- `Group`
- `Dataset`
- `Attribute`
- `Datatype`
- `Dataspace`
- `DatasetBuilder`
- `WritableFile`
- `MutableFile`
- `H5Type`
- `H5Value`

The `format`, `engine`, `filters`, and `io` modules are currently public and
therefore expose low-level layout structs such as datatype messages, layout
messages, B-tree helpers, heaps, and writer internals. These are useful for the
faithfulness work and raw inspection APIs, but they should be treated as
unstable unless the crate introduces an explicit semver policy for low-level
modules.

Release rule:

- High-level `hl::*` types re-exported from `lib.rs` are the intended public API.
- Low-level `format::*`, `engine::*`, `filters::*`, and `io::*` exports are
  implementation-facing and may need a future `#[doc(hidden)]`, feature gate,
  or `unstable` module boundary before a compatibility-focused release.
- New raw inspection methods should return cloned parsed message values rather
  than borrowing internal object-header storage.
