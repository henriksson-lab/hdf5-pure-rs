# Tracehash Corpus Status

Date: 2026-04-27

The old Rust-vs-HDF5-C tracehash corpus runner has been retired. It depended
on a patched C harness and a Rust `tracehash_corpus` binary that no longer
matched the current crate surface, so the previous 22,839-row parity claim is
not treated as current evidence.

The `tracehash` Cargo feature is kept only for inline local probe hooks that
developers may use while debugging. It is not a supported release validation
path until a new corpus runner is designed and revalidated from a clean
checkout.
