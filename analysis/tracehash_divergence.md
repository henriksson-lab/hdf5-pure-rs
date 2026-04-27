# Tracehash Notes

The previous Rust-vs-HDF5-C tracehash corpus workflow has been retired. The
patched C harness and Rust corpus binary were not kept in sync with the current
crate surface, so their historical parity output is no longer used as release
evidence.

The `tracehash` Cargo feature remains as a lightweight local-debug hook for
inline probes. It should compile, but it is not a supported parity workflow
until a new corpus runner is built and revalidated from a clean checkout.
