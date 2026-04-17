#!/usr/bin/env sh
set -eu

RUST_TRACE="${1:-/tmp/rust.tsv}"
C_TRACE="${2:-/tmp/c.tsv}"

# The `tracehash-compare` binary is shipped by the `tracehash-rs` crate on
# crates.io. `cargo install tracehash-rs` places it on $PATH; we fall back
# to `cargo run --package tracehash-rs --bin tracehash-compare` so CI
# doesn't need a pre-install step.
if command -v tracehash-compare >/dev/null 2>&1; then
  tracehash-compare "${RUST_TRACE}" "${C_TRACE}"
else
  cargo run --quiet --release --package tracehash-rs --bin tracehash-compare -- \
    "${RUST_TRACE}" "${C_TRACE}"
fi
