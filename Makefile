all:
publish:
	cargo ws publish

ccc_tui:
	/home/mahogny/github/claude/code-complexity-comparator/target/release/ccc-rs compare-tui --rust-root src --other-root hdf5/src --other-lang c

# ---------------------------------------------------------------------------
# Lines-of-code comparison — original C reference vs Rust port.
# Counts non-blank, non-comment lines. Skips test files / dirs.
# ---------------------------------------------------------------------------

# Awk filter that strips `// ...` line comments, `/* ... */` block comments
# (single-line and multi-line), and counts only lines with leftover content.
# Same comment grammar works for C and Rust.
define LOC_AWK
BEGIN { in_block = 0 }
{
  l = $$0; out = ""; i = 1
  while (i <= length(l)) {
    if (in_block) {
      p = index(substr(l, i), "*/")
      if (p == 0) { i = length(l) + 1 }
      else { i = i + p + 1; in_block = 0 }
    } else {
      c = substr(l, i, 2)
      if (c == "/*") { in_block = 1; i = i + 2 }
      else if (c == "//") { i = length(l) + 1 }
      else { out = out substr(l, i, 1); i = i + 1 }
    }
  }
  gsub(/^[ \t]+|[ \t]+$$/, "", out)
  if (out != "") n = n + 1
}
END { print n }
endef
export LOC_AWK

# Reference C source from libhdf5. Skips:
#   - hdf5/test, hdf5/testpar           (full test trees)
#   - hdf5/src/H5*test.c                (per-package internal tests)
.PHONY: loc-c
loc-c:
	@find hdf5/src -type f \( -name '*.c' -o -name '*.h' \) \
		! -name 'H5*test.c' \
		! -name 'H5*tst.c' \
		-print0 \
		| xargs -0 cat \
		| awk "$$LOC_AWK" \
		| awk '{ printf "C   (libhdf5/src):  %s lines\n", $$1 }'

# Rust port. Skips:
#   - tests/                            (integration tests directory)
#   - src/bin                           (CLI / driver binaries)
.PHONY: loc-rust
loc-rust:
	@find src -type f -name '*.rs' \
		! -path 'src/bin/*' \
		-print0 \
		| xargs -0 cat \
		| awk "$$LOC_AWK" \
		| awk '{ printf "Rust (src/):       %s lines\n", $$1 }'

# Convenience: run both and print a ratio.
.PHONY: loc
loc:
	@$(MAKE) -s loc-c
	@$(MAKE) -s loc-rust
