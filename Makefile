all:
publish:
	cargo ws publish

ccc_tui:
	/home/mahogny/github/claude/code-complexity-comparator/target/release/ccc-rs compare-tui --rust-root src --other-root hdf5/src --other-lang c
