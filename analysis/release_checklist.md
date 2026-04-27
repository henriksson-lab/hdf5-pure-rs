# Release Checklist

- Regenerate checked-in HDF5 fixtures with `scripts/generate-modern-fixtures.py`.
- Record the HDF5 C source revision in `hdf5-source.json`.
- Run the default test matrix locally: `cargo test --workspace`.
- Run feature checks used by CI: `--no-default-features`, `--features derive`, `--features blosc`, and `--features tracehash`.
- Do not rely on the retired tracehash corpus workflow for release validation; use fixture tests and documented unsupported-feature coverage instead.
- Run `cargo check --examples` and `cargo check --benches`.
- Update README feature notes and avoid hard-coding the total test count.
- Review `analysis/compatibility_matrix.md` for any newly supported or removed behavior.
- Audit untracked files against `analysis/repo_artifact_audit.md`.
- Run packaging checks: `cargo package --allow-dirty` for local validation, then repeat without `--allow-dirty` before publishing.
- Run `cargo semver-checks check-release` before publishing.
- Run `cargo deny check` before publishing.
