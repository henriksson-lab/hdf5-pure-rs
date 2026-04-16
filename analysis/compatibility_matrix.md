# Compatibility Matrix

This matrix maps supported behavior to at least one fixture and regression test.
It is not a claim of full HDF5 compatibility.

| Feature | Fixture | Test |
| --- | --- | --- |
| Superblock v0/v2 reads | `tests/data/simple_v0.h5`, `tests/data/simple_v2.h5` | `test_list_root_members_v0`, `test_list_root_members_v3` |
| Primitive contiguous datasets | `tests/data/datasets_v0.h5` | `test_read_typed_f64`, `test_read_typed_i32` |
| Compact datasets | `tests/data/hdf5_ref/compact_read_cases.h5` | `test_compact_zero_sized_dataset_read`, `test_compact_scalar_compound_payload_read` |
| Chunked v1 B-tree datasets | `tests/data/datasets_v0.h5` | `test_read_chunked_typed` |
| v4 fixed-array chunk indexes | `tests/data/hdf5_ref/v4_fixed_array_chunks.h5` | `test_v4_fixed_array_chunks_read` |
| v4 extensible-array chunk indexes | `tests/data/hdf5_ref/v4_extensible_array_chunks.h5` | `test_v4_extensible_array_chunks_read` |
| v4 v2-B-tree chunk indexes | `tests/data/hdf5_ref/v4_btree2_chunks.h5` | `test_v4_btree2_chunks_read` |
| Filter pipelines | generated writer fixtures and reference files | `test_write_chunked_with_shuffle_and_deflate`, `test_filtered_fractal_heap_direct_object_read_fletcher32` |
| Variable-length strings via global heap | `tests/data/strings.h5` | `test_read_vlen_strings` |
| Dense groups and attributes | `tests/data/hdf5_ref/dense_group_cases.h5`, `tests/data/hdf5_ref/dense_attr_cases.h5` | `t9c_fractal_heap_indirect_growth_beyond_one_level`, dense attribute tests |
| Soft and external links | generated link fixtures | `test_soft_link_resolution_and_cycle_limit`, `test_external_link_traversal_missing_relative_absolute_and_same_directory` |
| Virtual datasets | `tests/data/hdf5_ref/vds_all.h5` and related VDS fixtures | `test_virtual_dataset_all_selection`, VDS regression tests |
| Writer contiguous/compact/chunked datasets | temporary files from writer tests | `test_writable_file_simple`, `test_writable_file_compact`, `test_writable_file_chunked_compressed` |
| Mutable v1 chunk replacement | temporary files from resize tests | `test_write_chunk_replaces_existing_chunk`, `test_write_chunk_replaces_filtered_chunk` |
| Mutable v4 fixed-array replacement | `tests/data/hdf5_ref/v4_fixed_array_chunks.h5` copy | `test_write_chunk_replaces_existing_v4_fixed_array_chunk` |
