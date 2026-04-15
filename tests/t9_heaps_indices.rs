//! Phase T9: Heap and index structure tests.

use hdf5_pure_rust::File;

// T9a: Global heap (variable-length data)

#[test]
fn t9a_global_heap_vlen_strings() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("vlen_ds").unwrap();
    let strings = ds.read_strings().unwrap();
    assert_eq!(strings, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn t9a_global_heap_vlen_attr() {
    // The simple_v2.h5 has a vlen string attribute "test_attr"
    let f = File::open("tests/data/simple_v2.h5").unwrap();
    let names = f.attr_names().unwrap();
    assert!(names.contains(&"test_attr".to_string()));
}

// T9b: Local heap (v1 group name storage)

#[test]
fn t9b_local_heap_names() {
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let names = f.member_names().unwrap();
    // Names come from the local heap
    assert!(names.contains(&"data".to_string()));
    assert!(names.contains(&"group1".to_string()));
}

#[test]
fn t9b_local_heap_large_group() {
    // datasets_v0.h5 has more members
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let names = f.member_names().unwrap();
    assert!(names.len() >= 4); // float64_1d, int32_1d, scalar, int8_2d, chunked
}

// T9c: Fractal heap (dense link/attr storage)

#[test]
fn t9c_fractal_heap_dense_links() {
    let f = File::open("tests/data/dense_links.h5").unwrap();
    let names = f.member_names().unwrap();
    assert_eq!(names.len(), 20);
}

#[test]
fn t9c_fractal_heap_modern_dense_links() {
    let f = File::open("tests/data/hdf5_ref/fractal_heap_modern.h5").unwrap();
    let group = f.group("many_links").unwrap();
    let names = group.member_names().unwrap();
    assert_eq!(names.len(), 80);
    assert!(names.contains(&"link_000".to_string()));
    assert!(names.contains(&"link_079".to_string()));
}

#[test]
fn t9c_fractal_heap_dense_attrs() {
    // dense_attrs.h5 has the "data" dataset via inline link
    let f = File::open("tests/data/dense_attrs.h5").unwrap();
    let names = f.member_names().unwrap();
    assert!(names.contains(&"data".to_string()));
}

// T9d: V2 B-tree (used for dense link name index)

#[test]
fn t9d_v2_btree_link_lookup() {
    let f = File::open("tests/data/dense_links.h5").unwrap();
    // The links are indexed via v2 B-tree + fractal heap
    let root = f.root_group().unwrap();
    // Can find specific groups by name
    let g = root.open_group("group_10").unwrap();
    assert!(g.is_empty().unwrap());
}

// T9e/f: Chunk index structures (tested via dataset reads)

#[test]
fn t9ef_btree_v1_chunk_index() {
    // btree_idx_1_6 and btree_idx_1_8 from C test suite
    let f = File::open("tests/data/hdf5_ref/btree_idx_1_6.h5").unwrap();
    let names = f.member_names().unwrap();
    println!("btree_idx_1_6 members: {names:?}");
    // Just verify it opens and lists without error
    assert!(!names.is_empty());
}

#[test]
fn t9ef_btree_v1_chunk_index_18() {
    let f = File::open("tests/data/hdf5_ref/btree_idx_1_8.h5").unwrap();
    let names = f.member_names().unwrap();
    println!("btree_idx_1_8 members: {names:?}");
    assert!(!names.is_empty());
}

#[test]
fn t9ef_non_default_heap_sizes() {
    let f = File::open("tests/data/hdf5_ref/tsizeslheap.h5").unwrap();
    let names = f.member_names().unwrap();
    println!("tsizeslheap members: {names:?}");
}
