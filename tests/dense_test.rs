use hdf5_pure_rust::File;

#[test]
fn test_read_dense_links() {
    let f = File::open("tests/data/dense_links.h5").expect("failed to open dense links file");
    let names = f.member_names().expect("failed to list members");
    println!("Dense link members ({}):", names.len());
    for n in &names {
        println!("  {n}");
    }

    assert_eq!(names.len(), 20);
    for i in 0..20 {
        let expected = format!("group_{i:02}");
        assert!(names.contains(&expected), "missing {expected}");
    }
}

#[test]
fn test_read_dense_links_open_group() {
    let f = File::open("tests/data/dense_links.h5").expect("failed to open dense links file");

    let g = f.group("group_05").expect("failed to open group_05");
    assert_eq!(g.name(), "/group_05");
    assert!(g.is_empty().unwrap());
}

#[test]
fn test_read_dense_attrs_file() {
    // The dense_attrs.h5 file has 20 attributes on the root group
    // and a "data" dataset child
    let f = File::open("tests/data/dense_attrs.h5").expect("failed to open dense attrs file");

    let names = f.member_names().expect("failed to list members");
    println!("Dense attrs file members: {names:?}");
    assert!(names.contains(&"data".to_string()));
}

#[test]
fn test_dense_group_multiple_v2_btree_levels_name_index() {
    let f = File::open("tests/data/hdf5_ref/dense_group_cases.h5").unwrap();
    let group = f.group("name_index_deep").unwrap();
    let names = group.member_names().unwrap();

    assert_eq!(names.len(), 4096);
    for idx in [0, 1, 1023, 2048, 4095] {
        let name = format!("link_{idx:04}");
        assert!(names.contains(&name), "missing {name}");
        assert_eq!(
            group.member_type(&name).unwrap(),
            hdf5_pure_rust::hl::file::ObjectType::Dataset
        );
    }
}

#[test]
fn test_dense_group_creation_order_indexing_enabled_and_disabled() {
    let f = File::open("tests/data/hdf5_ref/dense_group_cases.h5").unwrap();

    let tracked = f.group("creation_order_tracked").unwrap();
    let tracked_names = tracked.member_names().unwrap();
    assert_eq!(tracked_names.len(), 64);
    assert!(tracked_names.contains(&"tracked_00".to_string()));
    assert!(tracked_names.contains(&"tracked_63".to_string()));
    assert_eq!(
        tracked.member_type("tracked_42").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );

    let untracked = f.group("creation_order_untracked").unwrap();
    let untracked_names = untracked.member_names().unwrap();
    assert_eq!(untracked_names.len(), 64);
    assert!(untracked_names.contains(&"untracked_00".to_string()));
    assert!(untracked_names.contains(&"untracked_63".to_string()));
    assert_eq!(
        untracked.member_type("untracked_42").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );
}
