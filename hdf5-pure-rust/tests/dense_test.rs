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
