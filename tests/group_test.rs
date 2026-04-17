use hdf5_pure_rust::File;

#[test]
fn test_list_root_members_v0() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    let names = f.member_names().expect("failed to list members");
    println!("v0 root members: {names:?}");

    assert!(names.contains(&"data".to_string()));
    assert!(names.contains(&"group1".to_string()));
}

#[test]
fn test_list_root_members_v3() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open v3 file");
    let names = f.member_names().expect("failed to list members");
    println!("v3 root members: {names:?}");

    assert!(names.contains(&"data".to_string()));
    assert!(names.contains(&"group1".to_string()));
}

#[test]
fn test_open_subgroup_v0() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    let g = f.group("group1").expect("failed to open group1");
    assert_eq!(g.name(), "/group1");

    let members = g.member_names().expect("failed to list group1 members");
    println!("v0 group1 members: {members:?}");
    assert!(members.is_empty()); // group1 is empty
}

#[test]
fn test_open_subgroup_v3() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open v3 file");
    let g = f.group("group1").expect("failed to open group1");
    assert_eq!(g.name(), "/group1");

    let members = g.member_names().expect("failed to list group1 members");
    println!("v3 group1 members: {members:?}");
    assert!(members.is_empty());
}

#[test]
fn test_member_types_v0() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    let root = f.root_group().expect("failed to get root");

    let data_type = root
        .member_type("data")
        .expect("failed to get type of data");
    let group_type = root
        .member_type("group1")
        .expect("failed to get type of group1");

    println!("v0: data={data_type:?}, group1={group_type:?}");
    assert_eq!(data_type, hdf5_pure_rust::hl::file::ObjectType::Dataset);
    assert_eq!(group_type, hdf5_pure_rust::hl::file::ObjectType::Group);
}

#[test]
fn test_member_types_v3() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open v3 file");
    let root = f.root_group().expect("failed to get root");

    let data_type = root
        .member_type("data")
        .expect("failed to get type of data");
    let group_type = root
        .member_type("group1")
        .expect("failed to get type of group1");

    println!("v3: data={data_type:?}, group1={group_type:?}");
    assert_eq!(data_type, hdf5_pure_rust::hl::file::ObjectType::Dataset);
    assert_eq!(group_type, hdf5_pure_rust::hl::file::ObjectType::Group);
}

#[test]
fn test_group_len() {
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let root = f.root_group().unwrap();
    assert_eq!(root.len().unwrap(), 2); // "data" and "group1"
    assert!(!root.is_empty().unwrap());

    let g1 = f.group("group1").unwrap();
    assert_eq!(g1.len().unwrap(), 0);
    assert!(g1.is_empty().unwrap());
}

#[test]
fn test_path_component_length_cap_rejects_oversized_segment() {
    // A single path component longer than 1024 bytes must be rejected
    // before traversal starts. The shape of the rest of the path doesn't
    // matter; we just need to confirm the cap fires with the documented
    // error rather than returning a generic "not found".
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let huge = "a".repeat(1025);
    let msg = match f.group(&huge) {
        Ok(_) => panic!("oversized component must not resolve"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("path component exceeds 1024-byte limit"),
        "expected length-cap error, got: {msg}"
    );
}

#[test]
fn test_path_component_length_cap_accepts_at_limit() {
    // Exactly 1024 bytes must NOT trigger the cap (it's a strict >, not >=).
    // The lookup will of course fail with a "not found" error — we just
    // assert the failure mode is *not* the length-cap one.
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let at_limit = "a".repeat(1024);
    let msg = match f.group(&at_limit) {
        Ok(_) => panic!("a 1024-byte component should not resolve in this fixture"),
        Err(e) => format!("{e}"),
    };
    assert!(
        !msg.contains("path component exceeds"),
        "1024-byte component should pass the cap, but got: {msg}"
    );
}
