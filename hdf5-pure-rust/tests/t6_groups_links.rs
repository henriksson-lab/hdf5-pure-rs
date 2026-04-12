//! Phase T6: Group and link tests.

use hdf5_pure_rust::File;

const FILE: &str = "tests/data/hdf5_ref/groups_and_links.h5";
fn open() -> File { File::open(FILE).unwrap() }

// T6a: Nested groups

#[test]
fn t6a_navigate_nested() {
    let f = open();
    let g = f.group("a").unwrap();
    let names = g.member_names().unwrap();
    assert!(names.contains(&"b".to_string()));
    assert!(names.contains(&"e".to_string()));

    let gb = f.group("a/b").unwrap();
    let names = gb.member_names().unwrap();
    assert!(names.contains(&"c".to_string()));
    assert!(names.contains(&"d".to_string()));
}

#[test]
fn t6a_deep_dataset() {
    let ds = open().dataset("a/b/c/data").unwrap();
    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

// T6b: Links

#[test]
fn t6b_hard_link_alias() {
    let f = open();
    // alias_data is a hard link to same object as /a/b/c/data
    let names = f.member_names().unwrap();
    assert!(names.contains(&"alias_data".to_string()));
    // Should be readable as a dataset
    let ds = f.dataset("alias_data").unwrap();
    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn t6b_soft_link() {
    let f = open();
    let names = f.member_names().unwrap();
    assert!(names.contains(&"soft_link".to_string()));

    let root = f.root_group().unwrap();
    let target = root.soft_link_target("soft_link").unwrap();
    assert_eq!(target, "/a/b/c/data");
}

#[test]
fn t6b_external_link() {
    let f = open();
    let root = f.root_group().unwrap();
    let (filename, obj_path) = root.external_link_target("ext_link").unwrap();
    assert_eq!(filename, "other_file.h5");
    assert_eq!(obj_path, "/some/path");
}

#[test]
fn t6b_link_exists() {
    let root = open().root_group().unwrap();
    assert!(root.link_exists("a").unwrap());
    assert!(root.link_exists("alias_data").unwrap());
    assert!(root.link_exists("soft_link").unwrap());
    assert!(root.link_exists("ext_link").unwrap());
    assert!(!root.link_exists("nonexistent").unwrap());
}

// T6c: Many groups (may trigger dense storage)

#[test]
fn t6c_many_groups() {
    let f = open();
    let names = f.member_names().unwrap();
    for i in 0..15 {
        let expected = format!("dense_{i:02}");
        assert!(names.contains(&expected), "missing {expected}");
    }
}

// T6d: Empty groups

#[test]
fn t6d_empty_group() {
    let g = open().group("a/b/d").unwrap();
    assert!(g.is_empty().unwrap());
}
