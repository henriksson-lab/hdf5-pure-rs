use hdf5_pure_rust::format::messages::link::{LinkMessage, LinkType};
use hdf5_pure_rust::{Error, File, WritableFile};

#[test]
fn test_write_and_read_soft_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_test.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.link_soft("alias", "/real_data");
        wf.flush().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let names = f.member_names().unwrap();
        assert!(names.contains(&"real_data".to_string()));
        assert!(names.contains(&"alias".to_string()));

        let root = f.root_group().unwrap();
        let lt = root.link_type("alias").unwrap();
        assert_eq!(lt, LinkType::Soft);

        let target = root.soft_link_target("alias").unwrap();
        assert_eq!(target, "/real_data");
    }
}

#[test]
fn test_soft_link_resolution_and_cycle_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_resolution.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[10, 20, 30])
            .unwrap();
        wf.create_group("real_group").unwrap();
        wf.link_soft("alias_data", "/real_data");
        wf.link_soft("alias_group", "/real_group");
        wf.link_soft("cycle_a", "/cycle_b");
        wf.link_soft("cycle_b", "/cycle_a");
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let alias_values: Vec<i32> = f.dataset("alias_data").unwrap().read().unwrap();
    assert_eq!(alias_values, vec![10, 20, 30]);
    assert_eq!(f.group("alias_group").unwrap().name(), "/real_group");

    let err = match f.dataset("cycle_a") {
        Ok(_) => panic!("soft-link cycle should hit traversal limit"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(err.to_string().contains("soft link traversal limit"));
}

#[test]
fn test_link_exists() {
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let root = f.root_group().unwrap();
    assert!(root.link_exists("data").unwrap());
    assert!(root.link_exists("group1").unwrap());
    assert!(!root.link_exists("nonexistent").unwrap());
}

#[test]
fn test_write_external_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ext_link_test.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.link_external("remote", "other_file.h5", "/some/dataset");
        wf.flush().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let names = f.member_names().unwrap();
        assert!(names.contains(&"remote".to_string()));

        let root = f.root_group().unwrap();
        let lt = root.link_type("remote").unwrap();
        assert_eq!(lt, LinkType::External);

        let (filename, obj_path) = root.external_link_target("remote").unwrap();
        assert_eq!(filename, "other_file.h5");
        assert_eq!(obj_path, "/some/dataset");
    }
}

#[test]
fn test_external_link_traversal_missing_relative_absolute_and_same_directory() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.h5");
    let target_path = dir.path().join("target.h5");
    let nested_dir = dir.path().join("nested");
    std::fs::create_dir(&nested_dir).unwrap();
    let nested_target_path = nested_dir.join("nested_target.h5");

    {
        let mut target = WritableFile::create(&target_path).unwrap();
        target
            .new_dataset_builder("data")
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        target.create_group("group").unwrap();
        target.flush().unwrap();

        let mut nested = WritableFile::create(&nested_target_path).unwrap();
        nested
            .new_dataset_builder("data")
            .write::<i32>(&[4, 5, 6])
            .unwrap();
        nested.flush().unwrap();

        let mut source = WritableFile::create(&source_path).unwrap();
        source.link_external("same_dir", "target.h5", "/data");
        source.link_external("relative", "nested/nested_target.h5", "/data");
        source.link_external("absolute", target_path.to_str().unwrap(), "/data");
        source.link_external("remote_group", "target.h5", "/group");
        source.link_external("missing", "missing.h5", "/data");
        source.flush().unwrap();
    }

    let f = File::open(&source_path).unwrap();
    assert_eq!(
        f.dataset("same_dir").unwrap().read::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("relative").unwrap().read::<i32>().unwrap(),
        vec![4, 5, 6]
    );
    assert_eq!(
        f.dataset("absolute").unwrap().read::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.group("remote_group").unwrap().name(), "/group");
    assert!(matches!(f.dataset("missing"), Err(Error::Io(_))));
}

#[test]
fn test_utf8_link_names_and_non_ascii_external_filename() {
    let f = File::open("tests/data/hdf5_ref/link_edge_cases.h5").unwrap();
    let root = f.root_group().unwrap();
    let names = root.member_names().unwrap();

    assert!(names.contains(&"猫_group".to_string()));
    assert!(names.contains(&"å_link".to_string()));
    assert!(names.contains(&"external_å".to_string()));
    assert_eq!(
        root.member_type("å_link").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );

    let (filename, object_path) = root.external_link_target("external_å").unwrap();
    assert_eq!(filename, "målfil.h5");
    assert_eq!(object_path, "/dåta");
}

#[test]
fn test_link_decoder_rejects_invalid_character_encoding() {
    let mut raw = vec![1, 0x10, 2, 1, b'x'];
    raw.extend_from_slice(&0u64.to_le_bytes());
    let err = LinkMessage::decode(&raw, 8).expect_err("invalid link cset should fail");
    assert!(matches!(err, Error::InvalidFormat(_)));
}
