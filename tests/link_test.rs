use hdf5_pure_rust::format::messages::link::LinkType;
use hdf5_pure_rust::{File, WritableFile};

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
