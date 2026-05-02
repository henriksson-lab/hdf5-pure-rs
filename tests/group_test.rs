use hdf5_pure_rust::{File, FileCloseDegree, FileIntent, LibverBound};

#[test]
fn test_file_size_matches_filesystem_metadata() {
    let path = "tests/data/simple_v0.h5";
    let f = File::open(path).expect("failed to open v0 file");
    let expected = std::fs::metadata(path).unwrap().len();

    assert_eq!(f.file_size().unwrap(), expected);
}

#[test]
fn test_file_path_returns_open_path() {
    let path = std::path::PathBuf::from("tests/data/simple_v0.h5");
    let f = File::open(&path).expect("failed to open v0 file");

    assert_eq!(f.path().unwrap(), path);
}

#[test]
fn test_file_metadata_and_access_queries() {
    let path = "tests/data/simple_v0.h5";
    let f = File::open(path).expect("failed to open v0 file");
    let image = f.file_image().unwrap();

    assert_eq!(f.intent(), FileIntent::ReadOnly);
    assert_eq!(f.eoa(), f.superblock().eof_addr);
    assert_eq!(f.freespace(), 0);
    let info = f.info().unwrap();
    assert_eq!(info, f.info_v1().unwrap());
    assert_eq!(info.superblock.version, f.superblock().version);
    assert_eq!(
        info.superblock.size,
        f.superblock().checked_size().unwrap() as u64
    );
    assert_eq!(info.free_space.total_space, 0);
    let access = f.access_plist();
    assert_eq!(f.mdc_config(), access.mdc_config());
    assert_eq!(f.mdc_hit_rate(), 0.0);
    assert_eq!(f.mdc_size().current_size, 0);
    assert_eq!(f.mdc_logging_status(), (false, false));
    assert_eq!(f.page_buffering_stats().raw_data_accesses, 0);
    assert_eq!(f.mdc_image_info().size, 0);
    assert!(!f.dset_no_attrs_hint());
    assert!(!f.mpi_atomicity());
    assert_eq!(image.len() as u64, f.file_size().unwrap());
    assert_eq!(&image[..8], b"\x89HDF\r\n\x1a\n");
    assert!(f.fileno().unwrap() > 0);
    #[cfg(unix)]
    assert!(f.vfd_handle().unwrap() >= 0);

    assert_eq!(access.driver(), "sec2");
    assert_eq!(access.driver_info(), None);
    assert_eq!(access.userblock(), 0);
    assert_eq!(access.alignment(), (1, 1));
    assert_eq!(access.cache(), (0, 521, 1024 * 1024, 0.75));
    assert!(!access.gc_references());
    assert_eq!(access.fclose_degree(), FileCloseDegree::Weak);
    assert_eq!(access.meta_block_size(), 2048);
    assert_eq!(access.sieve_buf_size(), 64 * 1024);
    assert_eq!(access.small_data_block_size(), 2048);
    assert_eq!(
        access.libver_bounds(),
        (LibverBound::Earliest, LibverBound::Latest)
    );
    assert!(!access.evict_on_close());
    assert_eq!(access.file_locking(), (true, false));
    assert_eq!(access.mdc_config().max_size, 0);
    assert!(!access.mdc_image_config().enabled);
    assert!(!access.mdc_log_options().enabled);
    assert!(!access.all_coll_metadata_ops());
    assert!(!access.coll_metadata_write());
    assert_eq!(access.page_buffer_size(), (0, 0, 0));
    assert_eq!(access.fapl_hdfs(), None);
    assert_eq!(access.fapl_direct(), None);
    assert_eq!(access.fapl_mirror(), None);
    assert_eq!(access.fapl_mpio(), None);
    assert_eq!(access.dxpl_mpio(), None);
    assert_eq!(access.fapl_family(), None);
    assert_eq!(access.family_offset(), None);
    assert_eq!(access.multi_type(), None);
    assert_eq!(access.fapl_ioc(), None);
    assert_eq!(access.fapl_subfiling(), None);
    assert_eq!(access.fapl_splitter(), None);
    assert_eq!(access.fapl_multi(), None);
    assert_eq!(access.fapl_onion(), None);
    assert!(!access.core_write_tracking());
    assert_eq!(access.fapl_core(), None);
    assert_eq!(access.fapl_ros3(), None);
    assert_eq!(access.fapl_ros3_endpoint(), None);
    assert_eq!(access.object_flush_cb(), None);
    assert_eq!(access.mpi_params(), None);
    assert_eq!(access.vol_id(), None);
    assert_eq!(access.vol_info(), None);
    assert_eq!(access.vol_cap_flags(), 0);
    assert_eq!(access.relax_file_integrity_checks(), 0);
    assert_eq!(access.map_iterate_hints(), None);
}

#[test]
fn test_file_open_object_registry_queries() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    assert_eq!(f.obj_count(), 1);
    assert_eq!(f.obj_ids(), vec![f.object_id()]);

    {
        let group = f.group("group1").unwrap();
        let dataset = f.dataset("data").unwrap();
        let mut ids = f.obj_ids();
        ids.sort_unstable();
        assert_eq!(f.obj_count(), 3);
        assert!(ids.contains(&f.object_id()));
        assert!(ids.contains(&group.object_id()));
        assert!(ids.contains(&dataset.object_id()));
    }

    assert_eq!(f.obj_count(), 1);
    assert_eq!(f.obj_ids(), vec![f.object_id()]);
}

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
