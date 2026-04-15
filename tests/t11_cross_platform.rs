//! Phase T11: Cross-platform compatibility tests.

use hdf5_pure_rust::File;

const REF_DIR: &str = "tests/data/hdf5_ref";

// T11a: Big-endian files on little-endian host

#[test]
fn t11a_be_data_read() {
    let f = File::open(&format!("{REF_DIR}/be_data.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("be_data.h5 members: {names:?}");
    // Verify we can navigate the file
    assert!(!names.is_empty());
}

#[test]
fn t11a_be_extlinks() {
    // Big-endian external link files
    let f1 = File::open(&format!("{REF_DIR}/be_extlink1.h5")).unwrap();
    let names1 = f1.member_names().unwrap();
    println!("be_extlink1 members: {names1:?}");

    let f2 = File::open(&format!("{REF_DIR}/be_extlink2.h5")).unwrap();
    let names2 = f2.member_names().unwrap();
    println!("be_extlink2 members: {names2:?}");
}

#[test]
fn t11a_be_filters() {
    let f = File::open(&format!("{REF_DIR}/test_filters_be.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("test_filters_be.h5 members: {names:?}");
    assert!(!names.is_empty());
}

#[test]
fn t11a_typed_read_be() {
    // Read big-endian data with byte-swap
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("be_float").unwrap();
    let vals: Vec<f64> = ds.read::<f64>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

// T11b: Old format files

#[test]
fn t11b_old_group_format() {
    let f = File::open(&format!("{REF_DIR}/group_old.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("group_old.h5 members: {names:?}");
    assert!(!names.is_empty());
}

#[test]
fn t11b_old_fill_values() {
    let f = File::open(&format!("{REF_DIR}/fill_old.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("fill_old members: {names:?}");
}

#[test]
fn t11b_old_layout() {
    let f = File::open(&format!("{REF_DIR}/tlayouto.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("tlayouto members: {names:?}");
}

#[test]
fn t11b_old_mtime() {
    for name in ["tmtimen.h5", "tmtimeo.h5"] {
        let f = File::open(&format!("{REF_DIR}/{name}")).unwrap();
        let names = f.member_names().unwrap();
        println!("{name} members: {names:?}");
    }
}

#[test]
fn t11b_old_array() {
    let f = File::open(&format!("{REF_DIR}/tarrold.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("tarrold members: {names:?}");
}

// T11c: Various file space strategies

#[test]
fn t11c_filespace_strategies() {
    for name in [
        "filespace_1_6.h5",
        "filespace_1_8.h5",
        "paged_nopersist.h5",
        "paged_persist.h5",
        "fsm_aggr_nopersist.h5",
        "fsm_aggr_persist.h5",
        "aggr.h5",
    ] {
        let f = File::open(&format!("{REF_DIR}/{name}")).unwrap();
        let sb = f.superblock();
        println!(
            "{name}: sb_version={}, sizeof_addr={}, sizeof_size={}",
            sb.version, sb.sizeof_addr, sb.sizeof_size
        );
    }
}

// T11d: Deflate filter files from C test suite

#[test]
fn t11d_deflate_reference() {
    let f = File::open(&format!("{REF_DIR}/deflate.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("deflate.h5 members: {names:?}");
    // Try reading a dataset if available
    for name in &names {
        if let Ok(ds) = f.root_group().unwrap().open_dataset(name) {
            let shape = ds.shape().unwrap();
            println!("  {name}: shape={shape:?}");
        }
    }
}

// T11e: Charsets

#[test]
fn t11e_charsets() {
    let f = File::open(&format!("{REF_DIR}/charsets.h5")).unwrap();
    let names = f.member_names().unwrap();
    println!("charsets.h5 members: {names:?}");
}
