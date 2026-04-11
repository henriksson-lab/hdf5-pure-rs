use std::fs::File;
use std::io::BufReader;

use hdf5_pure_rust::format::superblock::Superblock;
use hdf5_pure_rust::io::HdfReader;

#[test]
fn test_parse_v0_superblock() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open test file");
    let mut reader = HdfReader::new(BufReader::new(f));
    let sb = Superblock::read(&mut reader).expect("failed to parse superblock");

    assert_eq!(sb.version, 0);
    assert_eq!(sb.sizeof_addr, 8);
    assert_eq!(sb.sizeof_size, 8);
    assert_eq!(sb.sym_leaf_k, 4);
    assert_eq!(sb.snode_btree_k, 16);
    assert_eq!(sb.base_addr, 0);
    assert_eq!(sb.status_flags, 0);
    // Root group object header address should be valid
    assert_ne!(sb.root_addr, u64::MAX);
    println!("v0 superblock: {sb:#?}");
}

#[test]
fn test_parse_v3_superblock() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open test file");
    let mut reader = HdfReader::new(BufReader::new(f));
    let sb = Superblock::read(&mut reader).expect("failed to parse superblock");

    assert_eq!(sb.version, 3);
    assert_eq!(sb.sizeof_addr, 8);
    assert_eq!(sb.sizeof_size, 8);
    assert_eq!(sb.base_addr, 0);
    assert_eq!(sb.status_flags, 0);
    // Root group object header address should be valid
    assert_ne!(sb.root_addr, u64::MAX);
    println!("v3 superblock: {sb:#?}");
}
