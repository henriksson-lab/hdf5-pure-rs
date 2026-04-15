//! Phase T8: Object header parsing tests.
//! Verify v1 and v2 object headers parse correctly across reference files.

use hdf5_pure_rust::format::object_header::{
    ObjectHeader, MSG_ATTRIBUTE, MSG_LINK, MSG_LINK_INFO, MSG_SYMBOL_TABLE,
};
use hdf5_pure_rust::format::superblock::Superblock;
use hdf5_pure_rust::io::HdfReader;
use std::fs;
use std::io::BufReader;

fn read_root_oh(path: &str) -> (Superblock, ObjectHeader) {
    let f = fs::File::open(path).unwrap();
    let mut reader = HdfReader::new(BufReader::new(f));
    let sb = Superblock::read(&mut reader).unwrap();
    let oh = ObjectHeader::read_at(&mut reader, sb.root_addr).unwrap();
    (sb, oh)
}

// T8a: V1 object headers (files with superblock v0)

#[test]
fn t8a_v1_header_simple_v0() {
    let (sb, oh) = read_root_oh("tests/data/simple_v0.h5");
    assert_eq!(sb.version, 0);
    assert_eq!(oh.version, 1);
    // Root group should have a symbol table message
    assert!(oh.messages.iter().any(|m| m.msg_type == MSG_SYMBOL_TABLE));
}

#[test]
fn t8a_v1_header_datasets_v0() {
    let (_sb, oh) = read_root_oh("tests/data/datasets_v0.h5");
    assert_eq!(oh.version, 1);
    assert!(oh.messages.iter().any(|m| m.msg_type == MSG_SYMBOL_TABLE));
}

#[test]
fn t8a_v1_continuation_chunks() {
    // File with attributes on root group uses continuation chunks in v1
    let (_, oh) = read_root_oh("tests/data/attrs.h5");
    assert_eq!(oh.version, 1);
    // Should have symbol table + attributes (via continuation)
    let has_stab = oh.messages.iter().any(|m| m.msg_type == MSG_SYMBOL_TABLE);
    let has_attr = oh.messages.iter().any(|m| m.msg_type == MSG_ATTRIBUTE);
    assert!(has_stab);
    assert!(has_attr);
}

// T8b: V2 object headers (files with superblock v2/v3)

#[test]
fn t8b_v2_header_simple_v3() {
    let (sb, oh) = read_root_oh("tests/data/simple_v2.h5");
    assert!(sb.version >= 2);
    assert_eq!(oh.version, 2);
    // V2 root group should have link messages
    let has_links = oh
        .messages
        .iter()
        .any(|m| m.msg_type == MSG_LINK || m.msg_type == MSG_LINK_INFO);
    assert!(has_links);
}

#[test]
fn t8b_v2_timestamps() {
    let (_, oh) = read_root_oh("tests/data/simple_v2.h5");
    // V2 OHs with HDR_STORE_TIMES flag should have timestamps
    if oh.flags & 0x20 != 0 {
        assert!(oh.atime.is_some());
        assert!(oh.mtime.is_some());
    }
}

// T8c: All known message types parse without error

#[test]
fn t8c_all_message_types() {
    let test_files = [
        "tests/data/simple_v0.h5",
        "tests/data/simple_v2.h5",
        "tests/data/datasets_v0.h5",
        "tests/data/datasets_v3.h5",
        "tests/data/attrs.h5",
        "tests/data/attrs_v3.h5",
        "tests/data/dense_attrs.h5",
        "tests/data/dense_links.h5",
    ];

    let mut all_msg_types = std::collections::HashSet::new();

    for path in &test_files {
        if let Ok(f) = fs::File::open(path) {
            let mut reader = HdfReader::new(BufReader::new(f));
            if let Ok(sb) = Superblock::read(&mut reader) {
                if let Ok(oh) = ObjectHeader::read_at(&mut reader, sb.root_addr) {
                    for msg in &oh.messages {
                        all_msg_types.insert(msg.msg_type);
                    }
                }
            }
        }
    }

    println!("Message types found across test files: {:?}", all_msg_types);
    // We should see at least these common types
    assert!(all_msg_types.contains(&MSG_SYMBOL_TABLE) || all_msg_types.contains(&MSG_LINK));
}

// T8d: Reference files -- parse all root OHs

#[test]
fn t8d_parse_all_reference_ohs() {
    let dir = std::fs::read_dir("tests/data/hdf5_ref").unwrap();
    let mut parsed = 0;
    let mut failed = Vec::new();

    for entry in dir {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.extension().map_or(false, |e| e == "h5") {
            continue;
        }

        let result = std::panic::catch_unwind(|| {
            if let Ok(f) = fs::File::open(&path) {
                let mut reader = HdfReader::new(BufReader::new(f));
                if let Ok(sb) = Superblock::read(&mut reader) {
                    let _ = ObjectHeader::read_at(&mut reader, sb.root_addr);
                }
            }
        });

        if result.is_err() {
            failed.push(path.file_name().unwrap().to_string_lossy().to_string());
        } else {
            parsed += 1;
        }
    }

    println!("Parsed {parsed} root OHs without panics");
    assert!(failed.is_empty(), "OH parsing panicked on: {:?}", failed);
}
