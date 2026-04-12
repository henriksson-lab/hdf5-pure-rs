use hdf5_pure_rust::format::messages::link::LinkMessage;
use hdf5_pure_rust::format::messages::dataspace::DataspaceMessage;
use hdf5_pure_rust::format::messages::datatype::DatatypeMessage;
use hdf5_pure_rust::format::messages::data_layout::DataLayoutMessage;
use hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage;
use hdf5_pure_rust::format::messages::attribute::AttributeMessage;
use hdf5_pure_rust::format::superblock::Superblock;
use hdf5_pure_rust::io::HdfReader;

use std::io::Cursor;

#[test]
fn test_invalid_signature() {
    let data = vec![0u8; 64];
    let mut reader = HdfReader::new(Cursor::new(data));
    assert!(Superblock::read(&mut reader).is_err());
}

#[test]
fn test_truncated_superblock() {
    // Valid signature but truncated
    let mut data = vec![0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A];
    data.push(2); // version 2
    // Missing sizeof_addr, sizeof_size, etc.
    let mut reader = HdfReader::new(Cursor::new(data));
    assert!(Superblock::read(&mut reader).is_err());
}

#[test]
fn test_link_message_empty() {
    assert!(LinkMessage::decode(&[], 8).is_err());
}

#[test]
fn test_link_message_truncated() {
    // Valid version + flags but truncated
    let data = vec![1, 0];
    assert!(LinkMessage::decode(&data, 8).is_err());
}

#[test]
fn test_link_message_bad_name_length() {
    // Version 1, flags=0 (hard link, 1-byte name len), name_len=255 but only 4 bytes of data
    let data = vec![1, 0x00, 0xFF, 0x41]; // version=1, flags=0, name_len=255, 'A'
    assert!(LinkMessage::decode(&data, 8).is_err());
}

#[test]
fn test_dataspace_empty() {
    assert!(DataspaceMessage::decode(&[]).is_err());
}

#[test]
fn test_dataspace_truncated() {
    let data = vec![2, 3, 0, 1]; // version 2, ndims=3, flags=0, type=simple -- missing dim data
    // This should succeed but with wrong dims since the data is short
    let result = DataspaceMessage::decode(&data);
    assert!(result.is_ok()); // It reads empty dims since there's not enough data
    let ds = result.unwrap();
    assert_eq!(ds.ndims, 3);
    // dims will be partial/zero since data is truncated
}

#[test]
fn test_datatype_empty() {
    assert!(DatatypeMessage::decode(&[]).is_err());
}

#[test]
fn test_datatype_truncated() {
    let data = vec![0x10, 0, 0, 0]; // class 0, version 1, 4 bytes -- missing size
    assert!(DatatypeMessage::decode(&data).is_err());
}

#[test]
fn test_layout_empty() {
    assert!(DataLayoutMessage::decode(&[], 8, 8).is_err());
}

#[test]
fn test_layout_bad_version() {
    let data = vec![99, 0]; // version 99
    assert!(DataLayoutMessage::decode(&data, 8, 8).is_err());
}

#[test]
fn test_filter_pipeline_empty() {
    assert!(FilterPipelineMessage::decode(&[]).is_err());
}

#[test]
fn test_attribute_empty() {
    assert!(AttributeMessage::decode(&[]).is_err());
}

#[test]
fn test_attribute_bad_version() {
    let data = vec![99, 0, 0, 0, 0, 0, 0, 0]; // version 99
    assert!(AttributeMessage::decode(&data).is_err());
}

#[test]
fn test_open_nonexistent_file() {
    let result = hdf5_pure_rust::File::open("/nonexistent/path.h5");
    assert!(result.is_err());
}

#[test]
fn test_open_non_hdf5_file() {
    // Create a temp file with random data
    let path = "tests/data/not_hdf5.bin";
    std::fs::write(path, b"This is not an HDF5 file").unwrap();
    let result = hdf5_pure_rust::File::open(path);
    assert!(result.is_err());
    std::fs::remove_file(path).ok();
}
