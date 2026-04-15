use hdf5_pure_rust::format::fractal_heap::FractalHeapHeader;
use hdf5_pure_rust::format::messages::attribute::AttributeMessage;
use hdf5_pure_rust::format::messages::data_layout::DataLayoutMessage;
use hdf5_pure_rust::format::messages::dataspace::DataspaceMessage;
use hdf5_pure_rust::format::messages::datatype::{ByteOrder, DatatypeMessage};
use hdf5_pure_rust::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SZIP,
};
use hdf5_pure_rust::format::messages::link::LinkMessage;
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
fn test_compound_field_preserves_member_byte_order() {
    let mut data = Vec::new();
    data.push(0x36); // version 3, compound class
    data.extend_from_slice(&[1, 0, 0]); // one member
    data.extend_from_slice(&4u32.to_le_bytes()); // record size
    data.extend_from_slice(b"x\0");
    data.extend_from_slice(&0u32.to_le_bytes()); // member offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[1, 0, 0]); // big-endian member
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    data.extend_from_slice(&32u16.to_le_bytes()); // bit precision

    let dtype = DatatypeMessage::decode(&data).unwrap();
    let fields = dtype.compound_fields().unwrap();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[0].byte_order, Some(ByteOrder::BigEndian));
    assert_eq!(fields[0].datatype.size, 4);
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
fn test_unsupported_filters_fail_explicitly() {
    for id in [FILTER_SZIP, 65535] {
        let pipeline = FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id,
                name: None,
                flags: 0,
                client_data: Vec::new(),
            }],
        };
        let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &pipeline, 4)
            .expect_err("unsupported filter should return an error");
        assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    }
}

#[test]
fn test_datatype_aware_filters_reject_missing_parameters() {
    for id in [FILTER_NBIT, FILTER_SCALEOFFSET] {
        let pipeline = FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id,
                name: None,
                flags: 0,
                client_data: Vec::new(),
            }],
        };
        let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &pipeline, 4)
            .expect_err("datatype-aware filter should reject missing parameters");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_virtual_layout_parses_as_metadata_only() {
    let mut data = Vec::new();
    data.push(4); // layout message version
    data.push(3); // virtual layout class
    data.extend_from_slice(&0x1234u64.to_le_bytes());
    data.extend_from_slice(&7u32.to_le_bytes());

    let layout = DataLayoutMessage::decode(&data, 8, 8).unwrap();
    assert_eq!(
        layout.layout_class,
        hdf5_pure_rust::format::messages::data_layout::LayoutClass::Virtual
    );
    assert_eq!(layout.virtual_heap_addr, Some(0x1234));
    assert_eq!(layout.virtual_heap_index, Some(7));
}

#[test]
fn test_huge_fractal_heap_object_is_unsupported() {
    let heap = test_fractal_heap(0);
    let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
    let err = heap
        .read_managed_object(&mut reader, &[0x10])
        .expect_err("huge fractal-heap IDs should be unsupported");
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
}

#[test]
fn test_filtered_fractal_heap_object_is_unsupported() {
    let heap = test_fractal_heap(8);
    let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
    let err = heap
        .read_managed_object(&mut reader, &[0x00])
        .expect_err("filtered fractal heaps should be unsupported");
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
}

fn test_fractal_heap(io_filter_len: u16) -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_len: 8,
        io_filter_len,
        flags: 0,
        max_managed_obj_size: 1024,
        table_width: 4,
        start_block_size: 512,
        max_direct_block_size: 4096,
        max_heap_size: 32,
        start_root_rows: 1,
        root_block_addr: 0,
        current_root_rows: 0,
        num_managed_objects: 0,
        has_checksum: false,
    }
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not_hdf5.bin");
    std::fs::write(&path, b"This is not an HDF5 file").unwrap();
    let result = hdf5_pure_rust::File::open(&path);
    assert!(result.is_err());
}
