use hdf5_pure_rust::format::fractal_heap::FractalHeapHeader;
use hdf5_pure_rust::format::messages::attribute::AttributeMessage;
use hdf5_pure_rust::format::messages::attribute_info::AttributeInfoMessage;
use hdf5_pure_rust::format::messages::data_layout::DataLayoutMessage;
use hdf5_pure_rust::format::messages::dataspace::DataspaceMessage;
use hdf5_pure_rust::format::messages::datatype::{ByteOrder, DatatypeMessage};
use hdf5_pure_rust::format::messages::fill_value::FillValueMessage;
use hdf5_pure_rust::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_DEFLATE, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SZIP,
};
use hdf5_pure_rust::format::messages::link::LinkMessage;
use hdf5_pure_rust::format::messages::link_info::LinkInfoMessage;
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
fn test_link_message_rejects_truncated_optional_fields() {
    for data in [
        vec![1],
        vec![1, 0x08],
        vec![1, 0x04, 0, 0, 0],
        vec![1, 0x10],
        vec![1, 0x03, 1, 2],
        vec![1, 0x08, 1, 1, b's', 1],
        vec![1, 0x08, 64, 1, b'e', 3, 0, 0],
    ] {
        let err = LinkMessage::decode(&data, 8).expect_err("truncated link message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_link_message_bad_name_length() {
    // Version 1, flags=0 (hard link, 1-byte name len), name_len=255 but only 4 bytes of data
    let data = vec![1, 0x00, 0xFF, 0x41]; // version=1, flags=0, name_len=255, 'A'
    assert!(LinkMessage::decode(&data, 8).is_err());
}

#[test]
fn test_info_messages_reject_truncated_addresses() {
    for data in [
        vec![0, 0, 1, 2, 3],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        vec![0, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    ] {
        let err =
            LinkInfoMessage::decode(&data, 8).expect_err("truncated link info message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }

    for data in [
        vec![0, 0, 1, 2, 3],
        vec![0, 0x01, 1],
        vec![0, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    ] {
        let err = AttributeInfoMessage::decode(&data, 8)
            .expect_err("truncated attribute info message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_dataspace_empty() {
    assert!(DataspaceMessage::decode(&[]).is_err());
}

#[test]
fn test_dataspace_truncated() {
    let data = vec![2, 3, 0, 1]; // version 2, ndims=3, flags=0, type=simple -- missing dim data
    assert!(DataspaceMessage::decode(&data).is_err());
}

#[test]
fn test_dataspace_rejects_truncated_declared_dims() {
    for data in [
        vec![1, 1, 0, 0],
        vec![1, 1, 0, 0, 0, 0, 0, 0],
        vec![2, 1, 0, 1, 1, 2, 3],
        vec![2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ] {
        let err = DataspaceMessage::decode(&data)
            .expect_err("truncated dataspace dimensions should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
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
fn test_datatype_rejects_truncated_fixed_size_properties() {
    for data in [
        vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        vec![0x11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0x14, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    ] {
        let err =
            DatatypeMessage::decode(&data).expect_err("truncated datatype properties should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
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
fn test_compound_fields_reject_truncated_member_metadata() {
    let cases = [
        vec![0x36, 1, 0, 0, 4, 0, 0, 0],
        vec![0x36, 1, 0, 0, 4, 0, 0, 0, b'x'],
        vec![0x36, 1, 0, 0, 4, 0, 0, 0, b'x', 0, 1, 2],
        {
            let mut data = vec![0x16, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[1, 2, 3]);
            data
        },
        {
            let mut data = vec![0x36, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[0x10, 0, 0]);
            data
        },
        {
            let mut data = vec![0x36, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[0x10, 0, 0, 0, 4, 0, 0, 0, 1]);
            data
        },
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("compound header should decode");
        let err = dtype
            .compound_fields()
            .expect_err("truncated compound member metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_enum_members_reject_truncated_metadata() {
    let enum_header = |version: u8| vec![(version << 4) | 8, 1, 0, 0, 1, 0, 0, 0];
    let base_u8 = [0x10, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 8];
    let cases = [
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&[0x10, 0, 0]);
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&[0x10, 0, 0, 0, 1, 0, 0, 0, 0]);
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u8);
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A");
            data
        },
        {
            let mut data = enum_header(1);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A\0");
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A\0");
            data
        },
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("enum header should decode");
        let err = dtype
            .enum_members()
            .expect_err("truncated enum member metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_array_dims_base_rejects_truncated_metadata() {
    let base_i32 = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 32];
    let cases = [
        vec![0x3a, 0, 0, 0, 4, 0, 0, 0],
        vec![0x3a, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 2, 4, 0, 0, 0],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0],
        {
            let mut data = vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0];
            data.extend_from_slice(&[0x10, 0, 0]);
            data
        },
        {
            let mut data = vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0];
            data.extend_from_slice(&base_i32[..9]);
            data
        },
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("array header should decode");
        let err = dtype
            .array_dims_base()
            .expect_err("truncated array datatype metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_vlen_base_distinguishes_sequence_and_string_metadata() {
    let base_i32 = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 32];

    let mut direct_sequence = vec![0x39, 0, 0, 0, 16, 0, 0, 0];
    direct_sequence.extend_from_slice(&base_i32);
    let dtype = DatatypeMessage::decode(&direct_sequence).expect("vlen header should decode");
    assert_eq!(dtype.vlen_base().unwrap().unwrap().size, 4);

    let mut string_metadata = vec![0x39, 0, 0, 0, 16, 0, 0, 0];
    string_metadata.extend_from_slice(&[1, 0, 0, 0]);
    let dtype =
        DatatypeMessage::decode(&string_metadata).expect("vlen string header should decode");
    assert!(dtype.vlen_base().unwrap().is_none());

    let mut metadata_sequence = vec![0x39, 0, 0, 0, 16, 0, 0, 0];
    metadata_sequence.extend_from_slice(&[0, 0, 0, 0]);
    metadata_sequence.extend_from_slice(&base_i32);
    let dtype =
        DatatypeMessage::decode(&metadata_sequence).expect("vlen sequence header should decode");
    assert_eq!(dtype.vlen_base().unwrap().unwrap().size, 4);
}

#[test]
fn test_vlen_base_rejects_truncated_or_ambiguous_metadata() {
    let cases = [
        vec![0x39, 0, 0, 0, 16, 0, 0, 0],
        vec![0x39, 0, 0, 0, 16, 0, 0, 0, 1, 2],
        vec![0x39, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0],
        vec![
            0x39, 0, 0, 0, 16, 0, 0, 0, 0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 32, 99,
        ],
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("vlen header should decode");
        let err = dtype
            .vlen_base()
            .expect_err("truncated vlen metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
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
fn test_layout_rejects_truncated_payloads() {
    for data in [
        vec![1, 1, 1, 0, 0, 0, 0, 0],
        vec![3, 0, 4, 0, 1],
        vec![3, 1, 0, 0],
        vec![4, 2, 0, 1, 8],
        vec![4, 3],
    ] {
        let err = DataLayoutMessage::decode(&data, 8, 8)
            .expect_err("truncated data layout message should fail without panicking");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_filter_pipeline_empty() {
    assert!(FilterPipelineMessage::decode(&[]).is_err());
}

#[test]
fn test_filter_pipeline_rejects_truncated_decode_payloads() {
    for data in [
        vec![1, 1, 0, 0, 0, 0, 0, 0],
        vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0],
        vec![2, 1],
        vec![2, 1, 0, 1],
        vec![2, 1, 1, 0, 0, 0, 1, 0],
    ] {
        let err = FilterPipelineMessage::decode(&data)
            .expect_err("truncated filter pipeline message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_fill_value_rejects_truncated_defined_values() {
    assert!(FillValueMessage::decode(&[2, 0, 0, 1]).is_err());
    assert!(FillValueMessage::decode(&[3, 0x20, 4, 0, 0, 0, 1, 2]).is_err());
    assert!(FillValueMessage::decode_old(&[4, 0, 0, 0, 1, 2]).is_err());
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
fn test_filter_pipeline_rejects_out_of_range_filter_mask() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse_with_mask(
        &[1, 2, 3, 4],
        &pipeline,
        4,
        0b10,
    )
    .expect_err("out-of-range filter mask should return an error");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_filter_pipeline_rejects_more_than_32_filters() {
    let filter = FilterDesc {
        id: FILTER_DEFLATE,
        name: None,
        flags: 0,
        client_data: Vec::new(),
    };
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![filter; 33],
    };
    let err =
        hdf5_pure_rust::filters::apply_pipeline_reverse_with_mask(&[1, 2, 3, 4], &pipeline, 4, 0)
            .expect_err("pipeline longer than the 32-bit filter mask should return an error");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
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
fn test_huge_fractal_heap_indirect_object_is_unsupported() {
    let heap = test_fractal_heap(0);
    let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
    let err = heap
        .read_managed_object(&mut reader, &[0x10])
        .expect_err("indirect huge fractal-heap IDs should fail explicitly");
    assert!(matches!(
        err,
        hdf5_pure_rust::Error::Unsupported(_) | hdf5_pure_rust::Error::InvalidFormat(_)
    ));
}

#[test]
fn test_huge_fractal_heap_direct_object_read() {
    let heap = test_fractal_heap(0);
    let payload = b"huge object".to_vec();
    let mut file_bytes = vec![0u8; 32];
    file_bytes.extend_from_slice(&payload);
    let addr = 32u64;
    let len = payload.len() as u64;

    let mut id = vec![0x10];
    id.extend_from_slice(&addr.to_le_bytes());
    id.extend_from_slice(&len.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

#[test]
fn test_filtered_huge_fractal_heap_direct_object_read() {
    let mut heap = test_fractal_heap(8);
    let payload = b"filtered huge heap object".to_vec();
    let filtered = hdf5_pure_rust::filters::deflate::compress(&payload, 6).unwrap();
    let mut file_bytes = vec![0u8; 48];
    file_bytes.extend_from_slice(&filtered);
    let addr = 48u64;

    heap.heap_id_len = 1 + 8 + 8 + 4 + 8;
    heap.filter_pipeline = Some(FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: vec![6],
        }],
    });

    let mut id = vec![0x10];
    id.extend_from_slice(&addr.to_le_bytes());
    id.extend_from_slice(&(filtered.len() as u64).to_le_bytes());
    id.extend_from_slice(&0u32.to_le_bytes());
    id.extend_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

#[test]
fn test_filtered_fractal_heap_direct_object_read() {
    let heap = test_fractal_heap(8);
    let payload = b"filtered heap object".to_vec();
    let filtered = hdf5_pure_rust::filters::deflate::compress(&payload, 6).unwrap();
    let mut file_bytes = vec![0u8; 64];
    file_bytes.extend_from_slice(&filtered);

    let mut heap = heap;
    heap.root_block_addr = 64;
    heap.root_direct_filtered_size = Some(filtered.len() as u64);
    heap.filter_pipeline = Some(FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: vec![6],
        }],
    });

    let mut id = vec![0x00];
    id.extend_from_slice(&0u32.to_le_bytes());
    id.extend_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
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
        sizeof_addr: 8,
        sizeof_size: 8,
        huge_btree_addr: hdf5_pure_rust::io::reader::UNDEF_ADDR,
        root_direct_filtered_size: None,
        root_direct_filter_mask: 0,
        filter_pipeline: None,
    }
}

#[test]
fn test_attribute_empty() {
    assert!(AttributeMessage::decode(&[]).is_err());
}

#[test]
fn test_attribute_rejects_truncated_metadata_sections() {
    for data in [
        vec![1, 0, 1, 0, 8, 0, 4, 0],
        vec![2, 0, 4, 0, 8, 0, 4, 0, b'a'],
        vec![2, 0, 1, 0, 8, 0, 4, 0, b'a', 0x10, 0, 0],
        vec![3, 0, 1, 0, 8, 0, 4, 0],
        vec![3, 0, 1, 0, 8, 0, 4, 0, 0, b'a', 0x10],
    ] {
        let err =
            AttributeMessage::decode(&data).expect_err("truncated attribute message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
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
