use hdf5_pure_rust::format::fractal_heap::FractalHeapHeader;
use hdf5_pure_rust::format::global_heap::GlobalHeapCollection;
use hdf5_pure_rust::format::local_heap::LocalHeap;
use hdf5_pure_rust::format::messages::attribute::AttributeMessage;
use hdf5_pure_rust::format::messages::attribute_info::AttributeInfoMessage;
use hdf5_pure_rust::format::messages::data_layout::DataLayoutMessage;
use hdf5_pure_rust::format::messages::dataspace::DataspaceMessage;
use hdf5_pure_rust::format::messages::datatype::{ByteOrder, DatatypeClass, DatatypeMessage};
use hdf5_pure_rust::format::messages::fill_value::FillValueMessage;
use hdf5_pure_rust::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SZIP,
};
use hdf5_pure_rust::format::messages::link::LinkMessage;
use hdf5_pure_rust::format::messages::link_info::LinkInfoMessage;
use hdf5_pure_rust::format::object_header::{
    ObjectHeader, MSG_DATATYPE, MSG_HEADER_CONTINUATION, MSG_SHARED_MSG_TABLE,
};
use hdf5_pure_rust::format::superblock::Superblock;
use hdf5_pure_rust::format::{extensible_array, fixed_array};
use hdf5_pure_rust::io::reader::UNDEF_ADDR;
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
fn test_object_header_rejects_truncated_v1_message_containers() {
    fn v1_header(chunk: &[u8]) -> Vec<u8> {
        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(chunk);
        data
    }

    let mut payload_exceeds_chunk = Vec::new();
    payload_exceeds_chunk.extend_from_slice(&MSG_DATATYPE.to_le_bytes());
    payload_exceeds_chunk.extend_from_slice(&8u16.to_le_bytes());
    payload_exceeds_chunk.push(0);
    payload_exceeds_chunk.extend_from_slice(&[0; 3]);

    let mut truncated_continuation = Vec::new();
    truncated_continuation.extend_from_slice(&MSG_HEADER_CONTINUATION.to_le_bytes());
    truncated_continuation.extend_from_slice(&4u16.to_le_bytes());
    truncated_continuation.push(0);
    truncated_continuation.extend_from_slice(&[0; 3]);

    for data in [
        v1_header(&[1; 7]),
        v1_header(&payload_exceeds_chunk),
        v1_header(&truncated_continuation),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("truncated object-header message container should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_object_header_rejects_invalid_v1_continuation_ranges() {
    fn v1_header(chunk: &[u8]) -> Vec<u8> {
        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(chunk);
        data
    }

    fn continuation_message(addr: u64, length: u64) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&MSG_HEADER_CONTINUATION.to_le_bytes());
        chunk.extend_from_slice(&16u16.to_le_bytes());
        chunk.push(0);
        chunk.extend_from_slice(&[0; 3]);
        chunk.extend_from_slice(&addr.to_le_bytes());
        chunk.extend_from_slice(&length.to_le_bytes());
        chunk
    }

    for data in [
        v1_header(&continuation_message(u64::MAX - 4, 16)),
        v1_header(&continuation_message(64, 16)),
        v1_header(&continuation_message(16, 16)),
        v1_header(&continuation_message(64, 0)),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("invalid object-header continuation range should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_object_header_rejects_malformed_shared_message_payloads() {
    fn v1_header(msg_type: u16, flags: u8, payload: &[u8]) -> Vec<u8> {
        let aligned_size = (payload.len() + 7) & !7;
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&msg_type.to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        chunk.push(flags);
        chunk.extend_from_slice(&[0; 3]);
        chunk.extend_from_slice(payload);
        chunk.resize(8 + aligned_size, 0);

        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&chunk);
        data
    }

    let shared_flag = 0x02;
    let mut valid_table = vec![0];
    valid_table.extend_from_slice(&64u64.to_le_bytes());
    valid_table.push(1);
    let mut shared_v3_sohm = vec![3, 1];
    shared_v3_sohm.extend_from_slice(&[0x5a; 8]);

    for data in [
        v1_header(MSG_SHARED_MSG_TABLE, 0, &[]),
        v1_header(MSG_SHARED_MSG_TABLE, 0, &[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        v1_header(
            MSG_SHARED_MSG_TABLE,
            0,
            &[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1],
        ),
        v1_header(MSG_SHARED_MSG_TABLE, 0, &[0, 64, 0, 0, 0, 0, 0, 0, 0, 0]),
        v1_header(MSG_DATATYPE, shared_flag, &[4, 2, 64, 0, 0, 0, 0, 0, 0, 0]),
        v1_header(MSG_DATATYPE, shared_flag, &[3, 3, 64, 0, 0, 0, 0, 0, 0, 0]),
        v1_header(MSG_DATATYPE, shared_flag, &[3, 2, 64, 0, 0]),
        v1_header(
            MSG_DATATYPE,
            shared_flag,
            &[2, 2, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("malformed shared-message payload should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }

    for data in [
        v1_header(MSG_SHARED_MSG_TABLE, 0, &valid_table),
        v1_header(MSG_DATATYPE, shared_flag, &shared_v3_sohm),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        ObjectHeader::read_at(&mut reader, 0)
            .expect("well-formed shared-message payload should parse as a raw message");
    }
}

#[test]
fn test_object_header_truncated_synthetic_files_do_not_panic() {
    fn v1_header_with_tail(chunk: &[u8], tail: &[u8]) -> Vec<u8> {
        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(chunk);
        data.extend_from_slice(tail);
        data
    }

    fn v1_message(msg_type: u16, flags: u8, payload: &[u8]) -> Vec<u8> {
        let aligned_size = (payload.len() + 7) & !7;
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&msg_type.to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        chunk.push(flags);
        chunk.extend_from_slice(&[0; 3]);
        chunk.extend_from_slice(payload);
        chunk.resize(8 + aligned_size, 0);
        chunk
    }

    let datatype_payload = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    let normal_message_file =
        v1_header_with_tail(&v1_message(MSG_DATATYPE, 0, &datatype_payload), &[]);

    let mut shared_table_payload = vec![0];
    shared_table_payload.extend_from_slice(&64u64.to_le_bytes());
    shared_table_payload.push(1);
    let shared_table_file = v1_header_with_tail(
        &v1_message(MSG_SHARED_MSG_TABLE, 0, &shared_table_payload),
        &[],
    );

    let mut shared_ref_payload = vec![3, 1];
    shared_ref_payload.extend_from_slice(&[0x5a; 8]);
    let shared_ref_file =
        v1_header_with_tail(&v1_message(MSG_DATATYPE, 0x02, &shared_ref_payload), &[]);

    let continuation_chunk = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&40u64.to_le_bytes());
        payload.extend_from_slice(&8u64.to_le_bytes());
        v1_message(MSG_HEADER_CONTINUATION, 0, &payload)
    };
    let continuation_file = v1_header_with_tail(&continuation_chunk, &[0; 8]);

    for (name, data) in [
        ("normal", normal_message_file),
        ("shared table", shared_table_file),
        ("shared reference", shared_ref_file),
        ("continuation", continuation_file),
    ] {
        for len in 0..=data.len() {
            let prefix = data[..len].to_vec();
            let result = std::panic::catch_unwind(|| {
                let mut reader = HdfReader::new(Cursor::new(prefix));
                let _ = ObjectHeader::read_at(&mut reader, 0);
            });
            assert!(
                result.is_ok(),
                "{name} object-header prefix length {len} panicked"
            );
        }
    }
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
fn test_message_decoders_reject_invalid_versions_and_classes_as_format_errors() {
    let datatype_cases = [
        vec![0x00, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 8],
        vec![0x10, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
        vec![0x1a, 1, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0],
        vec![0x19, 2, 0, 0, 8, 0, 0, 0, 1, 2, 3, 4],
    ];
    for data in datatype_cases {
        let err = DatatypeMessage::decode(&data)
            .expect_err("invalid datatype version/class combination should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }

    for err in [
        DataspaceMessage::decode(&[3, 0, 0, 0]).expect_err("invalid dataspace version"),
        DataLayoutMessage::decode(&[9, 0], 8, 8).expect_err("invalid layout version"),
        LinkMessage::decode(&[2, 0], 8).expect_err("invalid link version"),
        AttributeMessage::decode(&[4, 0, 0, 0, 0, 0]).expect_err("invalid attribute version"),
        FillValueMessage::decode(&[9]).expect_err("invalid fill value version"),
        FilterPipelineMessage::decode(&[9, 0]).expect_err("invalid filter pipeline version"),
        LinkInfoMessage::decode(&[1, 0], 8).expect_err("invalid link info version"),
        AttributeInfoMessage::decode(&[1, 0], 8).expect_err("invalid attribute info version"),
    ] {
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_message_decoders_reject_every_truncated_prefix() {
    fn assert_prefixes_fail<F>(name: &str, data: &[u8], decode: F)
    where
        F: Fn(&[u8]) -> hdf5_pure_rust::Result<()>,
    {
        decode(data).unwrap_or_else(|err| panic!("{name} full payload should decode: {err}"));
        for len in 0..data.len() {
            let prefix = &data[..len];
            assert!(
                decode(prefix).is_err(),
                "{name} prefix length {len} decoded unexpectedly"
            );
        }
    }

    let datatype = vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    assert_prefixes_fail("datatype", &datatype, |data| {
        DatatypeMessage::decode(data).map(|_| ())
    });

    let mut dataspace = vec![2, 1, 0, 1];
    dataspace.extend_from_slice(&3u64.to_le_bytes());
    assert_prefixes_fail("dataspace", &dataspace, |data| {
        DataspaceMessage::decode(data).map(|_| ())
    });

    let mut layout = vec![3, 1];
    layout.extend_from_slice(&64u64.to_le_bytes());
    layout.extend_from_slice(&12u64.to_le_bytes());
    assert_prefixes_fail("data layout", &layout, |data| {
        DataLayoutMessage::decode(data, 8, 8).map(|_| ())
    });

    let mut link = vec![1, 0, 1, b'x'];
    link.extend_from_slice(&64u64.to_le_bytes());
    assert_prefixes_fail("link", &link, |data| {
        LinkMessage::decode(data, 8).map(|_| ())
    });

    let mut attribute = vec![3, 0];
    attribute.extend_from_slice(&2u16.to_le_bytes());
    attribute.extend_from_slice(&(datatype.len() as u16).to_le_bytes());
    attribute.extend_from_slice(&4u16.to_le_bytes());
    attribute.push(0);
    attribute.extend_from_slice(b"a\0");
    attribute.extend_from_slice(&datatype);
    attribute.extend_from_slice(&[2, 0, 0, 0]);
    assert_prefixes_fail("attribute", &attribute, |data| {
        AttributeMessage::decode(data).map(|_| ())
    });

    let fill_value = vec![3, 0x20, 4, 0, 0, 0, 1, 2, 3, 4];
    assert_prefixes_fail("fill value", &fill_value, |data| {
        FillValueMessage::decode(data).map(|_| ())
    });

    let mut filter_pipeline = vec![2, 1];
    filter_pipeline.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
    filter_pipeline.extend_from_slice(&0u16.to_le_bytes());
    filter_pipeline.extend_from_slice(&1u16.to_le_bytes());
    filter_pipeline.extend_from_slice(&6u32.to_le_bytes());
    assert_prefixes_fail("filter pipeline", &filter_pipeline, |data| {
        FilterPipelineMessage::decode(data).map(|_| ())
    });
}

#[test]
fn test_message_size_arithmetic_overflow_returns_format_error() {
    let mut overflowing_layout = vec![1, 3, 1, 0, 0, 0, 0, 0];
    overflowing_layout.extend_from_slice(&64u64.to_le_bytes());
    overflowing_layout.extend_from_slice(&u32::MAX.to_le_bytes());
    overflowing_layout.extend_from_slice(&u32::MAX.to_le_bytes());
    overflowing_layout.extend_from_slice(&u32::MAX.to_le_bytes());
    let err = DataLayoutMessage::decode(&overflowing_layout, 8, 8)
        .expect_err("overflowing contiguous v1 layout should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let attribute = AttributeMessage {
        version: 3,
        name: "overflow".to_string(),
        datatype: DatatypeMessage {
            version: 1,
            class: DatatypeClass::FixedPoint,
            class_bits: [0; 3],
            size: 8,
            properties: vec![0, 0, 64, 0],
        },
        dataspace: DataspaceMessage {
            version: 2,
            space_type: hdf5_pure_rust::format::messages::dataspace::DataspaceType::Simple,
            ndims: 2,
            dims: vec![u64::MAX, 2],
            max_dims: None,
        },
        data: Vec::new(),
    };
    let err = attribute
        .data_size()
        .expect_err("overflowing attribute data size should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_declared_allocation_counts_are_capped() {
    let err =
        DataspaceMessage::decode(&[2, 33, 0, 1]).expect_err("dataspace rank above cap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let err =
        FilterPipelineMessage::decode(&[2, 33]).expect_err("filter count above cap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let err = DataLayoutMessage::decode(&[1, 33, 1, 0, 0, 0, 0, 0], 8, 8)
        .expect_err("layout rank above cap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut local_heap = b"HEAP".to_vec();
    local_heap.push(0);
    local_heap.extend_from_slice(&[0; 3]);
    local_heap.extend_from_slice(&(4u64 * 1024 * 1024 * 1024 + 1).to_le_bytes());
    local_heap.extend_from_slice(&0u64.to_le_bytes());
    local_heap.extend_from_slice(&32u64.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(local_heap));
    let err = LocalHeap::read_at(&mut reader, 0).expect_err("oversized local heap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let heap = LocalHeap {
        data: b"valid\0unterminated".to_vec(),
    };
    assert_eq!(heap.get_string(0).unwrap(), "valid");
    let err = heap
        .get_string(6)
        .expect_err("unterminated local heap string should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    let err = heap
        .get_string(heap.data.len())
        .expect_err("out-of-bounds local heap string should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut global_heap = b"GCOL".to_vec();
    global_heap.push(1);
    global_heap.extend_from_slice(&[0; 3]);
    global_heap.extend_from_slice(&64u64.to_le_bytes());
    global_heap.extend_from_slice(&1u16.to_le_bytes());
    global_heap.extend_from_slice(&1u16.to_le_bytes());
    global_heap.extend_from_slice(&[0; 4]);
    global_heap.extend_from_slice(&(4u64 * 1024 * 1024 * 1024 + 1).to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(global_heap));
    let err = GlobalHeapCollection::read_at(&mut reader, 0)
        .expect_err("oversized global heap object should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut fixed = b"FAHD".to_vec();
    fixed.push(0);
    fixed.push(0);
    fixed.push(8);
    fixed.push(0);
    fixed.extend_from_slice(&1_000_001u64.to_le_bytes());
    fixed.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    fixed.extend_from_slice(&0u32.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(fixed));
    let err = fixed_array::read_fixed_array_chunks(&mut reader, 0, false, 0)
        .expect_err("oversized fixed array should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut extensible = b"EAHD".to_vec();
    extensible.extend_from_slice(&[0, 0, 8, 1, 1, 1, 1, 1]);
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&1_000_001u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    extensible.extend_from_slice(&0u32.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(extensible));
    let err = extensible_array::read_extensible_array_chunks(&mut reader, 0, false, 0)
        .expect_err("oversized extensible array should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
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
fn test_compound_fields_reject_overlapping_members() {
    let mut data = Vec::new();
    data.push(0x36); // version 3, compound class
    data.extend_from_slice(&[2, 0, 0]); // two members
    data.extend_from_slice(&4u32.to_le_bytes()); // record size

    data.extend_from_slice(b"lo\0");
    data.extend_from_slice(&0u32.to_le_bytes()); // member offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());

    data.extend_from_slice(b"word\0");
    data.extend_from_slice(&0u32.to_le_bytes()); // overlapping offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&32u16.to_le_bytes());

    let dtype = DatatypeMessage::decode(&data).unwrap();
    let err = dtype
        .compound_fields()
        .expect_err("overlapping compound members should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_compound_fields_reject_truncated_member_metadata() {
    let cases = [
        vec![0x36, 1, 0, 0, 4, 0, 0, 0],
        vec![0x36, 1, 0, 0, 4, 0, 0, 0, b'x'],
        vec![0x36, 1, 0, 0, 4, 0, 0, 0, b'x', 0, 1, 2],
        {
            let mut data = vec![0x36, 2, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 32]);
            data
        },
        {
            let mut data = vec![0x16, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[1, 2, 3]);
            data
        },
        {
            let mut data = vec![0x26, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[1, 2, 3, 4]);
            data
        },
        {
            let mut data = vec![0x46, 1, 0, 0, 0, 1, 0, 0];
            data.extend_from_slice(b"x\0");
            data.push(0);
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
    let enum_header_n =
        |version: u8, nmembers: u8| vec![(version << 4) | 8, nmembers, 0, 0, 1, 0, 0, 0];
    let base_u8 = [0x10, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 8];
    let base_u16 = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 16, 0, 16];
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
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u16);
            data.extend_from_slice(b"A\0");
            data.push(1);
            data
        },
        {
            let mut data = enum_header_n(3, 2);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A\0B\0");
            data.push(1);
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
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 33],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 255],
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
fn test_array_dims_base_handles_v2_v3_v4_and_rejects_v1() {
    let base_i16 = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 16];

    let mut v1 = vec![0x1a, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0];
    v1.extend_from_slice(&2u32.to_le_bytes());
    v1.extend_from_slice(&3u32.to_le_bytes());
    v1.extend_from_slice(&base_i16);
    assert!(DatatypeMessage::decode(&v1).is_err());

    for version in [2_u8, 3] {
        let mut data = vec![(version << 4) | 0x0a, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0];
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&base_i16);
        let dtype = DatatypeMessage::decode(&data).unwrap();
        let (dims, base) = dtype.array_dims_base().unwrap();
        assert_eq!(dims, vec![2, 3]);
        assert_eq!(base.size, 2);
    }

    let mut v4 = vec![0x4a, 0, 0, 0, 12, 0, 0, 0, 2];
    v4.extend_from_slice(&2u32.to_le_bytes());
    v4.extend_from_slice(&3u32.to_le_bytes());
    v4.extend_from_slice(&base_i16);
    let dtype = DatatypeMessage::decode(&v4).unwrap();
    let (dims, base) = dtype.array_dims_base().unwrap();
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size, 2);
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
fn test_branchable_error_messages_are_stable() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: 65535,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &pipeline, 4)
        .expect_err("unknown required filter should fail");
    match err {
        hdf5_pure_rust::Error::Unsupported(message) => {
            assert!(message.contains("filter 65535 not implemented"));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }

    let err = hdf5_pure_rust::filters::apply_pipeline_reverse_with_mask(
        &[1, 2, 3, 4],
        &FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: Vec::new(),
            }],
        },
        4,
        0b10,
    )
    .expect_err("invalid filter mask should fail");
    match err {
        hdf5_pure_rust::Error::InvalidFormat(message) => {
            assert!(message.contains("references filters outside pipeline length"));
        }
        other => panic!("expected InvalidFormat, got {other:?}"),
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

#[test]
fn test_filtered_fractal_heap_direct_object_read_fletcher32() {
    let heap = test_fractal_heap(8);
    let payload = b"fletcher filtered heap object".to_vec();
    let mut filtered = payload.clone();
    filtered.extend_from_slice(&hdf5_fletcher32(&payload).to_le_bytes());
    let mut file_bytes = vec![0u8; 64];
    file_bytes.extend_from_slice(&filtered);

    let mut heap = heap;
    heap.root_block_addr = 64;
    heap.root_direct_filtered_size = Some(filtered.len() as u64);
    heap.filter_pipeline = Some(FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_FLETCHER32,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    });

    let mut id = vec![0x00];
    id.extend_from_slice(&0u32.to_le_bytes());
    id.extend_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

fn hdf5_fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let mut pos = 0;
    let mut remaining = data.len() / 2;

    while remaining > 0 {
        let tlen = remaining.min(360);
        remaining -= tlen;

        for _ in 0..tlen {
            let value = ((data[pos] as u32) << 8) | data[pos + 1] as u32;
            sum1 += value;
            sum2 += sum1;
            pos += 2;
        }

        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if data.len() % 2 != 0 {
        sum1 += (data[pos] as u32) << 8;
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

fn test_fractal_heap(io_filter_len: u16) -> FractalHeapHeader {
    FractalHeapHeader {
        heap_addr: 0,
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
