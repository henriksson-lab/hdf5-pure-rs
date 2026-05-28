use hdf5_pure_rust::engine::writer::{CompoundFieldSpec, DtypeSpec};
use hdf5_pure_rust::format::messages::datatype::DatatypeClass;
use hdf5_pure_rust::DeriveH5Type;
use hdf5_pure_rust::H5Type;
use hdf5_pure_rust::{File, WritableFile};

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Point {
    x: f64,
    y: f64,
    label: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, DeriveH5Type)]
#[repr(u8)]
#[allow(dead_code)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, DeriveH5Type)]
#[repr(u64)]
#[allow(dead_code)]
enum LargeCode {
    Max = u64::MAX,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, DeriveH5Type)]
#[repr(i8)]
#[allow(dead_code)]
enum SignedCode {
    Negative = -1,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, DeriveH5Type)]
#[repr(transparent)]
struct SampleId(u32);

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Pixel {
    color: Color,
    intensity: u8,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Measurement {
    value: f32,
    #[hdf5(rename = "error_margin")]
    error: f32,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct InnerPair {
    a: u8,
    b: u8,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct NestedRecord {
    inner: InnerPair,
    tag: u8,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct NestedPaddedRecord {
    tag: u8,
    point: Point,
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct ManualUnknown(u8);

unsafe impl H5Type for ManualUnknown {
    fn type_size() -> usize {
        1
    }
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
#[allow(dead_code)]
struct HasUnknownField {
    value: ManualUnknown,
}

#[test]
fn test_derive_struct_size() {
    assert_eq!(Point::type_size(), std::mem::size_of::<Point>());
    // f64(8) + f64(8) + i32(4) + padding(4) = 24 on most platforms, or 20 with packed
    assert!(Point::type_size() >= 20);
    assert!(Point::has_padding());
    assert!(!Measurement::has_padding());
}

#[test]
fn test_derive_struct_fields() {
    let mut index = 0;
    Point::visit_compound_fields(|field| {
        match index {
            0 => {
                assert_eq!(field.name, "x");
                assert_eq!(field.offset, 0);
                assert_eq!(field.size, 8);
            }
            1 => {
                assert_eq!(field.name, "y");
                assert_eq!(field.offset, 8);
                assert_eq!(field.size, 8);
            }
            2 => {
                assert_eq!(field.name, "label");
                assert_eq!(field.offset, 16);
                assert_eq!(field.size, 4);
            }
            _ => panic!("unexpected extra field: {field:?}"),
        }
        index += 1;
    })
    .unwrap();
    assert_eq!(index, 3);
}

#[test]
fn test_derive_enum_size() {
    assert_eq!(Color::type_size(), 1);
    assert!(Color::requires_validation());
    assert_eq!(Color::enum_base_type(), Some((false, 1)));
}

#[test]
fn test_derive_enum_members() {
    let expected = [("Red", 0), ("Green", 1), ("Blue", 2)];
    let mut index = 0;
    Color::visit_enum_members(|name, value| {
        assert_eq!((name, value), expected[index]);
        index += 1;
    })
    .unwrap();
    assert_eq!(index, expected.len());
}

#[test]
fn test_derive_enum_members_preserve_raw_unsigned_values() {
    assert_eq!(LargeCode::enum_base_type(), Some((false, 8)));
    assert_eq!(
        LargeCode::enum_members().unwrap(),
        vec![("Max".to_string(), u64::MAX)]
    );
}

#[test]
fn test_derive_signed_enum_members_are_raw_encoded_values() {
    assert_eq!(SignedCode::enum_base_type(), Some((true, 1)));
    assert_eq!(
        SignedCode::enum_members().unwrap(),
        vec![("Negative".to_string(), u8::MAX as u64)]
    );
}

#[test]
fn test_derive_enum_rejects_invalid_discriminant_bytes() {
    let err = match hdf5_pure_rust::hl::types::bytes_to_vec::<Color>(vec![3]) {
        Ok(_) => panic!("invalid enum discriminant should be rejected"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("enum value"));
}

#[test]
fn test_derive_compound_validates_enum_fields() {
    assert!(Pixel::requires_validation());
    assert!(!Pixel::has_padding());

    let err = match hdf5_pure_rust::hl::types::bytes_to_vec::<Pixel>(vec![3, 9]) {
        Ok(_) => panic!("invalid nested enum discriminant should be rejected"),
        Err(err) => err,
    };
    assert!(format!("{err}").contains("enum value"));
}

#[test]
fn test_derive_padded_struct_rejects_typed_write_byte_view() {
    let value = Point {
        x: 1.0,
        y: 2.0,
        label: 42,
    };
    let err = hdf5_pure_rust::hl::types::slice_as_bytes_checked(std::slice::from_ref(&value))
        .unwrap_err();
    assert!(format!("{err}").contains("padding"));
}

#[test]
fn test_derive_padded_struct_typed_write_roundtrips_without_padding_exposure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("derive_padded_write.h5");
    let expected = Point {
        x: 1.0,
        y: 2.0,
        label: 42,
    };
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("points")
            .write(&[expected])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let values = file.dataset("points").unwrap().read::<Point>().unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].x, expected.x);
    assert_eq!(values[0].y, expected.y);
    assert_eq!(values[0].label, expected.label);
}

#[test]
fn test_derive_nested_padded_compound_write_zeros_all_padding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("derive_nested_padded_write.h5");
    let expected = NestedPaddedRecord {
        tag: 7,
        point: Point {
            x: 1.0,
            y: 2.0,
            label: 42,
        },
    };
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("records")
            .write(&[expected])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let dataset = file.dataset("records").unwrap();
    let values = dataset.read::<NestedPaddedRecord>().unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].tag, expected.tag);
    assert_eq!(values[0].point.x, expected.point.x);
    assert_eq!(values[0].point.y, expected.point.y);
    assert_eq!(values[0].point.label, expected.point.label);

    let raw = dataset.read_raw().unwrap();
    assert_eq!(raw.len(), NestedPaddedRecord::type_size());
    let point_offset = std::mem::offset_of!(NestedPaddedRecord, point);
    assert!(raw[1..point_offset].iter().all(|byte| *byte == 0));
    let point_tail_start = point_offset + 20;
    let point_tail_end = point_offset + Point::type_size();
    assert!(raw[point_tail_start..point_tail_end]
        .iter()
        .all(|byte| *byte == 0));
}

#[test]
fn test_derive_enum_typed_write_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("derive_enum_write.h5");
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("colors")
            .write(&[Color::Red, Color::Blue])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let values = file.dataset("colors").unwrap().read::<Color>().unwrap();
    assert_eq!(values, vec![Color::Red, Color::Blue]);
}

#[test]
fn test_transparent_primitive_wrapper_typed_write_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("transparent_newtype_write.h5");
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("ids")
            .write(&[SampleId(7), SampleId(11)])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let values = file.dataset("ids").unwrap().read::<SampleId>().unwrap();
    assert_eq!(values, vec![SampleId(7), SampleId(11)]);
}

#[test]
fn test_derive_compound_enum_field_writes_enum_datatype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compound_enum_dtype.h5");
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("pixels")
            .write(&[Pixel {
                color: Color::Green,
                intensity: 9,
            }])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let info = file.dataset("pixels").unwrap().info().unwrap();
    let fields = info.datatype.compound_fields_iter().unwrap();
    let field = fields.into_iter().next().unwrap().unwrap();
    assert_eq!(field.name, "color");
    assert_eq!(field.class, DatatypeClass::Enum);
    let members = field
        .datatype
        .enum_members_iter()
        .unwrap()
        .map(|member| {
            let member = member.unwrap();
            (member.name.to_string(), member.value)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        members,
        vec![
            ("Red".to_string(), 0),
            ("Green".to_string(), 1),
            ("Blue".to_string(), 2),
        ]
    );
}

#[test]
fn test_derive_nested_compound_field_writes_compound_datatype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested_compound_dtype.h5");
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("records")
            .write(&[NestedRecord {
                inner: InnerPair { a: 1, b: 2 },
                tag: 3,
            }])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let info = file.dataset("records").unwrap().info().unwrap();
    let fields = info.datatype.compound_fields_iter().unwrap();
    let field = fields.into_iter().next().unwrap().unwrap();
    assert_eq!(field.name, "inner");
    assert_eq!(field.class, DatatypeClass::Compound);
    let nested_fields = field.datatype.compound_fields_iter().unwrap();
    let names = nested_fields
        .map(|field| field.unwrap().name.to_string())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn test_derive_unknown_field_metadata_is_not_silently_unsigned() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unknown_field_dtype.h5");
    let mut writer = WritableFile::create(&path).unwrap();
    let err = writer
        .new_dataset_builder("bad")
        .write(&[HasUnknownField {
            value: ManualUnknown(7),
        }])
        .unwrap_err();
    assert!(format!("{err}").contains("unsupported type"));
}

#[test]
fn test_derive_enum_invalid_direct_slice_read_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid_enum_slice.h5");
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("codes")
            .write_raw_with_dtype(DtypeSpec::U8, &[0, 3, 1])
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let err = file
        .dataset("codes")
        .unwrap()
        .read_slice::<Color, _>(0..3)
        .unwrap_err();
    assert!(format!("{err}").contains("enum value"));
}

#[test]
fn test_derive_enum_invalid_compound_field_read_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid_enum_field.h5");
    {
        let mut writer = WritableFile::create(&path).unwrap();
        writer
            .new_dataset_builder("pixels")
            .write_raw_with_dtype(
                DtypeSpec::Compound {
                    size: 2,
                    fields: vec![
                        CompoundFieldSpec {
                            name: "color".to_string(),
                            offset: 0,
                            dtype: DtypeSpec::U8,
                        },
                        CompoundFieldSpec {
                            name: "intensity".to_string(),
                            offset: 1,
                            dtype: DtypeSpec::U8,
                        },
                    ],
                },
                &[3, 9],
            )
            .unwrap();
        writer.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let err = file
        .dataset("pixels")
        .unwrap()
        .read_field::<Color>("color")
        .unwrap_err();
    assert!(format!("{err}").contains("enum value"));
}

#[test]
fn test_padded_fill_value_writes_without_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("padded_fill_value.h5");
    let mut writer = WritableFile::create(&path).unwrap();
    let point = Point {
        x: 1.0,
        y: 2.0,
        label: 42,
    };

    writer
        .new_dataset_builder("points")
        .chunk(&[1])
        .shape(&[1])
        .fill_value(point)
        .write_fill::<Point>()
        .unwrap();
}

#[test]
fn test_derive_with_rename() {
    let expected = ["value", "error_margin"];
    let mut index = 0;
    Measurement::visit_compound_fields(|field| {
        assert_eq!(field.name, expected[index]);
        index += 1;
    })
    .unwrap();
    assert_eq!(index, expected.len());
}

#[test]
fn test_derive_struct_can_read() {
    // Verify the derived type works with read operations
    // (uses type_size for byte reinterpretation)
    let mut bytes = [0u8; 24];
    bytes[0..8].copy_from_slice(&1.0f64.to_le_bytes());
    bytes[8..16].copy_from_slice(&2.0f64.to_le_bytes());
    bytes[16..20].copy_from_slice(&42i32.to_le_bytes());

    let points: Vec<Point> =
        hdf5_pure_rust::hl::types::bytes_to_vec::<Point>(bytes.to_vec()).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].x, 1.0);
    assert_eq!(points[0].y, 2.0);
    assert_eq!(points[0].label, 42);
}
