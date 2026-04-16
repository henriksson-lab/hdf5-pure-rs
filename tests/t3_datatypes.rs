//! Phase T3: Datatype tests -- read all primitive, compound, enum, string types.

use hdf5_pure_rust::format::messages::datatype::DatatypeClass;
use hdf5_pure_rust::Error;
use hdf5_pure_rust::File;

const FILE: &str = "tests/data/hdf5_ref/all_dtypes.h5";

fn open() -> File {
    File::open(FILE).expect("failed to open all_dtypes.h5")
}

// T3a: Primitive integer types

#[test]
fn t3a_int8() {
    let f = open();
    let vals: Vec<i8> = f.dataset("int8").unwrap().read::<i8>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_int16() {
    let vals: Vec<i16> = open().dataset("int16").unwrap().read::<i16>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_int32() {
    let vals: Vec<i32> = open().dataset("int32").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_int64() {
    let vals: Vec<i64> = open().dataset("int64").unwrap().read::<i64>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint8() {
    let vals: Vec<u8> = open().dataset("uint8").unwrap().read::<u8>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint16() {
    let vals: Vec<u16> = open().dataset("uint16").unwrap().read::<u16>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint32() {
    let vals: Vec<u32> = open().dataset("uint32").unwrap().read::<u32>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint64() {
    let vals: Vec<u64> = open().dataset("uint64").unwrap().read::<u64>().unwrap();
    assert_eq!(vals, vec![0, 1, 2, 3, 4]);
}

#[test]
fn t3a_integer_signed_widening_and_narrowing() {
    let f = File::open("tests/data/hdf5_ref/integer_conversion_vectors.h5").unwrap();
    let ds = f.dataset("i16_conversion").unwrap();

    let widened: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(widened, vec![-129, -1, 0, 1, 127, 128, 255, 256, 32767]);

    let narrowed: Vec<i8> = ds.read::<i8>().unwrap();
    assert_eq!(narrowed, vec![-128, -1, 0, 1, 127, 127, 127, 127, 127]);

    let to_unsigned: Vec<u8> = ds.read::<u8>().unwrap();
    assert_eq!(to_unsigned, vec![0, 0, 0, 1, 127, 128, 255, 255, 255]);
}

#[test]
fn t3a_integer_unsigned_widening_and_narrowing() {
    let f = File::open("tests/data/hdf5_ref/integer_conversion_vectors.h5").unwrap();
    let ds = f.dataset("u16_conversion").unwrap();

    let widened: Vec<u32> = ds.read::<u32>().unwrap();
    assert_eq!(widened, vec![0, 1, 127, 128, 255, 256, 32767, 32768, 65535]);

    let to_signed: Vec<i8> = ds.read::<i8>().unwrap();
    assert_eq!(to_signed, vec![0, 1, 127, 127, 127, 127, 127, 127, 127]);

    let narrowed: Vec<u8> = ds.read::<u8>().unwrap();
    assert_eq!(narrowed, vec![0, 1, 127, 128, 255, 255, 255, 255, 255]);
}

#[test]
fn t3a_float32() {
    let vals: Vec<f32> = open().dataset("float32").unwrap().read::<f32>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

#[test]
fn t3a_float64() {
    let vals: Vec<f64> = open().dataset("float64").unwrap().read::<f64>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

#[test]
fn t3a_float_widening_narrowing_nan_inf_and_endian() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();

    let f32_to_f64: Vec<f64> = f.dataset("f32_conversion").unwrap().read::<f64>().unwrap();
    let expected_f64 = [
        f64::NEG_INFINITY,
        -129.75,
        -1.5,
        -0.0,
        0.0,
        1.5,
        127.25,
        128.75,
        f64::INFINITY,
        f64::NAN,
    ];
    assert_eq!(
        f32_to_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let f64_to_f32: Vec<f32> = f.dataset("f64_conversion").unwrap().read::<f32>().unwrap();
    let expected_f32 = [
        f32::NEG_INFINITY,
        -129.75,
        -1.5,
        -0.0,
        0.0,
        1.5,
        127.25,
        128.75,
        f32::INFINITY,
        f32::NAN,
    ];
    assert_eq!(
        f64_to_f32
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_f32
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let be_f32_to_f64: Vec<f64> = f
        .dataset("be_f32_conversion")
        .unwrap()
        .read::<f64>()
        .unwrap();
    assert_eq!(
        be_f32_to_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn t3a_integer_to_float_conversions() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();

    let signed: Vec<f32> = f
        .dataset("i16_to_float_conversion")
        .unwrap()
        .read::<f32>()
        .unwrap();
    assert_eq!(
        signed,
        vec![-129.0, -1.0, 0.0, 1.0, 127.0, 128.0, 255.0, 32767.0]
    );

    let unsigned: Vec<f64> = f
        .dataset("u16_to_float_conversion")
        .unwrap()
        .read::<f64>()
        .unwrap();
    assert_eq!(
        unsigned,
        vec![0.0, 1.0, 127.0, 128.0, 255.0, 256.0, 32767.0, 65535.0]
    );
}

#[test]
fn t3a_float_to_integer_conversions() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();
    let ds = f.dataset("f64_conversion").unwrap();

    let signed: Vec<i8> = ds.read::<i8>().unwrap();
    assert_eq!(signed, vec![-128, -128, -1, 0, 0, 1, 127, 127, 127, 0]);

    let unsigned: Vec<u8> = ds.read::<u8>().unwrap();
    assert_eq!(unsigned, vec![0, 0, 0, 0, 0, 1, 127, 128, 255, 0]);
}

// T3a: Big-endian variants

#[test]
fn t3a_be_float64() {
    let vals: Vec<f64> = open().dataset("be_f8").unwrap().read::<f64>().unwrap();
    assert_eq!(vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn t3a_be_int32() {
    let vals: Vec<i32> = open().dataset("be_i4").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn t3a_be_uint16() {
    let vals: Vec<u16> = open().dataset("be_u2").unwrap().read::<u16>().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

// T3b: Compound type

#[test]
fn t3b_compound_fields() {
    let ds = open().dataset("compound").unwrap();
    let fields = ds.compound_fields().unwrap();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[1].name, "y");
    assert_eq!(fields[2].name, "flag");
}

#[test]
fn t3b_compound_read_field() {
    let ds = open().dataset("compound").unwrap();
    let x: Vec<f64> = ds.read_field::<f64>("x").unwrap();
    assert_eq!(x, vec![1.0, 3.0]);
}

// T3c: Enum type

#[test]
fn t3c_enum_members() {
    let ds = open().dataset("enum").unwrap();
    let dt = ds.dtype().unwrap();
    let members = dt.enum_members().unwrap();
    assert!(members.iter().any(|(n, v)| n == "OFF" && *v == 0));
    assert!(members.iter().any(|(n, v)| n == "ON" && *v == 1));
    assert!(members.iter().any(|(n, v)| n == "AUTO" && *v == 2));
}

#[test]
fn t3c_enum_values() {
    let vals: Vec<u8> = open().dataset("enum").unwrap().read::<u8>().unwrap();
    assert_eq!(vals, vec![0, 1, 2]);
}

#[test]
fn t3c_enum_u16_big_endian_members_and_values() {
    let f = File::open("tests/data/hdf5_ref/enum_conversion_cases.h5").unwrap();
    let ds = f.dataset("enum_u16be").unwrap();
    let dt = ds.dtype().unwrap();
    let members = dt.enum_members().unwrap();
    assert!(members.iter().any(|(n, v)| n == "LOW" && *v == 1));
    assert!(members.iter().any(|(n, v)| n == "MID" && *v == 258));
    assert!(members.iter().any(|(n, v)| n == "HIGH" && *v == 4095));

    let vals: Vec<u16> = ds.read::<u16>().unwrap();
    assert_eq!(vals, vec![1, 258, 4095]);
}

// T3d: String types

#[test]
fn t3d_fixed_string() {
    let ds = open().dataset("fstring").unwrap();
    let strings = ds.read_strings().unwrap();
    assert_eq!(strings, vec!["hello", "world"]);
}

#[test]
fn t3d_fixed_string_padding_and_charset_flags() {
    let f = File::open("tests/data/hdf5_ref/fixed_string_cases.h5").unwrap();

    let null_padded = f.dataset("null_padded").unwrap();
    assert_eq!(null_padded.dtype().unwrap().string_padding(), Some(1));
    assert_eq!(null_padded.dtype().unwrap().char_set(), Some(0));
    assert_eq!(
        null_padded.read_strings().unwrap(),
        vec!["hi", "a b", "trail "]
    );

    let space_padded = f.dataset("space_padded").unwrap();
    assert_eq!(space_padded.dtype().unwrap().string_padding(), Some(2));
    assert_eq!(
        space_padded.read_strings().unwrap(),
        vec!["hi", "a b", "trail"]
    );

    let null_terminated = f.dataset("null_terminated").unwrap();
    assert_eq!(null_terminated.dtype().unwrap().string_padding(), Some(0));
    assert_eq!(
        null_terminated.read_strings().unwrap(),
        vec!["hi", "a b", "trail "]
    );

    let utf8 = f.dataset("utf8_fixed").unwrap();
    assert_eq!(utf8.dtype().unwrap().char_set(), Some(1));
    assert_eq!(utf8.read_strings().unwrap(), vec!["å", "猫", "hi"]);
}

#[test]
fn t3d_vlen_string() {
    let ds = open().dataset("vstring").unwrap();
    let strings = ds.read_strings().unwrap();
    assert_eq!(strings, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn t3d_vlen_string_empty_null_utf8_and_heap_edges() {
    let f = File::open("tests/data/hdf5_ref/vlen_string_cases.h5").unwrap();

    let utf8 = f.dataset("vlen_utf8_strings").unwrap();
    assert_eq!(utf8.read_strings().unwrap(), vec!["", "猫", "å", "alpha"]);

    let heap_edges = f.dataset("vlen_global_heap_edges").unwrap();
    let long = format!("long-{}", "x".repeat(96));
    assert_eq!(
        heap_edges.read_strings().unwrap(),
        vec!["dup".to_string(), "dup".to_string(), long]
    );

    let null_descriptor = f.dataset("vlen_null_descriptor").unwrap();
    assert_eq!(null_descriptor.read_strings().unwrap(), vec!["", "kept"]);
}

#[test]
fn t3d_opaque_tag_and_raw_payload() {
    let f = File::open("tests/data/hdf5_ref/opaque_cases.h5").unwrap();
    let ds = f.dataset("opaque_tagged").unwrap();
    let dtype = ds.dtype().unwrap();

    assert_eq!(dtype.class(), DatatypeClass::Opaque);
    assert_eq!(dtype.size(), 4);
    assert_eq!(
        dtype.opaque_tag().as_deref(),
        Some("hdf5-pure-rust opaque fixture")
    );
    assert_eq!(ds.read_raw().unwrap(), b"abcd\x00\x01\x02\x03wxyz".to_vec());
}

#[test]
fn t3d_reference_object_and_region_payloads() {
    let f = File::open("tests/data/hdf5_ref/reference_cases.h5").unwrap();

    let object_refs = f.dataset("object_refs").unwrap();
    let object_dtype = object_refs.dtype().unwrap();
    assert_eq!(object_dtype.class(), DatatypeClass::Reference);
    assert_eq!(object_dtype.size(), 8);
    assert_eq!(object_dtype.reference_type(), Some(0));
    let object_raw = object_refs.read_raw().unwrap();
    assert_eq!(object_raw.len(), 24);
    assert!(object_raw[0..8].iter().any(|&b| b != 0));
    assert!(object_raw[8..16].iter().any(|&b| b != 0));
    assert!(object_raw[16..24].iter().all(|&b| b == 0));
    assert_ne!(&object_raw[0..8], &object_raw[8..16]);

    let region_refs = f.dataset("region_refs").unwrap();
    let region_dtype = region_refs.dtype().unwrap();
    assert_eq!(region_dtype.class(), DatatypeClass::Reference);
    assert_eq!(region_dtype.size(), 12);
    assert_eq!(region_dtype.reference_type(), Some(1));
    let region_raw = region_refs.read_raw().unwrap();
    assert_eq!(region_raw.len(), 24);
    assert!(region_raw[0..12].iter().any(|&b| b != 0));
    assert!(region_raw[12..24].iter().all(|&b| b == 0));
}

#[test]
fn t3d_time_datatype_raw_read_and_typed_rejection() {
    let f = File::open("tests/data/hdf5_ref/time_cases.h5").unwrap();

    let d32 = f.dataset("unix_d32le").unwrap();
    let dtype32 = d32.dtype().unwrap();
    assert_eq!(dtype32.class(), DatatypeClass::Time);
    assert_eq!(dtype32.size(), 4);
    assert_eq!(
        d32.read_raw().unwrap(),
        [0_u32, 1, 2_147_483_647]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert!(matches!(d32.read::<u32>(), Err(Error::Unsupported(_))));

    let d64 = f.dataset("unix_d64be").unwrap();
    let dtype64 = d64.dtype().unwrap();
    assert_eq!(dtype64.class(), DatatypeClass::Time);
    assert_eq!(dtype64.size(), 8);
    assert_eq!(
        d64.read_raw().unwrap(),
        [0_u64, 1, 4_102_444_800]
            .into_iter()
            .flat_map(u64::to_be_bytes)
            .collect::<Vec<_>>()
    );
    assert!(matches!(d64.read::<u64>(), Err(Error::Unsupported(_))));
}

#[test]
fn t3d_array_datatype_multidimensional_dims_and_raw_payload() {
    let f = File::open("tests/data/hdf5_ref/array_datatype_cases.h5").unwrap();
    let ds = f.dataset("array_i16_2x3").unwrap();
    let dtype = ds.dtype().unwrap();
    let (dims, base) = dtype.array_dims_base().unwrap();

    assert_eq!(dtype.class(), DatatypeClass::Array);
    assert_eq!(dtype.size(), 12);
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.class(), DatatypeClass::FixedPoint);
    assert_eq!(base.size(), 2);
    assert_eq!(
        ds.read_raw().unwrap(),
        (0_i16..12).flat_map(i16::to_le_bytes).collect::<Vec<_>>()
    );
}

// T3e: Multi-dimensional

#[test]
fn t3e_2d_matrix() {
    let ds = open().dataset("matrix").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![3, 4]);
    let arr = ds.read_2d::<i32>().unwrap();
    assert_eq!(arr[[0, 0]], 0);
    assert_eq!(arr[[2, 3]], 11);
}

#[test]
fn t3e_3d_cube() {
    let ds = open().dataset("cube").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3, 4]);
    let vals: Vec<f32> = ds.read::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[23], 23.0);
}

// T3f: Null dataspace (empty dataset)

#[test]
fn t3f_null_dataspace() {
    let ds = open().dataset("null").unwrap();
    let space = ds.space().unwrap();
    // h5py creates this as null dataspace (no data), not scalar
    assert!(space.is_null() || space.is_scalar());
    assert_eq!(space.ndim(), 0);
}
