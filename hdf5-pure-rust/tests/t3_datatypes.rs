//! Phase T3: Datatype tests -- read all primitive, compound, enum, string types.

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
fn t3a_float32() {
    let vals: Vec<f32> = open().dataset("float32").unwrap().read::<f32>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
}

#[test]
fn t3a_float64() {
    let vals: Vec<f64> = open().dataset("float64").unwrap().read::<f64>().unwrap();
    assert_eq!(vals, vec![1.5, 2.5, 3.5]);
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

// T3d: String types

#[test]
fn t3d_fixed_string() {
    let ds = open().dataset("fstring").unwrap();
    let strings = ds.read_strings().unwrap();
    assert_eq!(strings, vec!["hello", "world"]);
}

#[test]
fn t3d_vlen_string() {
    let ds = open().dataset("vstring").unwrap();
    let strings = ds.read_strings().unwrap();
    assert_eq!(strings, vec!["alpha", "beta", "gamma"]);
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
