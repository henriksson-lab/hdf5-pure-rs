use hdf5_pure_rust::{File, H5Value};

#[test]
fn test_compound_dtype_info() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_compound());
    assert_eq!(dtype.size(), 20); // f64 + f64 + i32

    let fields = dtype.compound_fields().unwrap();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[0].byte_offset, 0);
    assert_eq!(fields[0].size, 8);
    assert_eq!(fields[1].name, "y");
    assert_eq!(fields[1].byte_offset, 8);
    assert_eq!(fields[1].size, 8);
    assert_eq!(fields[2].name, "label");
    assert_eq!(fields[2].byte_offset, 16);
    assert_eq!(fields[2].size, 4);
}

#[test]
fn test_compound_read_field_f64() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let x_vals: Vec<f64> = ds.read_field::<f64>("x").unwrap();
    assert_eq!(x_vals, vec![1.0, 3.0, 5.0]);

    let y_vals: Vec<f64> = ds.read_field::<f64>("y").unwrap();
    assert_eq!(y_vals, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_compound_read_field_i32() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let labels: Vec<i32> = ds.read_field::<i32>("label").unwrap();
    assert_eq!(labels, vec![10, 20, 30]);
}

#[test]
fn test_compound_read_field_raw() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let raw = ds.read_field_raw("label").unwrap();
    let labels: Vec<i32> = raw
        .iter()
        .map(|bytes| i32::from_le_bytes(bytes.as_slice().try_into().unwrap()))
        .collect();
    assert_eq!(labels, vec![10, 20, 30]);
}

#[test]
fn test_compound_read_field_wrong_size() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    // Try to read f64 field as i32 (wrong size)
    let result = ds.read_field::<i32>("x");
    assert!(result.is_err());
}

#[test]
fn test_compound_fields_api() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let fields = ds.compound_fields().unwrap();
    let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["x", "y", "label"]);
}

#[test]
fn test_recursive_compound_nested_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let nested = ds.read_field_values("nested").unwrap();
    assert_eq!(
        nested[0],
        H5Value::Compound(vec![
            ("a".to_string(), H5Value::Int(7)),
            ("b".to_string(), H5Value::Float(1.5)),
        ])
    );
    assert_eq!(
        nested[1],
        H5Value::Compound(vec![
            ("a".to_string(), H5Value::Int(8)),
            ("b".to_string(), H5Value::Float(2.5)),
        ])
    );
}

#[test]
fn test_recursive_compound_array_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let arrays = ds.read_field_values("arr").unwrap();
    assert_eq!(
        arrays[0],
        H5Value::Array(vec![H5Value::Int(1), H5Value::Int(2), H5Value::Int(3),])
    );
    assert_eq!(
        arrays[1],
        H5Value::Array(vec![H5Value::Int(4), H5Value::Int(5), H5Value::Int(6),])
    );
}

#[test]
fn test_recursive_compound_vlen_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let values = ds.read_field_values("vlen").unwrap();
    assert_eq!(
        values[0],
        H5Value::VarLen(vec![H5Value::Int(10), H5Value::Int(11)])
    );
    assert_eq!(
        values[1],
        H5Value::VarLen(vec![H5Value::Int(20), H5Value::Int(21), H5Value::Int(22),])
    );
}

#[test]
fn test_recursive_compound_reference_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let refs = ds.read_field_values("ref").unwrap();
    match (&refs[0], &refs[1]) {
        (H5Value::Reference(a), H5Value::Reference(b)) => {
            assert_ne!(*a, 0);
            assert_eq!(a, b);
        }
        other => panic!("unexpected reference values: {other:?}"),
    }
}

#[test]
fn test_compound_padded_reordered_members() {
    let f = File::open("tests/data/hdf5_ref/compound_layout_cases.h5").unwrap();
    let ds = f.dataset("padded_reordered").unwrap();
    let fields = ds.compound_fields().unwrap();

    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "second");
    assert_eq!(fields[0].byte_offset, 4);
    assert_eq!(fields[1].name, "first");
    assert_eq!(fields[1].byte_offset, 0);
    assert_eq!(fields[2].name, "last");
    assert_eq!(fields[2].byte_offset, 8);
    assert_eq!(ds.dtype().unwrap().size(), 12);

    assert_eq!(ds.read_field::<i32>("first").unwrap(), vec![1000, 2000]);
    assert_eq!(ds.read_field::<i16>("second").unwrap(), vec![10, 20]);
    assert_eq!(ds.read_field::<u8>("last").unwrap(), vec![7, 8]);
}

#[test]
fn test_recursive_compound_nested_vlen_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_layout_cases.h5").unwrap();
    let ds = f.dataset("nested_vlen").unwrap();

    let values = ds.read_field_values("nested_vlen").unwrap();
    assert_eq!(
        values[0],
        H5Value::Compound(vec![
            ("tag".to_string(), H5Value::Int(3)),
            (
                "seq".to_string(),
                H5Value::VarLen(vec![H5Value::Int(1), H5Value::Int(2)])
            ),
        ])
    );
    assert_eq!(
        values[1],
        H5Value::Compound(vec![
            ("tag".to_string(), H5Value::Int(4)),
            (
                "seq".to_string(),
                H5Value::VarLen(vec![H5Value::Int(5), H5Value::Int(6), H5Value::Int(7)])
            ),
        ])
    );
}

#[test]
fn test_compound_multidimensional_array_member_values() {
    let f = File::open("tests/data/hdf5_ref/array_datatype_cases.h5").unwrap();
    let ds = f.dataset("compound_array2d").unwrap();
    let fields = ds.compound_fields().unwrap();
    let grid = fields.iter().find(|field| field.name == "grid").unwrap();
    let (dims, base) = grid.datatype.array_dims_base().unwrap();

    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size, 2);
    let values = ds.read_field_values("grid").unwrap();
    assert_eq!(
        values[0],
        H5Value::Array(vec![
            H5Value::Int(1),
            H5Value::Int(2),
            H5Value::Int(3),
            H5Value::Int(4),
            H5Value::Int(5),
            H5Value::Int(6),
        ])
    );
    assert_eq!(
        values[1],
        H5Value::Array(vec![
            H5Value::Int(7),
            H5Value::Int(8),
            H5Value::Int(9),
            H5Value::Int(10),
            H5Value::Int(11),
            H5Value::Int(12),
        ])
    );
}
