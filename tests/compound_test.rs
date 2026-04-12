use hdf5_pure_rust::File;

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
