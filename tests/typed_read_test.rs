use hdf5_pure_rust::File;

#[test]
fn test_read_typed_f64() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let values: Vec<f64> = ds.read::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_typed_i32() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();
    let values: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_read_scalar_typed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();
    let val: f64 = ds.read_scalar::<f64>().unwrap();
    assert_eq!(val, 42.0);
}

#[test]
fn test_read_1d_ndarray() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let arr = ds.read_1d::<f64>().unwrap();
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0], 1.0);
    assert_eq!(arr[4], 5.0);
}

#[test]
fn test_read_2d_ndarray() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let arr = ds.read_2d::<i8>().unwrap();
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr[[0, 0]], 1);
    assert_eq!(arr[[1, 2]], 6);
}

#[test]
fn test_read_chunked_typed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    let values: Vec<f32> = ds.read::<f32>().unwrap();
    assert_eq!(values.len(), 100);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as f32);
    }
}

#[test]
fn test_attr_read_typed() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val: i64 = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn test_attr_read_array_typed() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("array_attr").unwrap();
    let values: Vec<f64> = attr.read::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_read_wrong_type_size() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();
    // 12 bytes (3 * i32) is not a multiple of 8 (f64), should error
    let result = ds.read::<f64>();
    assert!(result.is_err());
}
