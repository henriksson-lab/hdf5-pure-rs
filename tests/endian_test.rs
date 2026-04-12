use hdf5_pure_rust::File;

#[test]
fn test_read_bigendian_float() {
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("be_float").unwrap();

    let dtype = ds.dtype().unwrap();
    assert_eq!(dtype.byte_order(), Some(hdf5_pure_rust::format::messages::datatype::ByteOrder::BigEndian));

    let vals: Vec<f64> = ds.read::<f64>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_read_bigendian_int() {
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("be_int").unwrap();

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, vec![10, 20, 30]);
}

#[test]
fn test_read_littleendian_unchanged() {
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("le_float").unwrap();

    let vals: Vec<f64> = ds.read::<f64>().unwrap();
    assert_eq!(vals, vec![4.0, 5.0, 6.0]);
}
