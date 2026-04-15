//! Phase T7: Dataspace and selection tests.

use hdf5_pure_rust::File;

const FILE: &str = "tests/data/hdf5_ref/selections_test.h5";
fn open() -> File {
    File::open(FILE).unwrap()
}

// T7a: Dataspace types

#[test]
fn t7a_simple_dataspace() {
    let ds = open().dataset("seq100").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_simple());
    assert!(!space.is_scalar());
    assert!(!space.is_null());
    assert_eq!(space.ndim(), 1);
    assert_eq!(space.shape(), &[100]);
    assert_eq!(space.size(), 100);
}

#[test]
fn t7a_scalar_dataspace() {
    let ds = open().dataset("scalar_val").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_scalar());
    assert_eq!(space.ndim(), 0);
    assert_eq!(space.size(), 1);
}

#[test]
fn t7a_null_dataspace() {
    let ds = open().dataset("null_ds").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_null() || space.is_scalar());
    assert_eq!(space.ndim(), 0);
}

// T7b: Dimension queries

#[test]
fn t7b_2d_shape() {
    let ds = open().dataset("matrix").unwrap();
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), &[6, 10]);
    assert_eq!(space.ndim(), 2);
    assert_eq!(space.size(), 60);
}

#[test]
fn t7b_3d_shape() {
    let ds = open().dataset("cube").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4, 5, 6]);
}

#[test]
fn t7b_resizable_maxdims() {
    let ds = open().dataset("resizable").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_resizable());
    let maxdims = space.maxdims().unwrap();
    assert_eq!(maxdims[0], u64::MAX); // unlimited
}

#[test]
fn t7b_non_resizable() {
    let ds = open().dataset("seq100").unwrap();
    assert!(!ds.space().unwrap().is_resizable());
}

// T7c-e: Selection / read_slice tests

#[test]
fn t7c_slice_1d_range() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(10..20).unwrap();
    assert_eq!(vals.len(), 10);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, (10 + i) as f64);
    }
}

#[test]
fn t7c_slice_1d_from_start() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..5).unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn t7c_slice_1d_to_end() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(95..).unwrap();
    assert_eq!(vals, vec![95.0, 96.0, 97.0, 98.0, 99.0]);
}

#[test]
fn t7c_slice_all() {
    let ds = open().dataset("seq100").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..).unwrap();
    assert_eq!(vals.len(), 100);
}

#[test]
fn t7d_slice_2d() {
    let ds = open().dataset("matrix").unwrap();
    // Read rows 1-3, cols 2-5
    let vals: Vec<i32> = ds.read_slice::<i32, _>((1..3, 2..5)).unwrap();
    // row 1: [10,11,12,13,14,15,16,17,18,19], cols 2-4 = [12,13,14]
    // row 2: [20,21,22,23,24,25,26,27,28,29], cols 2-4 = [22,23,24]
    assert_eq!(vals, vec![12, 13, 14, 22, 23, 24]);
}

#[test]
fn t7d_slice_2d_single_row() {
    let ds = open().dataset("matrix").unwrap();
    let vals: Vec<i32> = ds.read_slice::<i32, _>((0..1, 0..10)).unwrap();
    assert_eq!(vals, (0..10).collect::<Vec<i32>>());
}

#[test]
fn t7e_slice_chunked() {
    let ds = open().dataset("chunked_seq").unwrap();
    // Slice across chunk boundary (chunks are 25 elements)
    let vals: Vec<i64> = ds.read_slice::<i64, _>(20..30).unwrap();
    assert_eq!(vals.len(), 10);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, (20 + i) as i64);
    }
}

#[test]
fn t7e_slice_chunked_all() {
    let ds = open().dataset("chunked_seq").unwrap();
    let vals: Vec<i64> = ds.read_slice::<i64, _>(..).unwrap();
    assert_eq!(vals.len(), 200);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[199], 199);
}

#[test]
fn t7e_read_scalar() {
    let ds = open().dataset("scalar_val").unwrap();
    let val: f64 = ds.read_scalar::<f64>().unwrap();
    assert_eq!(val, 42.0);
}
