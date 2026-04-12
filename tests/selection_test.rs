use hdf5_pure_rust::File;

#[test]
fn test_read_slice_1d_range() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    // Read elements 1..4 (indices 1, 2, 3)
    let vals: Vec<f64> = ds.read_slice::<f64, _>(1..4).unwrap();
    assert_eq!(vals, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_read_slice_1d_from_start() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..3).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_read_slice_1d_to_end() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(3..).unwrap();
    assert_eq!(vals, vec![4.0, 5.0]);
}

#[test]
fn test_read_slice_all() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let vals: Vec<f64> = ds.read_slice::<f64, _>(..).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_slice_2d() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    // Read row 1 only (row 1, all columns)
    let vals: Vec<i8> = ds.read_slice::<i8, _>((1..2, 0..3)).unwrap();
    assert_eq!(vals, vec![4, 5, 6]);
}

#[test]
fn test_read_slice_2d_subregion() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    // Read rows 0-1, cols 1-2
    let vals: Vec<i8> = ds.read_slice::<i8, _>((0..2, 1..3)).unwrap();
    assert_eq!(vals, vec![2, 3, 5, 6]);
}

#[test]
fn test_read_slice_chunked() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    // Read elements 50..55 from a chunked dataset
    let vals: Vec<f32> = ds.read_slice::<f32, _>(50..55).unwrap();
    assert_eq!(vals, vec![50.0, 51.0, 52.0, 53.0, 54.0]);
}
