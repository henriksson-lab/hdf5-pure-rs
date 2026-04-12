use hdf5_pure_rust::File;

fn bytes_to_f64_le(data: &[u8]) -> Vec<f64> {
    data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_i32_le(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_f32_le(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn test_read_float64_1d_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![5]);
    assert_eq!(ds.element_size().unwrap(), 8);

    let raw = ds.read_raw().unwrap();
    let values = bytes_to_f64_le(&raw);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_int32_1d_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![3]);
    assert_eq!(ds.element_size().unwrap(), 4);

    let raw = ds.read_raw().unwrap();
    let values = bytes_to_i32_le(&raw);
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_read_scalar_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();

    assert_eq!(ds.shape().unwrap(), Vec::<u64>::new());
    assert_eq!(ds.size().unwrap(), 1);

    let raw = ds.read_raw().unwrap();
    let value = f64::from_le_bytes(raw[..8].try_into().unwrap());
    assert_eq!(value, 42.0);
}

#[test]
fn test_read_int8_2d_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    assert_eq!(ds.element_size().unwrap(), 1);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_read_chunked_compressed_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![100]);
    assert_eq!(ds.element_size().unwrap(), 4);

    let raw = ds.read_raw().unwrap();
    let values = bytes_to_f32_le(&raw);
    assert_eq!(values.len(), 100);

    for (i, val) in values.iter().enumerate() {
        assert_eq!(*val, i as f32, "mismatch at index {i}");
    }
}

#[test]
fn test_read_float64_1d_v3() {
    let f = File::open("tests/data/datasets_v3.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();

    let raw = ds.read_raw().unwrap();
    let values = bytes_to_f64_le(&raw);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_int32_1d_v3() {
    let f = File::open("tests/data/datasets_v3.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();

    let raw = ds.read_raw().unwrap();
    let values = bytes_to_i32_le(&raw);
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_read_scalar_v3() {
    let f = File::open("tests/data/datasets_v3.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();

    let raw = ds.read_raw().unwrap();
    let value = f64::from_le_bytes(raw[..8].try_into().unwrap());
    assert_eq!(value, 42.0);
}
