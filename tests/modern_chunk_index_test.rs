use hdf5_pure_rust::File;

#[test]
fn test_v4_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_chunks.h5").unwrap();
    let vals: Vec<i32> = f.dataset("fixed_array").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, (0..100).collect::<Vec<_>>());
}

#[test]
fn test_v4_filtered_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_filtered_chunked.h5").unwrap();
    let vals: Vec<i16> = f
        .dataset("filtered_chunked")
        .unwrap()
        .read::<i16>()
        .unwrap();
    assert_eq!(vals, (0..64).collect::<Vec<_>>());
}

#[test]
fn test_v4_extensible_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_chunks.h5").unwrap();
    let vals: Vec<f64> = f
        .dataset("extensible_array")
        .unwrap()
        .read::<f64>()
        .unwrap();
    assert_eq!(vals, (0..80).map(|v| v as f64).collect::<Vec<_>>());
}

#[test]
fn test_v4_btree2_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_btree2_chunks.h5").unwrap();
    let vals: Vec<i32> = f.dataset("btree_v2").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, (0..64).collect::<Vec<_>>());
}

#[test]
fn test_v4_btree2_internal_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_btree2_internal_chunks.h5").unwrap();
    let vals: Vec<i32> = f
        .dataset("btree_v2_internal")
        .unwrap()
        .read::<i32>()
        .unwrap();
    assert_eq!(vals.len(), 80 * 80);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[79], 79);
    assert_eq!(vals[80], 80);
    assert_eq!(vals[6399], 6399);
}

#[test]
fn test_nbit_filter_i32_read() {
    let f = File::open("tests/data/hdf5_ref/nbit_filter_i32.h5").unwrap();
    let vals: Vec<i32> = f.dataset("nbit_i32").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, (0..100).collect::<Vec<_>>());
}

#[test]
fn test_scaleoffset_filter_i32_read() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_filter_i32.h5").unwrap();
    let vals: Vec<i32> = f.dataset("scaleoffset_i32").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, (0..100).collect::<Vec<_>>());
}
