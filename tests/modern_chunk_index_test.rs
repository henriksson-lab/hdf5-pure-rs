use hdf5_pure_rust::File;

#[test]
fn test_v4_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_chunks.h5").unwrap();
    let vals: Vec<i32> = f.dataset("fixed_array").unwrap().read::<i32>().unwrap();
    assert_eq!(vals, (0..100).collect::<Vec<_>>());
}

#[test]
fn test_v4_paged_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5").unwrap();
    let vals: Vec<i32> = f
        .dataset("fixed_array_paged")
        .unwrap()
        .read::<i32>()
        .unwrap();
    assert_eq!(vals.len(), 4096);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[1024], 1024);
    assert_eq!(vals[4095], 4095);
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
fn test_sparse_chunked_fill_value_read() {
    let f = File::open("tests/data/hdf5_ref/sparse_chunked_fill_value.h5").unwrap();
    let vals: Vec<i32> = f
        .dataset("sparse_chunked_fill")
        .unwrap()
        .read::<i32>()
        .unwrap();

    let mut expected = vec![-7; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[row * 6 + col] = (row * 3 + col) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_filtered_chunk_mask_skips_unapplied_filters() {
    let f = File::open("tests/data/hdf5_ref/filtered_chunk_filter_mask.h5").unwrap();
    let vals: Vec<i32> = f
        .dataset("filtered_chunk_filter_mask")
        .unwrap()
        .read::<i32>()
        .unwrap();

    let mut expected = vec![-7; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[row * 6 + col] = (row * 3 + col) as i32;
            expected[(row + 2) * 6 + col + 3] = (row * 3 + col + 100) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_filtered_single_chunk_mask_skips_unapplied_filters() {
    let f = File::open("tests/data/hdf5_ref/filtered_single_chunk_filter_mask.h5").unwrap();
    let ds = f.dataset("filtered_single_chunk_filter_mask").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, (0..6).collect::<Vec<_>>());
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
fn test_v4_extensible_array_spillover_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_spillover.h5").unwrap();
    let vals: Vec<f64> = f
        .dataset("extensible_array_spillover")
        .unwrap()
        .read::<f64>()
        .unwrap();
    assert_eq!(vals.len(), 4096);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[8], 8.0);
    assert_eq!(vals[4095], 4095.0);
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

#[test]
fn test_scaleoffset_filter_f32_read() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_filter_i32.h5").unwrap();
    let vals: Vec<f32> = f.dataset("scaleoffset_f32").unwrap().read::<f32>().unwrap();
    let expected: Vec<f32> = (0..40).map(|v| v as f32 / 10.0 + 1.25).collect();
    for (actual, expected) in vals.iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.011);
    }
}
