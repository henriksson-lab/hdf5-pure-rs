use hdf5_pure_rust::File;

#[test]
fn test_reference_virtual_dataset_regular_hyperslabs_read() {
    let f = File::open("hdf5/tools/test/testfiles/vds/1_vds.h5").unwrap();
    let ds = f.dataset("vds_dset").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![5, 18, 8]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals.len(), 5 * 18 * 8);

    let row = |plane: usize, y: usize| -> &[i32] {
        let start = (plane * 18 * 8) + (y * 8);
        &vals[start..start + 8]
    };
    assert_eq!(row(0, 0), &[10; 8]);
    assert_eq!(row(0, 2), &[20; 8]);
    assert_eq!(row(0, 14), &[60; 8]);
    assert_eq!(row(4, 0), &[14; 8]);
    assert_eq!(row(4, 14), &[64; 8]);
}

#[test]
fn test_virtual_dataset_all_selection_read() {
    let f = File::open("tests/data/hdf5_ref/vds_all.h5").unwrap();
    let ds = f.dataset("vds_all").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![4, 6]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_same_file_source_read() {
    let f = File::open("tests/data/hdf5_ref/vds_same_file.h5").unwrap();
    let ds = f.dataset("vds_same_file").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![3, 4]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, (0..12).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_mixed_all_and_regular_selection_read() {
    let f = File::open("tests/data/hdf5_ref/vds_mixed_all_regular.h5").unwrap();
    let ds = f.dataset("vds_mixed_all_regular").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![4, 6]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    let mut expected = vec![0; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[(row + 1) * 6 + col + 2] = (row * 3 + col) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_fill_value_for_unmapped_regions() {
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![4, 6]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    let mut expected = vec![-7; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[(row + 1) * 6 + col + 2] = (row * 3 + col) as i32;
        }
    }
    assert_eq!(vals, expected);
}
