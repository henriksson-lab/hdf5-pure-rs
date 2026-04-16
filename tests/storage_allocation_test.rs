use hdf5_pure_rust::File;

#[test]
fn test_contiguous_dataset_with_undefined_storage_address_reads_fill_value() {
    let f = File::open("tests/data/hdf5_ref/undefined_storage_address.h5").unwrap();
    let ds = f.dataset("late_fill").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![4]);
    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, vec![-5, -5, -5, -5]);
}

#[test]
fn test_late_allocation_fill_time_never_reads_zeroes() {
    let f = File::open("tests/data/hdf5_ref/late_fill_time_never.h5").unwrap();
    let ds = f.dataset("late_never").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![4]);
    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, vec![0, 0, 0, 0]);
}
