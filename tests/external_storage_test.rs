use hdf5_pure_rust::{Error, File};

#[test]
fn test_external_raw_data_storage_is_explicitly_unsupported() {
    let f = File::open("tests/data/hdf5_ref/external_raw_storage.h5").unwrap();
    let ds = f.dataset("external_raw").unwrap();

    let err = ds
        .read::<i32>()
        .expect_err("external raw data storage should be rejected explicitly");

    assert!(matches!(err, Error::Unsupported(_)));
    assert!(
        err.to_string().contains("external raw data storage"),
        "unexpected error: {err}"
    );
}
