use hdf5_pure_rust::{Error, File};
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct ThreeBytes([u8; 3]);

unsafe impl hdf5_pure_rust::H5Type for ThreeBytes {
    fn type_size() -> usize {
        3
    }
}

fn vds_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

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

#[test]
fn test_virtual_dataset_f64_read() {
    let f = File::open("tests/data/hdf5_ref/vds_f64.h5").unwrap();
    let ds = f.dataset("vds_f64").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![3, 4]);

    let vals: Vec<f64> = ds.read::<f64>().unwrap();
    let expected = (0..12)
        .map(|value| (value as f64 / 2.0) + 0.25)
        .collect::<Vec<_>>();
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_rejects_mismatched_read_element_size() {
    let f = File::open("tests/data/hdf5_ref/vds_f64.h5").unwrap();
    let ds = f.dataset("vds_f64").unwrap();
    let err = ds
        .read::<ThreeBytes>()
        .expect_err("VDS read should reject mismatched destination element sizes");

    assert!(matches!(err, Error::InvalidFormat(_)));
}

#[test]
fn test_virtual_dataset_scalar_mapping_read() {
    let f = File::open("tests/data/hdf5_ref/vds_scalar.h5").unwrap();
    let ds = f.dataset("vds_scalar").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), Vec::<u64>::new());
    assert_eq!(ds.size().unwrap(), 1);

    let val = ds.read_scalar::<i32>().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn test_virtual_dataset_zero_sized_mapping_read() {
    let f = File::open("tests/data/hdf5_ref/vds_zero_sized.h5").unwrap();
    let ds = f.dataset("vds_zero_sized").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![0, 4]);
    assert_eq!(ds.size().unwrap(), 0);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_virtual_dataset_null_mapping_read() {
    let f = File::open("tests/data/hdf5_ref/vds_null.h5").unwrap();
    let ds = f.dataset("vds_null").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert!(ds.space().unwrap().is_null());
    assert_eq!(ds.shape().unwrap(), Vec::<u64>::new());
    assert_eq!(ds.size().unwrap(), 0);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_virtual_dataset_rank_mismatch_mapping_read() {
    let f = File::open("tests/data/hdf5_ref/vds_rank_mismatch.h5").unwrap();
    let ds = f.dataset("vds_rank_mismatch").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, (0..6).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_overlapping_mappings_later_mapping_wins() {
    let f = File::open("tests/data/hdf5_ref/vds_overlap.h5").unwrap();
    let ds = f.dataset("vds_overlap").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![4]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(vals, vec![1, 9, 8, 4]);
}

#[test]
fn test_virtual_dataset_irregular_hyperslab_read() {
    let f = File::open("tests/data/hdf5_ref/vds_irregular_hyperslab.h5").unwrap();
    let ds = f.dataset("vds_irregular_hyperslab").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![4, 4]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    let mut expected = vec![-2; 4 * 4];
    expected[1] = 1;
    expected[2] = 2;
    expected[2 * 4] = 8;
    expected[2 * 4 + 1] = 9;
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_point_selection_read() {
    let f = File::open("tests/data/hdf5_ref/vds_point_selection.h5").unwrap();
    let ds = f.dataset("vds_point_selection").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert_eq!(ds.shape().unwrap(), vec![4, 4]);

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    let mut expected = vec![-2; 4 * 4];
    expected[3] = 9;
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_missing_source_file_fails_without_access_property_policy() {
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let err = ds
        .read::<i32>()
        .expect_err("missing VDS source should fail without a VDS access policy");

    assert!(
        matches!(err, Error::Io(_)),
        "missing source should surface as file I/O without VDS access-property behavior: {err}"
    );
}

#[test]
fn test_virtual_dataset_uses_hdf5_vds_prefix_directory() {
    let _guard = vds_env_lock().lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    std::env::set_var("HDF5_VDS_PREFIX", prefixed_dir.to_str().unwrap());
    let f = File::open(&vds_path).unwrap();
    let vals: Vec<i32> = f.dataset("vds_all").unwrap().read::<i32>().unwrap();
    std::env::remove_var("HDF5_VDS_PREFIX");

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_uses_hdf5_vds_prefix_origin_expansion() {
    let _guard = vds_env_lock().lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    std::env::set_var("HDF5_VDS_PREFIX", "${ORIGIN}/prefixed");
    let f = File::open(&vds_path).unwrap();
    let vals: Vec<i32> = f.dataset("vds_all").unwrap().read::<i32>().unwrap();
    std::env::remove_var("HDF5_VDS_PREFIX");

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}
