use std::fs;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::File;

#[test]
fn test_write_compact_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_compact.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<u8> = vec![1u8, 2, 3, 4, 5];

        w.create_compact_dataset(
            "/",
            &DatasetSpec {
                name: "small",
                shape: &[5],
                max_shape: None,
                dtype: DtypeSpec::U8,
                data: &data,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back with pure-Rust
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("small").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5]);
        let raw = ds.read_raw().unwrap();
        assert_eq!(raw, vec![1, 2, 3, 4, 5]);
    }

    // Verify with h5dump
    {
        let out = std::process::Command::new("h5dump").arg(&path).output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump compact:\n{stdout}");
            assert!(
                out.status.success(),
                "h5dump failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(stdout.contains("1, 2, 3, 4, 5"));
        }
    }
}

#[test]
fn test_engine_writer_rejects_compact_data_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_compact_bad_len.h5");

    let f = fs::File::create(&path).unwrap();
    let mut w = HdfFileWriter::new(f);
    w.begin().unwrap();
    w.create_root_group().unwrap();

    let err = w
        .create_compact_dataset(
            "/",
            &DatasetSpec {
                name: "bad",
                shape: &[2],
                max_shape: None,
                dtype: DtypeSpec::I16,
                data: &[1, 2],
            },
        )
        .expect_err("compact data byte length should match shape * dtype size");
    assert!(err.to_string().contains("dataset byte length"));
}

#[test]
fn test_compact_zero_sized_dataset_read() {
    let f = File::open("tests/data/hdf5_ref/compact_read_cases.h5").unwrap();
    let ds = f.dataset("compact_zero").unwrap();

    assert_eq!(ds.shape().unwrap(), vec![0]);
    assert_eq!(ds.size().unwrap(), 0);
    assert!(ds.read_raw().unwrap().is_empty());

    let vals: Vec<i32> = ds.read::<i32>().unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_compact_scalar_compound_payload_read() {
    let f = File::open("tests/data/hdf5_ref/compact_read_cases.h5").unwrap();
    let ds = f.dataset("compact_compound_scalar").unwrap();

    assert_eq!(ds.shape().unwrap(), Vec::<u64>::new());
    assert_eq!(ds.size().unwrap(), 1);
    assert_eq!(ds.read_raw().unwrap().len(), 12);

    let fields = ds.compound_fields().unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[1].name, "label");

    let x_vals: Vec<f64> = ds.read_field("x").unwrap();
    assert_eq!(x_vals, vec![1.5]);

    let labels: Vec<i32> = ds.read_field("label").unwrap();
    assert_eq!(labels, vec![7]);
}
