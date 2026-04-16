use std::fs;

use hdf5_pure_rust::engine::writer::{AttrSpec, DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::File;

#[test]
fn test_write_dataset_with_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_with_attrs.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let attr_data = 42i64.to_le_bytes().to_vec();

        w.create_dataset_with_attrs(
            "/",
            &DatasetSpec {
                name: "data",
                shape: &[3],
                dtype: DtypeSpec::F64,
                data: &data,
            },
            &[AttrSpec {
                name: "count",
                shape: &[],
                dtype: DtypeSpec::I64,
                data: &attr_data,
            }],
        )
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();

        // Read dataset
        let raw = ds.read_raw().unwrap();
        let values: Vec<f64> = raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        // Read attribute
        let attr_names = ds.attr_names().unwrap();
        assert!(attr_names.contains(&"count".to_string()));

        let attr = ds.attr("count").unwrap();
        assert_eq!(attr.read_scalar_i64(), Some(42));
    }

    // Verify with h5dump
    {
        let out = std::process::Command::new("h5dump").arg(&path).output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump:\n{stdout}");
            assert!(out.status.success());
            assert!(stdout.contains("count"));
        }
    }
}

#[test]
fn test_write_root_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_root_attrs.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let val_data = 3.14f64.to_le_bytes().to_vec();
        w.add_root_attr(&AttrSpec {
            name: "pi",
            shape: &[],
            dtype: DtypeSpec::F64,
            data: &val_data,
        });

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let attr_names = f.attr_names().unwrap();
        assert!(attr_names.contains(&"pi".to_string()));

        let attr = f.attr("pi").unwrap();
        let val = attr.read_scalar_f64().unwrap();
        assert!((val - 3.14).abs() < 1e-10);
    }
}

#[test]
fn test_write_dataset_with_many_compact_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_many_attrs.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data = 123i32.to_le_bytes().to_vec();
        let mut attr_names = Vec::new();
        let mut attr_payloads = Vec::new();
        for idx in 0..40 {
            attr_names.push(format!("attr_{idx:02}"));
            attr_payloads.push((idx as i64).to_le_bytes().to_vec());
        }
        let attrs: Vec<AttrSpec<'_>> = attr_names
            .iter()
            .zip(&attr_payloads)
            .map(|(name, payload)| AttrSpec {
                name,
                shape: &[],
                dtype: DtypeSpec::I64,
                data: payload,
            })
            .collect();

        w.create_dataset_with_attrs(
            "/",
            &DatasetSpec {
                name: "data",
                shape: &[1],
                dtype: DtypeSpec::I32,
                data: &data,
            },
            &attrs,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        let names = ds.attr_names().unwrap();
        assert_eq!(names.len(), 40);
        assert!(names.contains(&"attr_00".to_string()));
        assert!(names.contains(&"attr_39".to_string()));
        assert_eq!(ds.attr("attr_37").unwrap().read_scalar_i64(), Some(37));
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump failed on many-attribute writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
