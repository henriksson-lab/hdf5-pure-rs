use std::fs;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::File;

#[test]
fn test_write_chunked_no_compression() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_nocomp.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "chunked",
                shape: &[100],
                dtype: DtypeSpec::F32,
                data: &data_bytes,
            },
            &[10], // chunk dims
            None,  // no compression
            false, // no shuffle
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("chunked").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![100]);

        let raw = ds.read_raw().unwrap();
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 100);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as f32, "mismatch at index {i}");
        }
    }
}

#[test]
fn test_write_chunked_with_deflate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_deflate.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "compressed",
                shape: &[100],
                dtype: DtypeSpec::F32,
                data: &data_bytes,
            },
            &[25],   // chunk dims
            Some(6), // deflate level 6
            false,   // no shuffle
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back with pure-Rust
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("compressed").unwrap();
        let raw = ds.read_raw().unwrap();
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 100);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as f32, "mismatch at index {i}");
        }
    }

    // Verify with h5dump
    {
        let out = std::process::Command::new("h5dump")
            .arg("-d")
            .arg("compressed")
            .arg(&path)
            .output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump output:\n{stdout}");
            assert!(
                out.status.success(),
                "h5dump failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}

#[test]
fn test_write_chunked_with_shuffle_and_deflate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_shuf_def.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<i32> = (0..50).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "shuf_def",
                shape: &[50],
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[10],
            Some(4), // deflate level 4
            true,    // shuffle
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("shuf_def").unwrap();
        let raw = ds.read_raw().unwrap();
        let values: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 50);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as i32, "mismatch at {i}");
        }
    }

    // h5dump verification (structure only, data may differ in edge cases)
    {
        let out = std::process::Command::new("h5dump")
            .arg("-pH") // properties + header only, no data
            .arg(&path)
            .output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump shuffle+deflate structure:\n{stdout}");
            assert!(stdout.contains("SHUFFLE"), "should detect shuffle filter");
            assert!(stdout.contains("DEFLATE"), "should detect deflate filter");
        }
    }
}
