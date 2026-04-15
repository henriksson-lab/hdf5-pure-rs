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
