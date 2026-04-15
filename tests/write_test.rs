use std::fs;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::File;

#[test]
fn test_write_and_read_back_simple() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_simple.h5");

    // Write
    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "mydata",
                shape: &[5],
                dtype: DtypeSpec::F64,
                data: &data_bytes,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back with our own reader
    {
        let f = File::open(&path).unwrap();
        let sb = f.superblock();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.sizeof_addr, 8);

        let names = f.member_names().unwrap();
        println!("Written file members: {names:?}");
        assert!(names.contains(&"mydata".to_string()));

        let ds = f.dataset("mydata").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5]);
        assert_eq!(ds.element_size().unwrap(), 8);

        let raw = ds.read_raw().unwrap();
        let values: Vec<f64> = raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}

#[test]
fn test_write_multiple_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_multi.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let f64_data: Vec<u8> = vec![1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "floats",
                shape: &[3],
                dtype: DtypeSpec::F64,
                data: &f64_data,
            },
        )
        .unwrap();

        let i32_data: Vec<u8> = vec![10i32, 20, 30, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "ints",
                shape: &[4],
                dtype: DtypeSpec::I32,
                data: &i32_data,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let names = f.member_names().unwrap();
        assert!(names.contains(&"floats".to_string()));
        assert!(names.contains(&"ints".to_string()));

        let ds1 = f.dataset("floats").unwrap();
        let raw1 = ds1.read_raw().unwrap();
        let vals1: Vec<f64> = raw1
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals1, vec![1.0, 2.0, 3.0]);

        let ds2 = f.dataset("ints").unwrap();
        let raw2 = ds2.read_raw().unwrap();
        let vals2: Vec<i32> = raw2
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals2, vec![10, 20, 30, 40]);
    }
}

#[test]
fn test_write_with_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_group.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();
        w.create_group("/", "subgroup").unwrap();

        let data: Vec<u8> = vec![42u8];
        w.create_dataset(
            "/subgroup",
            &DatasetSpec {
                name: "tiny",
                shape: &[1],
                dtype: DtypeSpec::U8,
                data: &data,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let names = f.member_names().unwrap();
        assert!(names.contains(&"subgroup".to_string()));

        let g = f.group("subgroup").unwrap();
        let g_names = g.member_names().unwrap();
        assert!(g_names.contains(&"tiny".to_string()));

        let ds = f.dataset("subgroup/tiny").unwrap();
        let raw = ds.read_raw().unwrap();
        assert_eq!(raw, vec![42]);
    }
}

#[test]
fn test_write_readable_by_h5dump() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_h5dump.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "data",
                shape: &[3],
                dtype: DtypeSpec::F64,
                data: &data,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Verify with h5dump if available
    let output = std::process::Command::new("h5dump").arg(&path).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("h5dump stdout:\n{stdout}");
            if !stderr.is_empty() {
                println!("h5dump stderr:\n{stderr}");
            }
            assert!(out.status.success(), "h5dump failed on written file");
            assert!(
                stdout.contains("1"),
                "h5dump output should contain data values"
            );
        }
        Err(e) => {
            println!("h5dump not available: {e}, skipping C library verification");
        }
    }
}
