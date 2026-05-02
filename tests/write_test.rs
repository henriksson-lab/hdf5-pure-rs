use std::fs;

use hdf5_pure_rust::engine::writer::{CompoundFieldSpec, DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::hl::value::H5Value;
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
                max_shape: None,
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
fn test_engine_writer_rejects_dataset_data_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_bad_len.h5");

    let f = fs::File::create(&path).unwrap();
    let mut w = HdfFileWriter::new(f);
    w.begin().unwrap();
    w.create_root_group().unwrap();

    let err = w
        .create_dataset(
            "/",
            &DatasetSpec {
                name: "bad",
                shape: &[2],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &[1, 2, 3, 4],
            },
        )
        .expect_err("raw data byte length should match shape * dtype size");
    assert!(err.to_string().contains("dataset byte length"));
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
                max_shape: None,
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
                max_shape: None,
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
                max_shape: None,
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
fn test_write_enum_opaque_array_and_nested_compound_datatypes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_complex_dtypes.h5");
    let array_data: Vec<u8> = (0i16..12).flat_map(|v| v.to_le_bytes()).collect();

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let enum_data: Vec<u8> = [0u16, 1, 2]
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "status",
                shape: &[3],
                max_shape: None,
                dtype: DtypeSpec::Enum {
                    base: Box::new(DtypeSpec::U16),
                    members: vec![
                        ("zero".to_string(), 0),
                        ("one".to_string(), 1),
                        ("two".to_string(), 2),
                    ],
                },
                data: &enum_data,
            },
        )
        .unwrap();

        let opaque_data = b"abcdwxyz".to_vec();
        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "opaque",
                shape: &[2],
                max_shape: None,
                dtype: DtypeSpec::Opaque {
                    size: 4,
                    tag: "hdf5-pure-rust blob".to_string(),
                },
                data: &opaque_data,
            },
        )
        .unwrap();

        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "matrix_cells",
                shape: &[2],
                max_shape: None,
                dtype: DtypeSpec::Array {
                    dims: vec![2, 3],
                    base: Box::new(DtypeSpec::I16),
                },
                data: &array_data,
            },
        )
        .unwrap();

        let nested_dtype = DtypeSpec::Compound {
            size: 8,
            fields: vec![
                CompoundFieldSpec {
                    name: "a".to_string(),
                    offset: 0,
                    dtype: DtypeSpec::I32,
                },
                CompoundFieldSpec {
                    name: "b".to_string(),
                    offset: 4,
                    dtype: DtypeSpec::F32,
                },
            ],
        };
        let nested_compound_dtype = DtypeSpec::Compound {
            size: 16,
            fields: vec![
                CompoundFieldSpec {
                    name: "id".to_string(),
                    offset: 0,
                    dtype: DtypeSpec::I32,
                },
                CompoundFieldSpec {
                    name: "nested".to_string(),
                    offset: 8,
                    dtype: nested_dtype,
                },
            ],
        };
        let mut compound_data = Vec::new();
        for (id, a, b) in [(7i32, 10i32, 1.25f32), (8, 20, 2.5)] {
            compound_data.extend_from_slice(&id.to_le_bytes());
            compound_data.extend_from_slice(&[0; 4]);
            compound_data.extend_from_slice(&a.to_le_bytes());
            compound_data.extend_from_slice(&b.to_le_bytes());
        }
        w.create_dataset(
            "/",
            &DatasetSpec {
                name: "nested_compound",
                shape: &[2],
                max_shape: None,
                dtype: nested_compound_dtype,
                data: &compound_data,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();

    let enum_ds = f.dataset("status").unwrap();
    let enum_dtype = enum_ds.dtype().unwrap();
    assert!(enum_dtype.is_enum());
    assert_eq!(
        enum_dtype.enum_members().unwrap(),
        vec![
            ("zero".to_string(), 0),
            ("one".to_string(), 1),
            ("two".to_string(), 2)
        ]
    );
    assert_eq!(enum_ds.read::<u16>().unwrap(), vec![0, 1, 2]);

    let opaque_ds = f.dataset("opaque").unwrap();
    let opaque_dtype = opaque_ds.dtype().unwrap();
    assert_eq!(
        opaque_dtype.opaque_tag().as_deref(),
        Some("hdf5-pure-rust blob")
    );
    assert_eq!(opaque_ds.read_raw().unwrap(), b"abcdwxyz");

    let array_ds = f.dataset("matrix_cells").unwrap();
    let array_dtype = array_ds.dtype().unwrap();
    let (dims, base) = array_dtype.array_dims_base().unwrap();
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size(), 2);
    assert_eq!(array_ds.read_raw().unwrap(), array_data);

    let compound_ds = f.dataset("nested_compound").unwrap();
    let nested_values = compound_ds.read_field_values("nested").unwrap();
    assert_eq!(
        nested_values,
        vec![
            H5Value::Compound(vec![
                ("a".to_string(), H5Value::Int(10)),
                ("b".to_string(), H5Value::Float(1.25)),
            ]),
            H5Value::Compound(vec![
                ("a".to_string(), H5Value::Int(20)),
                ("b".to_string(), H5Value::Float(2.5)),
            ]),
        ]
    );

    for dataset in ["/status", "/opaque", "/matrix_cells", "/nested_compound"] {
        let output = std::process::Command::new("h5dump")
            .arg("-H")
            .arg("-d")
            .arg(dataset)
            .arg(&path)
            .output();
        if let Ok(out) = output {
            assert!(
                out.status.success(),
                "h5dump -H failed on dataset {dataset}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
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
                max_shape: None,
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
