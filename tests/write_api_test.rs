use hdf5_pure_rust::WritableFile;

#[test]
fn test_writable_file_simple() {
    let path = "tests/data/api_write_simple.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        wf.new_dataset_builder("temperatures")
            .shape(&[5])
            .write::<f64>(&[20.0, 21.5, 22.0, 19.8, 23.1])
            .unwrap();
        let f = wf.close().unwrap();
        let ds = f.dataset("temperatures").unwrap();
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals, vec![20.0, 21.5, 22.0, 19.8, 23.1]);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_writable_file_with_groups() {
    let path = "tests/data/api_write_groups.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        let mut g = wf.create_group("sensors").unwrap();
        g.new_dataset_builder("pressure")
            .write::<f32>(&[1013.25, 1012.0, 1011.5])
            .unwrap();
        let f = wf.close().unwrap();

        let names = f.member_names().unwrap();
        assert!(names.contains(&"sensors".to_string()));

        let g = f.group("sensors").unwrap();
        let ds = g.open_dataset("pressure").unwrap();
        let vals: Vec<f32> = ds.read::<f32>().unwrap();
        assert_eq!(vals, vec![1013.25, 1012.0, 1011.5]);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_writable_file_chunked_compressed() {
    let path = "tests/data/api_write_chunked.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        let data: Vec<i32> = (0..100).collect();
        wf.new_dataset_builder("data")
            .shape(&[100])
            .chunk(&[25])
            .deflate(4)
            .shuffle()
            .write::<i32>(&data)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        let vals: Vec<i32> = ds.read::<i32>().unwrap();
        assert_eq!(vals.len(), 100);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(*v, i as i32);
        }
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_writable_file_scalar() {
    let path = "tests/data/api_write_scalar.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        wf.new_dataset_builder("pi")
            .write_scalar::<f64>(std::f64::consts::PI)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("pi").unwrap();
        let val: f64 = ds.read_scalar::<f64>().unwrap();
        assert!((val - std::f64::consts::PI).abs() < 1e-15);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_writable_file_compact() {
    let path = "tests/data/api_write_compact.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        wf.new_dataset_builder("tiny")
            .compact()
            .write::<u8>(&[1, 2, 3, 4, 5])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("tiny").unwrap();
        let vals: Vec<u8> = ds.read::<u8>().unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_writable_file_h5dump_interop() {
    let path = "tests/data/api_write_h5dump.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        wf.new_dataset_builder("values")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    let out = std::process::Command::new("h5dump").arg(path).output();
    if let Ok(out) = out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "h5dump failed: {}", String::from_utf8_lossy(&out.stderr));
        assert!(stdout.contains("1, 2, 3"));
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn test_writable_file_root_attr() {
    let path = "tests/data/api_write_attr.h5";

    {
        let mut wf = WritableFile::create(path).unwrap();
        wf.add_attr("version", 42i64).unwrap();
        wf.new_dataset_builder("data")
            .write::<f32>(&[1.0, 2.0])
            .unwrap();
        let f = wf.close().unwrap();

        let attr = f.attr("version").unwrap();
        let val: i64 = attr.read_scalar::<i64>().unwrap();
        assert_eq!(val, 42);
    }
    std::fs::remove_file(path).ok();
}
