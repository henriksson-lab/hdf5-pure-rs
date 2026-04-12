//! Phase T10: Write and round-trip tests.
//! Write with pure-Rust, verify with our reader, h5dump, and h5py.

use hdf5_pure_rust::{File, WritableFile, MutableFile};

fn tmp(name: &str) -> String { format!("tests/data/t10_{name}.h5") }

fn cleanup(path: &str) { std::fs::remove_file(path).ok(); }

// T10a: Write + h5dump verify -- all layout types

#[test]
fn t10a_contiguous_h5dump() {
    let p = tmp("contiguous");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("data").write::<f64>(&[1.0, 2.0, 3.0]).unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("h5dump").arg(&p).output();
    if let Ok(out) = out {
        assert!(out.status.success(), "h5dump: {}", String::from_utf8_lossy(&out.stderr));
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("1, 2, 3"));
    }
    cleanup(&p);
}

#[test]
fn t10a_compact_h5dump() {
    let p = tmp("compact");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("tiny").compact().write::<u8>(&[10, 20, 30]).unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("h5dump").arg(&p).output();
    if let Ok(out) = out {
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("10, 20, 30"));
    }
    cleanup(&p);
}

#[test]
fn t10a_chunked_h5dump() {
    let p = tmp("chunked");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        let data: Vec<i32> = (0..50).collect();
        wf.new_dataset_builder("chunked").shape(&[50]).chunk(&[10]).deflate(4)
            .write::<i32>(&data).unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("h5dump").arg("-d").arg("chunked").arg(&p).output();
    if let Ok(out) = out {
        assert!(out.status.success(), "h5dump: {}", String::from_utf8_lossy(&out.stderr));
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("49"));
    }
    cleanup(&p);
}

// T10b: Write + h5py verify

#[test]
fn t10b_h5py_verify() {
    let p = tmp("h5py_verify");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("values").write::<f64>(&[1.5, 2.5, 3.5]).unwrap();
        wf.add_attr("version", 1i64).unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import h5py; f=h5py.File('{}','r'); \
             assert list(f['values'][:])==[1.5,2.5,3.5]; \
             assert f.attrs['version']==1; \
             print('OK'); f.close()", p))
        .output();
    if let Ok(out) = out {
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("OK"), "h5py: {}", String::from_utf8_lossy(&out.stderr));
    }
    cleanup(&p);
}

// T10c: Write + read round-trip

#[test]
fn t10c_roundtrip_all_types() {
    let p = tmp("roundtrip_types");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("f64").write::<f64>(&[1.0, 2.0]).unwrap();
        wf.new_dataset_builder("f32").write::<f32>(&[3.0, 4.0]).unwrap();
        wf.new_dataset_builder("i32").write::<i32>(&[5, 6]).unwrap();
        wf.new_dataset_builder("i64").write::<i64>(&[7, 8]).unwrap();
        wf.new_dataset_builder("u8").write::<u8>(&[9, 10]).unwrap();
        wf.flush().unwrap();
    }
    {
        let f = File::open(&p).unwrap();
        assert_eq!(f.dataset("f64").unwrap().read::<f64>().unwrap(), vec![1.0, 2.0]);
        assert_eq!(f.dataset("f32").unwrap().read::<f32>().unwrap(), vec![3.0, 4.0]);
        assert_eq!(f.dataset("i32").unwrap().read::<i32>().unwrap(), vec![5, 6]);
        assert_eq!(f.dataset("i64").unwrap().read::<i64>().unwrap(), vec![7, 8]);
        assert_eq!(f.dataset("u8").unwrap().read::<u8>().unwrap(), vec![9, 10]);
    }
    cleanup(&p);
}

// T10d: Chunked write with all filter combos

#[test]
fn t10d_deflate_only() {
    let p = tmp("deflate_only");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        wf.new_dataset_builder("d").chunk(&[25]).deflate(6).write::<f32>(&data).unwrap();
        wf.flush().unwrap();
    }
    let f = File::open(&p).unwrap();
    let vals: Vec<f32> = f.dataset("d").unwrap().read::<f32>().unwrap();
    for (i, v) in vals.iter().enumerate() { assert_eq!(*v, i as f32); }
    cleanup(&p);
}

#[test]
fn t10d_shuffle_deflate() {
    let p = tmp("shuffle_deflate");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        let data: Vec<i32> = (0..100).collect();
        wf.new_dataset_builder("d").chunk(&[20]).shuffle().deflate(4).write::<i32>(&data).unwrap();
        wf.flush().unwrap();
    }
    let f = File::open(&p).unwrap();
    let vals: Vec<i32> = f.dataset("d").unwrap().read::<i32>().unwrap();
    for (i, v) in vals.iter().enumerate() { assert_eq!(*v, i as i32); }
    cleanup(&p);
}

// T10e: Attribute write round-trip

#[test]
fn t10e_scalar_attrs() {
    let p = tmp("scalar_attrs");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.add_attr("count", 42i64).unwrap();
        wf.add_attr("pi", std::f64::consts::PI).unwrap();
        wf.new_dataset_builder("x").write::<f64>(&[1.0]).unwrap();
        wf.flush().unwrap();
    }
    let f = File::open(&p).unwrap();
    assert_eq!(f.attr("count").unwrap().read_scalar::<i64>().unwrap(), 42);
    let pi = f.attr("pi").unwrap().read_scalar::<f64>().unwrap();
    assert!((pi - std::f64::consts::PI).abs() < 1e-15);
    cleanup(&p);
}

// T10f: MutableFile round-trip

#[test]
fn t10f_resize_roundtrip() {
    let p = tmp("resize_rt");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("d").shape(&[10]).chunk(&[5]).resizable()
            .write::<f64>(&[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]).unwrap();
        wf.flush().unwrap();
    }
    {
        let mut mf = MutableFile::open_rw(&p).unwrap();
        mf.resize_dataset("d", &[5]).unwrap();
    }
    {
        let f = File::open(&p).unwrap();
        let ds = f.dataset("d").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    cleanup(&p);
}
