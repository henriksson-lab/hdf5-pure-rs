use hdf5_pure_rust::hl::types::{FieldDescriptor, H5Type, TypeClass};
use hdf5_pure_rust::WritableFile;

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    label: i32,
}

unsafe impl H5Type for Point {
    fn type_size() -> usize {
        std::mem::size_of::<Point>()
    }

    fn compound_fields() -> Option<Vec<FieldDescriptor>> {
        Some(vec![
            FieldDescriptor {
                name: "x".to_string(),
                offset: std::mem::offset_of!(Point, x),
                size: std::mem::size_of::<f64>(),
                type_class: TypeClass::Float,
            },
            FieldDescriptor {
                name: "label".to_string(),
                offset: std::mem::offset_of!(Point, label),
                size: std::mem::size_of::<i32>(),
                type_class: TypeClass::Integer { signed: true },
            },
        ])
    }
}

#[test]
fn test_writable_file_simple() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_simple.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("temperatures")
            .shape(&[5])
            .write::<f64>(&[20.0, 21.5, 22.0, 19.8, 23.1])
            .unwrap();
        let f = wf.close().unwrap();
        let ds = f.dataset("temperatures").unwrap();
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals, vec![20.0, 21.5, 22.0, 19.8, 23.1]);
    }
}

#[test]
fn test_writable_file_with_groups() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_groups.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
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
}

#[test]
fn test_writable_file_group_with_many_compact_links() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_many_links.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut g = wf.create_group("many").unwrap();
        for idx in 0..40 {
            let name = format!("value_{idx:02}");
            g.new_dataset_builder(&name).write::<i32>(&[idx]).unwrap();
        }
        let f = wf.close().unwrap();

        let group = f.group("many").unwrap();
        let names = group.member_names().unwrap();
        assert_eq!(names.len(), 40);
        assert!(names.contains(&"value_00".to_string()));
        assert!(names.contains(&"value_39".to_string()));
        assert_eq!(
            group
                .open_dataset("value_37")
                .unwrap()
                .read::<i32>()
                .unwrap(),
            vec![37]
        );
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump failed on many-link writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_writable_file_chunked_compressed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
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
}

#[test]
fn test_writable_file_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_scalar.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("pi")
            .write_scalar::<f64>(std::f64::consts::PI)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("pi").unwrap();
        let val: f64 = ds.read_scalar::<f64>().unwrap();
        assert!((val - std::f64::consts::PI).abs() < 1e-15);
    }
}

#[test]
fn test_writable_file_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("tiny")
            .compact()
            .write::<u8>(&[1, 2, 3, 4, 5])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("tiny").unwrap();
        let vals: Vec<u8> = ds.read::<u8>().unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }
}

#[test]
fn test_writable_file_explicit_fill_value_properties() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_fill_value.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("with_fill")
            .fill_properties(1, 2)
            .fill_value::<i32>(-7)
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("with_fill").unwrap();
        let plist = ds.create_plist().unwrap();
        assert_eq!(plist.fill_alloc_time, Some(1));
        assert_eq!(plist.fill_time, Some(2));
        assert!(plist.fill_value_defined);
        assert_eq!(plist.fill_value, Some((-7i32).to_le_bytes().to_vec()));
        assert_eq!(ds.read::<i32>().unwrap(), vec![1, 2, 3]);
    }
}

#[test]
fn test_writable_file_compact_fixed_strings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_strings.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .compact()
            .write_fixed_ascii_strings(&["red", "green", "blue"], 8)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert_eq!(
            ds.read_strings().unwrap(),
            vec!["red".to_string(), "green".to_string(), "blue".to_string()]
        );
    }
}

#[test]
fn test_writable_file_vlen_utf8_strings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_vlen_strings.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .write_vlen_utf8_strings(&["", "猫", "alpha"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert_eq!(
            ds.read_strings().unwrap(),
            vec!["".to_string(), "猫".to_string(), "alpha".to_string()]
        );
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump failed on vlen string writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_writable_file_compact_compound() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_compound.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("points")
            .compact()
            .write::<Point>(&[Point { x: 1.5, label: 10 }, Point { x: 2.5, label: 20 }])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("points").unwrap();
        assert_eq!(ds.read_field::<f64>("x").unwrap(), vec![1.5, 2.5]);
        assert_eq!(ds.read_field::<i32>("label").unwrap(), vec![10, 20]);
    }
}

#[test]
fn test_writable_file_h5dump_interop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_h5dump.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    let out = std::process::Command::new("h5dump").arg(&path).output();
    if let Ok(out) = out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "h5dump failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(stdout.contains("1, 2, 3"));
    }
}

#[test]
fn test_writable_file_root_attr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_attr.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_attr("version", 42i64).unwrap();
        wf.new_dataset_builder("data")
            .write::<f32>(&[1.0, 2.0])
            .unwrap();
        let f = wf.close().unwrap();

        let attr = f.attr("version").unwrap();
        let val: i64 = attr.read_scalar::<i64>().unwrap();
        assert_eq!(val, 42);
    }
}
