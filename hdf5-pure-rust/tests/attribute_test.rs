use hdf5_pure_rust::File;

#[test]
fn test_list_root_attrs_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let names = f.attr_names().unwrap();
    println!("v0 root attrs: {names:?}");

    assert!(names.contains(&"string_attr".to_string()));
    assert!(names.contains(&"int_attr".to_string()));
    assert!(names.contains(&"float_attr".to_string()));
    assert!(names.contains(&"array_attr".to_string()));
}

#[test]
fn test_read_int_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val = attr.read_scalar_i64().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn test_read_float_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("float_attr").unwrap();
    let val = attr.read_scalar_f64().unwrap();
    assert!((val - 3.14).abs() < 1e-10);
}

#[test]
fn test_read_array_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("array_attr").unwrap();
    assert_eq!(attr.shape(), &[3]);
    assert_eq!(attr.element_size(), 8);

    let data = attr.raw_data();
    let values: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dataset_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let ds = f.dataset("data").unwrap();
    let attr = ds.attr("ds_attr").unwrap();
    let val = attr.read_scalar_i64().unwrap();
    assert_eq!(val, 100);
}

#[test]
fn test_list_root_attrs_v3() {
    let f = File::open("tests/data/attrs_v3.h5").unwrap();
    let names = f.attr_names().unwrap();
    println!("v3 root attrs: {names:?}");

    assert!(names.contains(&"string_attr".to_string()));
    assert!(names.contains(&"int_attr".to_string()));
    assert!(names.contains(&"float_attr".to_string()));
}

#[test]
fn test_read_int_attr_v3() {
    let f = File::open("tests/data/attrs_v3.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val = attr.read_scalar_i64().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn test_dataset_attr_v3() {
    let f = File::open("tests/data/attrs_v3.h5").unwrap();
    let ds = f.dataset("data").unwrap();
    let attr = ds.attr("ds_attr").unwrap();
    let val = attr.read_scalar_i64().unwrap();
    assert_eq!(val, 100);
}
