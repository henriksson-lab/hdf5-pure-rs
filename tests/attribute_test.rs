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

#[test]
fn test_large_compact_attribute_read() {
    let f = File::open("tests/data/hdf5_ref/attribute_cases.h5").unwrap();
    let group = f.group("large_compact_attrs").unwrap();
    let attr = group.attr("large_i32").unwrap();
    let values: Vec<i32> = attr.read().unwrap();

    assert_eq!(attr.shape(), &[256]);
    assert_eq!(values.len(), 256);
    assert_eq!(values[0], 0);
    assert_eq!(values[255], 255);
}

#[test]
fn test_dense_attributes_creation_order_indexing_enabled_and_disabled() {
    let f = File::open("tests/data/hdf5_ref/attribute_cases.h5").unwrap();

    let tracked = f.group("dense_attrs_tracked").unwrap();
    let tracked_names = tracked.attr_names().unwrap();
    assert_eq!(tracked_names.len(), 32);
    assert!(tracked_names.contains(&"attr_00".to_string()));
    assert!(tracked_names.contains(&"attr_31".to_string()));
    assert_eq!(
        tracked.attr("attr_07").unwrap().read::<i32>().unwrap(),
        vec![7, 107]
    );

    let untracked = f.group("dense_attrs_untracked").unwrap();
    let untracked_names = untracked.attr_names().unwrap();
    assert_eq!(untracked_names.len(), 32);
    assert!(untracked_names.contains(&"attr_00".to_string()));
    assert!(untracked_names.contains(&"attr_31".to_string()));
    assert_eq!(
        untracked.attr("attr_07").unwrap().read::<i32>().unwrap(),
        vec![7, 207]
    );
}

#[test]
fn test_variable_length_attribute_payload_raw_read() {
    let f = File::open("tests/data/hdf5_ref/attribute_cases.h5").unwrap();
    let group = f.group("vlen_attr_holder").unwrap();
    let attr = group.attr("vlen_strings").unwrap();

    assert_eq!(attr.shape(), &[3]);
    assert_eq!(attr.element_size(), 16);
    assert_eq!(attr.raw_data().len(), 48);
    assert!(attr.raw_data().iter().any(|&b| b != 0));
}
