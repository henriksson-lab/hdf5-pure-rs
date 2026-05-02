use hdf5_pure_rust::format::messages::datatype::DatatypeClass;
use hdf5_pure_rust::{File, Location};

#[test]
fn test_list_root_attrs_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let names = f.attr_names().unwrap();
    let attrs = f.attrs().unwrap();
    println!("v0 root attrs: {names:?}");

    assert!(names.contains(&"string_attr".to_string()));
    assert!(names.contains(&"int_attr".to_string()));
    assert!(names.contains(&"float_attr".to_string()));
    assert!(names.contains(&"array_attr".to_string()));
    assert_eq!(attrs.len(), names.len());
    assert!(attrs.iter().any(|attr| attr.name() == "int_attr"));
    assert_eq!(f.attr_count().unwrap(), names.len());
    let first_name = f.attr_name_by_idx(0).unwrap();
    assert_eq!(first_name, names[0]);
    assert_eq!(
        f.attr_info_by_idx(0).unwrap(),
        f.attr(&first_name).unwrap().info()
    );
    assert!(f.attr_name_by_idx(names.len()).is_err());
    assert!(f.attr_info_by_idx(names.len()).is_err());
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
    assert!(attr.dtype().is_float());
    assert_eq!(attr.dtype().class(), DatatypeClass::FloatingPoint);
    assert_eq!(attr.raw_datatype_message().size, 8);
    assert!(attr.space().is_simple());
    assert_eq!(attr.space().shape(), &[3]);
    assert_eq!(attr.raw_dataspace_message().dims, vec![3]);
    let info = attr.info();
    assert!(!info.creation_order_valid);
    assert_eq!(info.creation_order, 0);
    assert_eq!(info.char_encoding, 0);
    assert_eq!(info.data_size, 24);
    assert_eq!(attr.create_plist().char_encoding(), info.char_encoding);

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
    let attrs = ds.attrs().unwrap();
    let attr = ds.attr("ds_attr").unwrap();
    let val = attr.read_scalar_i64().unwrap();
    assert_eq!(ds.attr_count().unwrap(), 1);
    assert_eq!(ds.attr_name_by_idx(0).unwrap(), "ds_attr");
    assert_eq!(ds.attr_info_by_idx(0).unwrap(), attr.info());
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name(), "ds_attr");
    assert_eq!(val, 100);
}

#[test]
fn test_attr_exists_on_file_group_and_dataset() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    assert!(f.attr_exists("int_attr").unwrap());
    assert!(!f.attr_exists("missing_attr").unwrap());

    let ds = f.dataset("data").unwrap();
    assert!(ds.attr_exists("ds_attr").unwrap());
    assert!(!ds.attr_exists("missing_attr").unwrap());

    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let group = f.group("group1").unwrap();
    assert!(!group.attr_exists("missing_attr").unwrap());
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
    let tracked_attrs = tracked.attrs().unwrap();
    assert_eq!(tracked_names.len(), 32);
    assert_eq!(tracked_attrs.len(), 32);
    assert!(tracked_names.contains(&"attr_00".to_string()));
    assert!(tracked_names.contains(&"attr_31".to_string()));
    assert!(tracked_attrs.iter().any(|attr| attr.name() == "attr_07"));
    let tracked_by_order = tracked.attrs_by_creation_order().unwrap();
    assert_eq!(tracked_by_order.len(), 32);
    assert_eq!(
        tracked_by_order
            .iter()
            .map(|attr| attr.creation_order().unwrap())
            .collect::<Vec<_>>(),
        (0..32).collect::<Vec<_>>()
    );
    assert_eq!(
        tracked.attr("attr_07").unwrap().read::<i32>().unwrap(),
        vec![7, 107]
    );

    let untracked = f.group("dense_attrs_untracked").unwrap();
    let untracked_names = untracked.attr_names().unwrap();
    assert_eq!(untracked_names.len(), 32);
    assert!(untracked_names.contains(&"attr_00".to_string()));
    assert!(untracked_names.contains(&"attr_31".to_string()));
    assert_eq!(untracked.attrs_by_creation_order().unwrap().len(), 32);
    assert_eq!(
        untracked.attr("attr_07").unwrap().read::<i32>().unwrap(),
        vec![7, 207]
    );

    let old_file = File::open("tests/data/attrs.h5").unwrap();
    assert!(old_file.attrs_by_creation_order().is_err());
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
    assert_eq!(attr.read_strings().unwrap(), vec!["", "alpha", "猫"]);
}
