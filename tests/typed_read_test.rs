use hdf5_pure_rust::File;

#[test]
fn test_read_typed_f64() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let values: Vec<f64> = ds.read::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_typed_i32() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();
    let values: Vec<i32> = ds.read::<i32>().unwrap();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_read_scalar_typed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();
    let val: f64 = ds.read_scalar::<f64>().unwrap();
    assert_eq!(val, 42.0);
}

#[test]
fn test_read_1d_ndarray() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let arr = ds.read_1d::<f64>().unwrap();
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0], 1.0);
    assert_eq!(arr[4], 5.0);
}

#[test]
fn test_read_2d_ndarray() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let arr = ds.read_2d::<i8>().unwrap();
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr[[0, 0]], 1);
    assert_eq!(arr[[1, 2]], 6);
}

#[test]
fn test_read_dyn_ndarray_3d() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("read_dyn_3d.h5");

    {
        let mut wf = hdf5_pure_rust::WritableFile::create(&path).unwrap();
        let data: Vec<i32> = (0..24).collect();
        wf.new_dataset_builder("cube")
            .shape(&[2, 3, 4])
            .write::<i32>(&data)
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let arr = f.dataset("cube").unwrap().read_dyn::<i32>().unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4]);
    assert_eq!(arr[[0, 0, 0]], 0);
    assert_eq!(arr[[1, 2, 3]], 23);
}

#[test]
fn test_raw_message_inspection_apis() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();

    let dtype = ds.raw_datatype_message().unwrap();
    assert_eq!(dtype.size, 8);
    assert_eq!(ds.dtype().unwrap().raw_message().size, 8);

    let space = ds.raw_dataspace_message().unwrap();
    assert_eq!(space.dims, vec![5]);
    assert_eq!(ds.space().unwrap().raw_message().dims, vec![5]);

    let plist = ds.create_plist().unwrap();
    assert!(plist.filters.is_empty());
    assert_eq!(plist.external_count(), 0);
    assert!(plist.external(0).is_none());
    assert!(plist.filter(0).is_none());
    assert!(plist.filter_by_id(1).is_none());

    let access = ds.access_plist();
    assert_eq!(
        access.virtual_view(),
        hdf5_pure_rust::VdsView::LastAvailable
    );
    assert_eq!(access.virtual_prefix(), None);
    assert_eq!(
        access.virtual_missing_source_policy(),
        hdf5_pure_rust::VdsMissingSourcePolicy::Error
    );
}

#[test]
fn test_read_chunked_typed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    let plist = ds.create_plist().unwrap();
    assert!(plist.chunk_opts().is_none() || plist.chunk_opts() == Some(0));

    let values: Vec<f32> = ds.read::<f32>().unwrap();
    assert_eq!(values.len(), 100);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as f32);
    }
}

#[test]
fn test_attr_read_typed() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val: i64 = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 42);
    let narrowed: i8 = attr.read_scalar::<i8>().unwrap();
    assert_eq!(narrowed, 42);
    let as_float: f64 = attr.read_scalar::<f64>().unwrap();
    assert_eq!(as_float, 42.0);
}

#[test]
fn test_attr_read_array_typed() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("array_attr").unwrap();
    let values: Vec<f64> = attr.read::<f64>().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
    let values32: Vec<f32> = attr.read::<f32>().unwrap();
    assert_eq!(values32, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_read_wrong_type_size() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("fixed_str").unwrap();
    // Fixed strings are not part of the numeric conversion table.
    let result = ds.read::<u64>();
    assert!(result.is_err());
}
