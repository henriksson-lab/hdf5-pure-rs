use hdf5_pure_rust::File;

#[test]
fn test_dataset_create_plist_contiguous() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let plist = ds.create_plist().unwrap();

    assert!(!plist.is_chunked());
    assert!(!plist.is_compressed());
    assert!(!plist.has_shuffle());
    assert_eq!(plist.deflate_level(), None);
}

#[test]
fn test_dataset_create_plist_chunked_compressed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    let plist = ds.create_plist().unwrap();

    assert!(plist.is_chunked());
    assert!(plist.is_compressed());
    assert_eq!(plist.deflate_level(), Some(1));
    assert!(plist.chunk_dims.is_some());
    assert_eq!(plist.chunk_dims.as_ref().unwrap(), &[10]);
}

#[test]
fn test_file_create_plist() {
    use hdf5_pure_rust::hl::plist::file_create::FileCreate;

    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let plist = FileCreate::from_file(&f);
    assert_eq!(plist.superblock_version, 0);
    assert_eq!(plist.sizeof_addr, 8);
    assert_eq!(plist.sizeof_size, 8);
    assert_eq!(plist.sym_leaf_k, 4);
    assert_eq!(plist.btree_k, 16);
}

#[test]
fn test_dataset_metadata_queries() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();

    let ds = f.dataset("chunked").unwrap();
    assert!(ds.is_chunked().unwrap());
    assert_eq!(ds.chunk().unwrap(), Some(vec![10]));

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_float());
    assert_eq!(dtype.size(), 4);

    let space = ds.space().unwrap();
    assert!(space.is_simple());
    assert_eq!(space.ndim(), 1);
    assert_eq!(space.shape(), &[100]);
    assert!(!space.is_resizable());
}
