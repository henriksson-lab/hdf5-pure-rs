use hdf5_pure_rust::File;

#[test]
fn test_read_fixed_strings() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("fixed_str").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_string());
    assert_eq!(dtype.size(), 10);

    let strings = ds.read_strings().unwrap();
    assert_eq!(strings.len(), 2);
    assert_eq!(strings[0], "hello");
    assert_eq!(strings[1], "world");
}

#[test]
fn test_read_vlen_string_attr() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let attr = f.attr("vlen_str").unwrap();

    // Variable-length string attributes store data inline (via global heap ref)
    // For now, just verify we can read the raw data
    println!(
        "vlen_str attr raw data ({} bytes): {:?}",
        attr.raw_data().len(),
        attr.raw_data()
    );
}

#[test]
fn test_read_vlen_string_dataset() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("vlen_ds").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_vlen());

    let strings = ds.read_strings().unwrap();
    println!("vlen strings: {strings:?}");
    assert_eq!(strings.len(), 3);
    assert_eq!(strings[0], "alpha");
    assert_eq!(strings[1], "beta");
    assert_eq!(strings[2], "gamma");
}
