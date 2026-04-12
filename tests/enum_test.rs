use hdf5_pure_rust::File;

#[test]
fn test_enum_dtype_info() {
    let f = File::open("tests/data/enum.h5").unwrap();
    let ds = f.dataset("colors").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_enum());
    assert_eq!(dtype.size(), 1); // u8 base type

    let members = dtype.enum_members().unwrap();
    println!("Enum members: {members:?}");
    assert_eq!(members.len(), 3);

    // Find members by name
    let red = members.iter().find(|(n, _)| n == "RED").unwrap();
    let green = members.iter().find(|(n, _)| n == "GREEN").unwrap();
    let blue = members.iter().find(|(n, _)| n == "BLUE").unwrap();
    assert_eq!(red.1, 0);
    assert_eq!(green.1, 1);
    assert_eq!(blue.1, 2);
}

#[test]
fn test_enum_read_raw_values() {
    let f = File::open("tests/data/enum.h5").unwrap();
    let ds = f.dataset("colors").unwrap();

    let values: Vec<u8> = ds.read::<u8>().unwrap();
    assert_eq!(values, vec![0, 1, 2, 1]); // RED, GREEN, BLUE, GREEN
}
