use hdf5_pure_rust::DeriveH5Type;
use hdf5_pure_rust::H5Type;

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Point {
    x: f64,
    y: f64,
    label: i32,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(u8)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Measurement {
    value: f32,
    #[hdf5(rename = "error_margin")]
    error: f32,
}

#[test]
fn test_derive_struct_size() {
    assert_eq!(Point::type_size(), std::mem::size_of::<Point>());
    // f64(8) + f64(8) + i32(4) + padding(4) = 24 on most platforms, or 20 with packed
    assert!(Point::type_size() >= 20);
}

#[test]
fn test_derive_struct_fields() {
    let fields = Point::compound_fields().unwrap();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[0].offset, 0);
    assert_eq!(fields[0].size, 8);
    assert_eq!(fields[1].name, "y");
    assert_eq!(fields[1].offset, 8);
    assert_eq!(fields[1].size, 8);
    assert_eq!(fields[2].name, "label");
    assert_eq!(fields[2].offset, 16);
    assert_eq!(fields[2].size, 4);
}

#[test]
fn test_derive_enum_size() {
    assert_eq!(Color::type_size(), 1);
}

#[test]
fn test_derive_enum_members() {
    let members = Color::enum_members().unwrap();
    assert_eq!(members.len(), 3);
    assert_eq!(members[0], ("Red".to_string(), 0));
    assert_eq!(members[1], ("Green".to_string(), 1));
    assert_eq!(members[2], ("Blue".to_string(), 2));
}

#[test]
fn test_derive_with_rename() {
    let fields = Measurement::compound_fields().unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "value");
    assert_eq!(fields[1].name, "error_margin");
}

#[test]
fn test_derive_struct_can_read() {
    // Verify the derived type works with read operations
    // (uses type_size for byte reinterpretation)
    let data = vec![
        1.0f64.to_le_bytes(),
        2.0f64.to_le_bytes(),
        42i32.to_le_bytes().to_vec().into_iter()
            .chain(std::iter::repeat(0).take(4)) // padding
            .collect::<Vec<u8>>()
            .try_into().unwrap(),
    ];
    let bytes: Vec<u8> = data.into_iter().flat_map(|b: [u8; 8]| b).collect();

    let points: Vec<Point> = hdf5_pure_rust::hl::types::bytes_to_vec::<Point>(bytes).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].x, 1.0);
    assert_eq!(points[0].y, 2.0);
    assert_eq!(points[0].label, 42);
}
