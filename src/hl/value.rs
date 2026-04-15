/// Recursive high-level value decoded from an HDF5 datatype.
#[derive(Debug, Clone, PartialEq)]
pub enum H5Value {
    Int(i128),
    UInt(u128),
    Float(f64),
    String(String),
    Compound(Vec<(String, H5Value)>),
    Array(Vec<H5Value>),
    VarLen(Vec<H5Value>),
    Reference(u64),
    Raw(Vec<u8>),
    Empty,
}
