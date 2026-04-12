use crate::format::messages::datatype::{ByteOrder, CompoundField, DatatypeClass, DatatypeMessage};

/// High-level datatype descriptor.
#[derive(Debug, Clone)]
pub struct Datatype {
    msg: DatatypeMessage,
}

impl Datatype {
    pub(crate) fn from_message(msg: DatatypeMessage) -> Self {
        Self { msg }
    }

    /// Total size of one element in bytes.
    pub fn size(&self) -> usize {
        self.msg.size as usize
    }

    /// Datatype class (FixedPoint, FloatingPoint, String, Compound, etc.).
    pub fn class(&self) -> DatatypeClass {
        self.msg.class
    }

    /// Byte order for numeric types.
    pub fn byte_order(&self) -> Option<ByteOrder> {
        self.msg.byte_order()
    }

    /// Whether a fixed-point type is signed.
    pub fn is_signed(&self) -> Option<bool> {
        self.msg.is_signed()
    }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        self.msg.class == DatatypeClass::FloatingPoint
    }

    /// Whether this is an integer type.
    pub fn is_integer(&self) -> bool {
        self.msg.class == DatatypeClass::FixedPoint
    }

    /// Whether this is a string type.
    pub fn is_string(&self) -> bool {
        self.msg.class == DatatypeClass::String
    }

    /// Whether this is a compound type.
    pub fn is_compound(&self) -> bool {
        self.msg.class == DatatypeClass::Compound
    }

    /// Whether this is an enum type.
    pub fn is_enum(&self) -> bool {
        self.msg.class == DatatypeClass::Enum
    }

    /// Whether this is a variable-length type.
    pub fn is_vlen(&self) -> bool {
        self.msg.class == DatatypeClass::VarLen
    }

    /// Get compound type fields (returns None if not compound).
    pub fn compound_fields(&self) -> Option<Vec<CompoundField>> {
        self.msg.compound_fields()
    }

    /// Get the number of compound members.
    pub fn compound_nmembers(&self) -> Option<u16> {
        self.msg.compound_nmembers()
    }

    /// Get enum members as (name, value) pairs.
    pub fn enum_members(&self) -> Option<Vec<(String, u64)>> {
        self.msg.enum_members()
    }
}
