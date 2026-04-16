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

    /// Return the parsed low-level datatype message.
    pub fn raw_message(&self) -> DatatypeMessage {
        self.msg.clone()
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

    /// String character set for HDF5 string datatypes: 0=ASCII, 1=UTF-8.
    pub fn char_set(&self) -> Option<u8> {
        self.msg.char_set()
    }

    /// Fixed-length string padding type: 0=null-terminated, 1=null-padded, 2=space-padded.
    pub fn string_padding(&self) -> Option<u8> {
        self.msg.string_padding()
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

    /// Opaque datatype tag, if this is an opaque datatype.
    pub fn opaque_tag(&self) -> Option<String> {
        self.msg.opaque_tag()
    }

    /// Reference datatype kind: 0=object reference, 1=dataset region reference.
    pub fn reference_type(&self) -> Option<u8> {
        self.msg.reference_type()
    }

    /// Get compound type fields (returns None if not compound).
    pub fn compound_fields(&self) -> Option<Vec<CompoundField>> {
        self.msg.compound_fields().ok()
    }

    /// Get the number of compound members.
    pub fn compound_nmembers(&self) -> Option<u16> {
        self.msg.compound_nmembers()
    }

    /// Get enum members as (name, value) pairs.
    pub fn enum_members(&self) -> Option<Vec<(String, u64)>> {
        self.msg.enum_members().ok()
    }

    /// Get array dimensions and base datatype for array types.
    pub fn array_dims_base(&self) -> Option<(Vec<u64>, Datatype)> {
        self.msg
            .array_dims_base()
            .ok()
            .map(|(dims, base)| (dims, Datatype::from_message(base)))
    }

    /// Get the base datatype for variable-length sequence/string types.
    pub fn vlen_base(&self) -> Option<Datatype> {
        self.msg
            .vlen_base()
            .ok()
            .flatten()
            .map(Datatype::from_message)
    }
}
