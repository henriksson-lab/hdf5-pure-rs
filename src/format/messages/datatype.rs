use crate::error::{Error, Result};

/// A field in a compound datatype.
#[derive(Debug, Clone)]
pub struct CompoundField {
    pub name: String,
    pub byte_offset: usize,
    pub size: usize,
    pub class: DatatypeClass,
    pub byte_order: Option<ByteOrder>,
    pub datatype: Box<DatatypeMessage>,
}

/// HDF5 datatype class values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatatypeClass {
    FixedPoint,    // 0 - integers
    FloatingPoint, // 1
    Time,          // 2
    String,        // 3
    BitField,      // 4
    Opaque,        // 5
    Compound,      // 6
    Reference,     // 7
    Enum,          // 8
    VarLen,        // 9
    Array,         // 10
}

impl DatatypeClass {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::FixedPoint),
            1 => Ok(Self::FloatingPoint),
            2 => Ok(Self::Time),
            3 => Ok(Self::String),
            4 => Ok(Self::BitField),
            5 => Ok(Self::Opaque),
            6 => Ok(Self::Compound),
            7 => Ok(Self::Reference),
            8 => Ok(Self::Enum),
            9 => Ok(Self::VarLen),
            10 => Ok(Self::Array),
            _ => Err(Error::InvalidFormat(format!(
                "unknown datatype class {val}"
            ))),
        }
    }
}

/// Byte order for numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// Parsed Datatype message (type 0x0003).
/// This is a partial parse -- full type info depends on class.
#[derive(Debug, Clone)]
pub struct DatatypeMessage {
    /// Class + version packed byte: version in top 4 bits, class in bottom 4.
    pub version: u8,
    pub class: DatatypeClass,
    /// Class-specific bit fields (3 bytes).
    pub class_bits: [u8; 3],
    /// Total size of the datatype in bytes.
    pub size: u32,
    /// Raw class-specific properties (variable length).
    pub properties: Vec<u8>,
}

impl DatatypeMessage {
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::InvalidFormat("datatype message too short".into()));
        }

        let class_and_version = data[0];
        let version = (class_and_version >> 4) & 0x0F;
        let class_val = class_and_version & 0x0F;
        let class = DatatypeClass::from_u8(class_val)?;

        let class_bits = [data[1], data[2], data[3]];
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        let properties = if data.len() > 8 {
            data[8..].to_vec()
        } else {
            Vec::new()
        };

        match class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField if properties.len() < 4 => {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            DatatypeClass::FloatingPoint if properties.len() < 12 => {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            _ => {}
        }

        let message = Self {
            version,
            class,
            class_bits,
            size,
            properties,
        };

        #[cfg(feature = "tracehash")]
        {
            let mut th = tracehash::th_call!("hdf5.datatype.decode");
            th.input_bytes(data);
            th.output_bool(true);
            th.output_u64(class_val as u64);
            th.finish();
        }

        Ok(message)
    }

    /// Get byte order for numeric types.
    pub fn byte_order(&self) -> Option<ByteOrder> {
        match self.class {
            DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField => {
                if self.class_bits[0] & 0x01 == 0 {
                    Some(ByteOrder::LittleEndian)
                } else {
                    Some(ByteOrder::BigEndian)
                }
            }
            _ => None,
        }
    }

    /// Whether a fixed-point type is signed.
    pub fn is_signed(&self) -> Option<bool> {
        if self.class == DatatypeClass::FixedPoint {
            Some(self.class_bits[0] & 0x08 != 0)
        } else {
            None
        }
    }

    /// Whether this is a fixed-length string type.
    pub fn is_fixed_string(&self) -> bool {
        self.class == DatatypeClass::String
    }

    /// Whether this is a variable-length type (including vlen strings).
    pub fn is_variable_length(&self) -> bool {
        self.class == DatatypeClass::VarLen
    }

    /// Get the number of members for compound types.
    pub fn compound_nmembers(&self) -> Option<u16> {
        if self.class == DatatypeClass::Compound {
            Some(self.class_bits[0] as u16 | ((self.class_bits[1] as u16) << 8))
        } else {
            None
        }
    }

    /// Parse compound type member fields.
    /// Returns field names, byte offsets, member sizes, and member datatypes.
    pub fn compound_fields(&self) -> Result<Vec<CompoundField>> {
        let nmembers = self
            .compound_nmembers()
            .ok_or_else(|| Error::InvalidFormat("not a compound datatype".into()))?
            as usize;
        let mut fields = Vec::with_capacity(nmembers);
        let data = &self.properties;
        let mut p = 0;

        for _ in 0..nmembers {
            if p >= data.len() {
                return Err(Error::InvalidFormat(
                    "compound datatype truncated before member".into(),
                ));
            }

            // Name (null-terminated, padded to 8-byte boundary in v1/v2)
            let name_start = p;
            let name_end = data[p..].iter().position(|&b| b == 0).ok_or_else(|| {
                Error::InvalidFormat("compound datatype member name is not terminated".into())
            })?;
            let name = String::from_utf8_lossy(&data[p..p + name_end]).to_string();
            if self.version < 3 {
                let name_with_null = name_end + 1;
                let padded = (name_with_null + 7) & !7;
                p = name_start + padded;
            } else {
                p += name_end + 1;
            }

            // Byte offset. Version 4 stores this using the minimum number of
            // bytes needed for offsets in the parent compound datatype.
            let offset_size = compound_member_offset_size(self.version, self.size as usize);
            if p + offset_size > data.len() {
                return Err(Error::InvalidFormat(
                    "compound datatype member offset is truncated".into(),
                ));
            }
            let byte_offset = read_le_var_usize(&data[p..p + offset_size]);
            p += offset_size;

            // Version 1/2: skip dimension info (1+3+4+4+16 = 28 bytes)
            if self.version < 3 {
                if p + 28 > data.len() {
                    return Err(Error::InvalidFormat(
                        "compound datatype member dimension block is truncated".into(),
                    ));
                }
                p += 28;
            }

            // Embedded member datatype
            if p + 8 > data.len() {
                return Err(Error::InvalidFormat(
                    "compound datatype member datatype is truncated".into(),
                ));
            }
            let encoded_len = datatype_encoded_len(&data[p..]).ok_or_else(|| {
                Error::InvalidFormat("compound datatype member datatype is malformed".into())
            })?;
            let member_dt = DatatypeMessage::decode(&data[p..p + encoded_len])?;
            let member_type_size = member_dt.size as usize;
            let member_class = member_dt.class;
            let byte_order = member_dt.byte_order();
            p += encoded_len;

            fields.push(CompoundField {
                name,
                byte_offset,
                size: member_type_size,
                class: member_class,
                byte_order,
                datatype: Box::new(member_dt),
            });
        }

        Ok(fields)
    }

    /// Get the number of enum members.
    pub fn enum_nmembers(&self) -> Option<u16> {
        if self.class == DatatypeClass::Enum {
            Some(self.class_bits[0] as u16 | ((self.class_bits[1] as u16) << 8))
        } else {
            None
        }
    }

    /// Parse enum type members. Returns Vec of (name, integer_value).
    pub fn enum_members(&self) -> Result<Vec<(String, u64)>> {
        let nmembers = self
            .enum_nmembers()
            .ok_or_else(|| Error::InvalidFormat("not an enum datatype".into()))?
            as usize;
        let data = &self.properties;
        if data.len() < 8 {
            return Err(Error::InvalidFormat(
                "enum datatype base datatype is truncated".into(),
            ));
        }

        // Base type (embedded datatype)
        let base_len = datatype_encoded_len(data).ok_or_else(|| {
            Error::InvalidFormat("enum datatype base datatype is malformed".into())
        })?;
        let base_dt = DatatypeMessage::decode(&data[..base_len])?;
        let base_size = base_dt.size as usize;
        let mut p = base_len;

        // Member names (null-terminated, padded to 8 in v1/v2)
        let mut names = Vec::with_capacity(nmembers);
        for _ in 0..nmembers {
            if p >= data.len() {
                return Err(Error::InvalidFormat(
                    "enum datatype member name is truncated".into(),
                ));
            }
            let name_end = data[p..].iter().position(|&b| b == 0).ok_or_else(|| {
                Error::InvalidFormat("enum datatype member name is not terminated".into())
            })?;
            let name = String::from_utf8_lossy(&data[p..p + name_end]).to_string();
            if self.version < 3 {
                let padded = (name_end + 1 + 7) & !7;
                if p + padded > data.len() {
                    return Err(Error::InvalidFormat(
                        "enum datatype member name padding is truncated".into(),
                    ));
                }
                p += padded;
            } else {
                p += name_end + 1;
            }
            names.push(name);
        }

        // Member values (each base_size bytes)
        let mut members = Vec::with_capacity(nmembers);
        for name in names {
            if p + base_size > data.len() {
                return Err(Error::InvalidFormat(
                    "enum datatype member value is truncated".into(),
                ));
            }
            let mut val = 0u64;
            for i in 0..base_size.min(8) {
                val |= (data[p + i] as u64) << (i * 8);
            }
            p += base_size;
            members.push((name, val));
        }

        Ok(members)
    }

    /// Get the character set for string types (0=ASCII, 1=UTF-8).
    pub fn char_set(&self) -> Option<u8> {
        if self.class == DatatypeClass::String {
            // Bit field byte 0, bits 0-3: padding type; byte 1, bits 0-3: char set
            Some(self.class_bits[1] & 0x0F)
        } else {
            None
        }
    }

    /// Get array dimensions and base datatype for array datatypes.
    pub fn array_dims_base(&self) -> Result<(Vec<u64>, DatatypeMessage)> {
        if self.class != DatatypeClass::Array {
            return Err(Error::InvalidFormat("not an array datatype".into()));
        }
        if self.properties.is_empty() {
            return Err(Error::InvalidFormat(
                "array datatype properties are truncated".into(),
            ));
        }
        let ndims = self.properties[0] as usize;
        let mut p = if self.version >= 4 { 1usize } else { 4usize };
        if self.properties.len() < p {
            return Err(Error::InvalidFormat(
                "array datatype header is truncated".into(),
            ));
        }
        let dims_len = ndims.checked_mul(4).ok_or_else(|| {
            Error::InvalidFormat("array datatype dimension table overflow".into())
        })?;
        let dims_end = p.checked_add(dims_len).ok_or_else(|| {
            Error::InvalidFormat("array datatype dimension table overflow".into())
        })?;
        if self.properties.len() < dims_end {
            return Err(Error::InvalidFormat(
                "array datatype dimension table is truncated".into(),
            ));
        }

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let dim = u32::from_le_bytes([
                self.properties[p],
                self.properties[p + 1],
                self.properties[p + 2],
                self.properties[p + 3],
            ]);
            dims.push(dim as u64);
            p += 4;
        }

        if p >= self.properties.len() {
            return Err(Error::InvalidFormat(
                "array datatype base datatype is missing".into(),
            ));
        }
        let base = DatatypeMessage::decode(&self.properties[p..])?;
        datatype_encoded_len(&self.properties[p..]).ok_or_else(|| {
            Error::InvalidFormat("array datatype base datatype is malformed".into())
        })?;
        Ok((dims, base))
    }

    /// Get the base datatype for variable-length sequence/string datatypes.
    pub fn vlen_base(&self) -> Result<Option<DatatypeMessage>> {
        if self.class != DatatypeClass::VarLen {
            return Err(Error::InvalidFormat(
                "not a variable-length datatype".into(),
            ));
        }
        if self.properties.is_empty() {
            return Err(Error::InvalidFormat(
                "variable-length datatype properties are truncated".into(),
            ));
        }

        if let Some(base_len) = datatype_encoded_len(&self.properties) {
            if base_len == self.properties.len() {
                return DatatypeMessage::decode(&self.properties).map(Some);
            }
        }

        if self.properties.len() < 4 {
            return Err(Error::InvalidFormat(
                "variable-length datatype metadata is truncated".into(),
            ));
        }
        if self.properties.len() == 4 {
            return Ok(None);
        }

        let base = &self.properties[4..];
        let base_len = datatype_encoded_len(base).ok_or_else(|| {
            Error::InvalidFormat("variable-length datatype base datatype is malformed".into())
        })?;
        if base_len != base.len() {
            return Err(Error::InvalidFormat(
                "variable-length datatype base datatype has trailing bytes".into(),
            ));
        }
        DatatypeMessage::decode(base).map(Some)
    }
}

fn datatype_encoded_len(data: &[u8]) -> Option<usize> {
    if data.len() < 8 {
        return None;
    }

    let class_and_version = data[0];
    let version = (class_and_version >> 4) & 0x0F;
    let class_val = class_and_version & 0x0F;
    let class = DatatypeClass::from_u8(class_val).ok()?;
    let class_bits = [data[1], data[2], data[3]];
    let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

    let prop_len = match class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField => 4,
        DatatypeClass::FloatingPoint => 12,
        DatatypeClass::Time | DatatypeClass::String | DatatypeClass::Reference => 0,
        DatatypeClass::Opaque => data[8..]
            .iter()
            .position(|&b| b == 0)
            .map(|n| n + 1)
            .unwrap_or(0),
        DatatypeClass::Enum => {
            let base_len = datatype_encoded_len(&data[8..])?;
            let base = DatatypeMessage::decode(&data[8..8 + base_len]).ok()?;
            let nmembers = class_bits[0] as usize | ((class_bits[1] as usize) << 8);
            let mut p = 8 + base_len;

            for _ in 0..nmembers {
                let name_len = data[p..].iter().position(|&b| b == 0)? + 1;
                p += if version < 3 {
                    (name_len + 7) & !7
                } else {
                    name_len
                };
            }

            p += nmembers.checked_mul(base.size as usize)?;
            return (p <= data.len()).then_some(p);
        }
        DatatypeClass::Compound => {
            let msg = DatatypeMessage::decode(data).ok()?;
            let nmembers = msg.compound_nmembers()? as usize;
            let mut p = 8;

            for _ in 0..nmembers {
                let name_start = p;
                let name_len = data[p..].iter().position(|&b| b == 0)? + 1;
                p = if version < 3 {
                    name_start + ((name_len + 7) & !7)
                } else {
                    p + name_len
                };
                p += compound_member_offset_size(version, size);
                if version < 3 {
                    p += 28;
                }
                let member_len = datatype_encoded_len(&data[p..])?;
                p += member_len;
            }
            return (p <= data.len()).then_some(p);
        }
        DatatypeClass::VarLen => {
            if let Some(base_len) = datatype_encoded_len(&data[8..]) {
                return Some(8 + base_len);
            }
            if data.len() < 12 {
                return None;
            }
            let base_len = datatype_encoded_len(&data[12..])?;
            return Some(12 + base_len);
        }
        DatatypeClass::Array => {
            if data.len() < 9 {
                return None;
            }
            let ndims = data[8] as usize;
            let mut p = if version >= 4 { 9usize } else { 12usize };
            p += ndims.checked_mul(4)?;
            let base_len = datatype_encoded_len(&data[p..])?;
            p += base_len;
            return (p <= data.len()).then_some(p);
        }
    };

    let len = 8 + prop_len;
    (len <= data.len() && len >= 8 && size <= u32::MAX as usize).then_some(len)
}

fn compound_member_offset_size(version: u8, compound_size: usize) -> usize {
    if version < 4 {
        return 4;
    }

    bytes_needed(compound_size.saturating_sub(1).max(1))
}

fn bytes_needed(mut value: usize) -> usize {
    let mut bytes = 1;
    while value > 0xff {
        value >>= 8;
        bytes += 1;
    }
    bytes
}

fn read_le_var_usize(bytes: &[u8]) -> usize {
    let mut value = 0usize;
    for (idx, byte) in bytes.iter().enumerate() {
        value |= (*byte as usize) << (idx * 8);
    }
    value
}
