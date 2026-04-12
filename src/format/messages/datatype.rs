use crate::error::{Error, Result};

/// A field in a compound datatype.
#[derive(Debug, Clone)]
pub struct CompoundField {
    pub name: String,
    pub byte_offset: usize,
    pub size: usize,
    pub class: DatatypeClass,
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
            _ => Err(Error::InvalidFormat(format!("unknown datatype class {val}"))),
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

        Ok(Self {
            version,
            class,
            class_bits,
            size,
            properties,
        })
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
    /// Returns Vec of (name, byte_offset, member_type_size, member_type_class).
    pub fn compound_fields(&self) -> Option<Vec<CompoundField>> {
        let nmembers = self.compound_nmembers()? as usize;
        let mut fields = Vec::with_capacity(nmembers);
        let data = &self.properties;
        let mut p = 0;

        for _ in 0..nmembers {
            if p >= data.len() {
                break;
            }

            // Name (null-terminated, padded to 8-byte boundary in v1/v2)
            let name_start = p;
            let name_end = data[p..].iter().position(|&b| b == 0)?;
            let name = String::from_utf8_lossy(&data[p..p + name_end]).to_string();
            if self.version < 3 {
                let name_with_null = name_end + 1;
                let padded = (name_with_null + 7) & !7;
                p = name_start + padded;
            } else {
                p += name_end + 1;
            }

            // Byte offset
            if p + 4 > data.len() { break; }
            let byte_offset = u32::from_le_bytes([data[p], data[p+1], data[p+2], data[p+3]]) as usize;
            p += 4;

            // Version 1/2: skip dimension info (1+3+4+4+16 = 28 bytes)
            if self.version < 3 {
                p += 28;
            }

            // Embedded member datatype
            if p + 8 > data.len() { break; }
            let member_dt = DatatypeMessage::decode(&data[p..]).ok()?;
            let member_type_size = member_dt.size as usize;
            let member_class = member_dt.class;

            // Skip past the embedded type: header(8) + actual property size
            // (DatatypeMessage::decode consumes all remaining bytes as properties,
            //  so we need to calculate the real property size based on type class)
            let prop_size = match member_class {
                DatatypeClass::FixedPoint | DatatypeClass::BitField => 4, // bit_offset(2) + bit_precision(2)
                DatatypeClass::FloatingPoint => 12, // bit_offset(2) + bit_precision(2) + epos(1) + esize(1) + mpos(1) + msize(1) + ebias(4)
                DatatypeClass::String => 0, // no properties for string type in compound
                _ => 0,
            };
            p += 8 + prop_size;

            fields.push(CompoundField {
                name,
                byte_offset,
                size: member_type_size,
                class: member_class,
            });
        }

        Some(fields)
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
    pub fn enum_members(&self) -> Option<Vec<(String, u64)>> {
        let nmembers = self.enum_nmembers()? as usize;
        let data = &self.properties;
        if data.len() < 8 {
            return None;
        }

        // Base type (embedded datatype)
        let base_dt = DatatypeMessage::decode(data).ok()?;
        let base_size = base_dt.size as usize;
        let base_prop_size = match base_dt.class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField => 4,
            _ => 0,
        };
        let mut p = 8 + base_prop_size;

        // Member names (null-terminated, padded to 8 in v1/v2)
        let mut names = Vec::with_capacity(nmembers);
        for _ in 0..nmembers {
            if p >= data.len() { break; }
            let name_end = data[p..].iter().position(|&b| b == 0)?;
            let name = String::from_utf8_lossy(&data[p..p + name_end]).to_string();
            if self.version < 3 {
                let padded = (name_end + 1 + 7) & !7;
                p += padded;
            } else {
                p += name_end + 1;
            }
            names.push(name);
        }

        // Member values (each base_size bytes)
        let mut members = Vec::with_capacity(nmembers);
        for name in names {
            if p + base_size > data.len() { break; }
            let mut val = 0u64;
            for i in 0..base_size.min(8) {
                val |= (data[p + i] as u64) << (i * 8);
            }
            p += base_size;
            members.push((name, val));
        }

        Some(members)
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
}
