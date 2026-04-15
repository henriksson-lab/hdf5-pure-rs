use crate::error::{Error, Result};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;

/// Parsed Attribute message (type 0x000C).
#[derive(Debug, Clone)]
pub struct AttributeMessage {
    pub version: u8,
    pub name: String,
    pub datatype: DatatypeMessage,
    pub dataspace: DataspaceMessage,
    /// Raw attribute data bytes.
    pub data: Vec<u8>,
}

impl AttributeMessage {
    pub fn decode(raw: &[u8]) -> Result<Self> {
        if raw.len() < 6 {
            return Err(Error::InvalidFormat("attribute message too short".into()));
        }

        let version = raw[0];
        match version {
            1 => Self::decode_v1(raw),
            2 => Self::decode_v2(raw),
            3 => Self::decode_v3(raw),
            _ => Err(Error::Unsupported(format!(
                "attribute message version {version}"
            ))),
        }
    }

    fn decode_v1(raw: &[u8]) -> Result<Self> {
        // v1: version(1) + reserved(1) + name_size(2) + datatype_size(2) + dataspace_size(2)
        ensure_available(raw, 0, 8, "attribute v1 header")?;
        let name_size = u16::from_le_bytes([raw[2], raw[3]]) as usize;
        let dt_size = u16::from_le_bytes([raw[4], raw[5]]) as usize;
        let ds_size = u16::from_le_bytes([raw[6], raw[7]]) as usize;
        let mut pos = 8;

        // Name (null-padded to 8-byte boundary)
        ensure_available(raw, pos, name_size, "attribute v1 name")?;
        let name_end = raw[pos..pos + name_size]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_size);
        let name = String::from_utf8_lossy(&raw[pos..pos + name_end]).to_string();
        let name_padded = (name_size + 7) & !7;
        ensure_available(raw, pos, name_padded, "attribute v1 padded name")?;
        pos += name_padded;

        // Datatype (padded to 8-byte boundary)
        ensure_available(raw, pos, dt_size, "attribute v1 datatype")?;
        let datatype = DatatypeMessage::decode(&raw[pos..pos + dt_size])?;
        let dt_padded = (dt_size + 7) & !7;
        ensure_available(raw, pos, dt_padded, "attribute v1 padded datatype")?;
        pos += dt_padded;

        // Dataspace (padded to 8-byte boundary)
        ensure_available(raw, pos, ds_size, "attribute v1 dataspace")?;
        let dataspace = DataspaceMessage::decode(&raw[pos..pos + ds_size])?;
        let ds_padded = (ds_size + 7) & !7;
        ensure_available(raw, pos, ds_padded, "attribute v1 padded dataspace")?;
        pos += ds_padded;

        // Data
        let data = raw[pos..].to_vec();

        Ok(Self {
            version: 1,
            name,
            datatype,
            dataspace,
            data,
        })
    }

    fn decode_v2(raw: &[u8]) -> Result<Self> {
        // v2: version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2)
        ensure_available(raw, 0, 8, "attribute v2 header")?;
        let name_size = u16::from_le_bytes([raw[2], raw[3]]) as usize;
        let dt_size = u16::from_le_bytes([raw[4], raw[5]]) as usize;
        let ds_size = u16::from_le_bytes([raw[6], raw[7]]) as usize;
        let mut pos = 8;

        // Name (NOT padded in v2)
        ensure_available(raw, pos, name_size, "attribute v2 name")?;
        let name_end = raw[pos..pos + name_size]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_size);
        let name = String::from_utf8_lossy(&raw[pos..pos + name_end]).to_string();
        pos += name_size;

        // Datatype (NOT padded in v2)
        ensure_available(raw, pos, dt_size, "attribute v2 datatype")?;
        let datatype = DatatypeMessage::decode(&raw[pos..pos + dt_size])?;
        pos += dt_size;

        // Dataspace (NOT padded in v2)
        ensure_available(raw, pos, ds_size, "attribute v2 dataspace")?;
        let dataspace = DataspaceMessage::decode(&raw[pos..pos + ds_size])?;
        pos += ds_size;

        let data = raw[pos..].to_vec();

        Ok(Self {
            version: 2,
            name,
            datatype,
            dataspace,
            data,
        })
    }

    fn decode_v3(raw: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) + encoding(1)
        ensure_available(raw, 0, 9, "attribute v3 header")?;
        let _flags = raw[1];
        let name_size = u16::from_le_bytes([raw[2], raw[3]]) as usize;
        let dt_size = u16::from_le_bytes([raw[4], raw[5]]) as usize;
        let ds_size = u16::from_le_bytes([raw[6], raw[7]]) as usize;
        let _encoding = raw[8]; // character encoding: 0=ASCII, 1=UTF-8
        let mut pos = 9;

        ensure_available(raw, pos, name_size, "attribute v3 name")?;
        let name_end = raw[pos..pos + name_size]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_size);
        let name = String::from_utf8_lossy(&raw[pos..pos + name_end]).to_string();
        pos += name_size;

        ensure_available(raw, pos, dt_size, "attribute v3 datatype")?;
        let datatype = DatatypeMessage::decode(&raw[pos..pos + dt_size])?;
        pos += dt_size;

        ensure_available(raw, pos, ds_size, "attribute v3 dataspace")?;
        let dataspace = DataspaceMessage::decode(&raw[pos..pos + ds_size])?;
        pos += ds_size;

        let data = raw[pos..].to_vec();

        Ok(Self {
            version: 3,
            name,
            datatype,
            dataspace,
            data,
        })
    }

    /// Get total number of elements.
    pub fn num_elements(&self) -> u64 {
        if self.dataspace.dims.is_empty() {
            1 // scalar
        } else {
            self.dataspace.dims.iter().product()
        }
    }

    /// Get total data size in bytes.
    pub fn data_size(&self) -> usize {
        self.num_elements() as usize * self.datatype.size as usize
    }
}

fn ensure_available(raw: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > raw.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}
