use crate::error::{Error, Result};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;

/// Parsed Attribute message (type 0x000C).
#[derive(Debug, Clone)]
pub struct AttributeMessage {
    pub version: u8,
    pub name: String,
    /// Character encoding for the attribute name: 0=ASCII, 1=UTF-8.
    pub char_encoding: u8,
    pub datatype: DatatypeMessage,
    pub dataspace: DataspaceMessage,
    /// Raw attribute data bytes.
    pub data: Vec<u8>,
}

impl AttributeMessage {
    pub fn decode(raw: &[u8]) -> Result<Self> {
        Self::decode_impl(raw)
    }

    fn decode_impl(raw: &[u8]) -> Result<Self> {
        if raw.len() < 6 {
            return Err(Error::InvalidFormat("attribute message too short".into()));
        }

        let version = raw[0];
        match version {
            1 => Self::decode_v1(raw),
            2 => Self::decode_v2(raw),
            3 => Self::decode_v3(raw),
            _ => Err(Error::InvalidFormat(format!(
                "attribute message version {version}"
            ))),
        }
    }

    fn decode_v1(raw: &[u8]) -> Result<Self> {
        // v1: version(1) + reserved(1) + name_size(2) + datatype_size(2) + dataspace_size(2)
        ensure_available(raw, 0, 8, "attribute v1 header")?;
        let name_size = read_u16_le_at(raw, 2, "attribute v1 name size")? as usize;
        let dt_size = read_u16_le_at(raw, 4, "attribute v1 datatype size")? as usize;
        let ds_size = read_u16_le_at(raw, 6, "attribute v1 dataspace size")? as usize;
        if name_size == 0 {
            return Err(Error::InvalidFormat(
                "attribute message name length is zero".into(),
            ));
        }
        let mut pos = 8;

        // Name (null-padded to 8-byte boundary)
        ensure_available(raw, pos, name_size, "attribute v1 name")?;
        let name_field_end = checked_end(pos, name_size, "attribute v1 name")?;
        let name_len = raw[pos..name_field_end]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_size);
        let name_end = checked_end(pos, name_len, "attribute v1 name")?;
        let name = String::from_utf8_lossy(&raw[pos..name_end]).to_string();
        let name_padded = align8(name_size, "attribute v1 name")?;
        ensure_available(raw, pos, name_padded, "attribute v1 padded name")?;
        advance_pos(&mut pos, name_padded, "attribute v1 padded name")?;

        // Datatype (padded to 8-byte boundary)
        ensure_available(raw, pos, dt_size, "attribute v1 datatype")?;
        let dt_end = checked_end(pos, dt_size, "attribute v1 datatype")?;
        let datatype = DatatypeMessage::decode(&raw[pos..dt_end])?;
        let dt_padded = align8(dt_size, "attribute v1 datatype")?;
        ensure_available(raw, pos, dt_padded, "attribute v1 padded datatype")?;
        advance_pos(&mut pos, dt_padded, "attribute v1 padded datatype")?;

        // Dataspace (padded to 8-byte boundary)
        ensure_available(raw, pos, ds_size, "attribute v1 dataspace")?;
        let ds_end = checked_end(pos, ds_size, "attribute v1 dataspace")?;
        let dataspace = DataspaceMessage::decode(&raw[pos..ds_end])?;
        let ds_padded = align8(ds_size, "attribute v1 dataspace")?;
        ensure_available(raw, pos, ds_padded, "attribute v1 padded dataspace")?;
        advance_pos(&mut pos, ds_padded, "attribute v1 padded dataspace")?;

        // Data
        let data = raw[pos..].to_vec();

        Ok(Self {
            version: 1,
            name,
            char_encoding: 0,
            datatype,
            dataspace,
            data,
        })
    }

    fn decode_v2(raw: &[u8]) -> Result<Self> {
        // v2: version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2)
        ensure_available(raw, 0, 8, "attribute v2 header")?;
        let name_size = read_u16_le_at(raw, 2, "attribute v2 name size")? as usize;
        let dt_size = read_u16_le_at(raw, 4, "attribute v2 datatype size")? as usize;
        let ds_size = read_u16_le_at(raw, 6, "attribute v2 dataspace size")? as usize;
        if name_size == 0 {
            return Err(Error::InvalidFormat(
                "attribute message name length is zero".into(),
            ));
        }
        let mut pos = 8;

        // Name (NOT padded in v2)
        ensure_available(raw, pos, name_size, "attribute v2 name")?;
        let name_field_end = checked_end(pos, name_size, "attribute v2 name")?;
        let name_len = raw[pos..name_field_end]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_size);
        let name_end = checked_end(pos, name_len, "attribute v2 name")?;
        let name = String::from_utf8_lossy(&raw[pos..name_end]).to_string();
        advance_pos(&mut pos, name_size, "attribute v2 name")?;

        // Datatype (NOT padded in v2)
        ensure_available(raw, pos, dt_size, "attribute v2 datatype")?;
        let dt_end = checked_end(pos, dt_size, "attribute v2 datatype")?;
        let datatype = DatatypeMessage::decode(&raw[pos..dt_end])?;
        advance_pos(&mut pos, dt_size, "attribute v2 datatype")?;

        // Dataspace (NOT padded in v2)
        ensure_available(raw, pos, ds_size, "attribute v2 dataspace")?;
        let ds_end = checked_end(pos, ds_size, "attribute v2 dataspace")?;
        let dataspace = DataspaceMessage::decode(&raw[pos..ds_end])?;
        advance_pos(&mut pos, ds_size, "attribute v2 dataspace")?;

        let data = raw[pos..].to_vec();

        Ok(Self {
            version: 2,
            name,
            char_encoding: 0,
            datatype,
            dataspace,
            data,
        })
    }

    fn decode_v3(raw: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) + encoding(1)
        ensure_available(raw, 0, 9, "attribute v3 header")?;
        let _flags = raw[1];
        let name_size = read_u16_le_at(raw, 2, "attribute v3 name size")? as usize;
        let dt_size = read_u16_le_at(raw, 4, "attribute v3 datatype size")? as usize;
        let ds_size = read_u16_le_at(raw, 6, "attribute v3 dataspace size")? as usize;
        let encoding = raw[8]; // character encoding: 0=ASCII, 1=UTF-8
        if name_size == 0 {
            return Err(Error::InvalidFormat(
                "attribute message name length is zero".into(),
            ));
        }
        let mut pos = 9;

        ensure_available(raw, pos, name_size, "attribute v3 name")?;
        let name_field_end = checked_end(pos, name_size, "attribute v3 name")?;
        let name_len = raw[pos..name_field_end]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_size);
        let name_end = checked_end(pos, name_len, "attribute v3 name")?;
        let name = String::from_utf8_lossy(&raw[pos..name_end]).to_string();
        advance_pos(&mut pos, name_size, "attribute v3 name")?;

        ensure_available(raw, pos, dt_size, "attribute v3 datatype")?;
        let dt_end = checked_end(pos, dt_size, "attribute v3 datatype")?;
        let datatype = DatatypeMessage::decode(&raw[pos..dt_end])?;
        advance_pos(&mut pos, dt_size, "attribute v3 datatype")?;

        ensure_available(raw, pos, ds_size, "attribute v3 dataspace")?;
        let ds_end = checked_end(pos, ds_size, "attribute v3 dataspace")?;
        let dataspace = DataspaceMessage::decode(&raw[pos..ds_end])?;
        advance_pos(&mut pos, ds_size, "attribute v3 dataspace")?;

        let data = raw[pos..].to_vec();

        Ok(Self {
            version: 3,
            name,
            char_encoding: encoding,
            datatype,
            dataspace,
            data,
        })
    }

    /// Get total number of elements.
    pub fn num_elements(&self) -> Result<u64> {
        if self.dataspace.dims.is_empty() {
            Ok(1) // scalar
        } else {
            self.dataspace.dims.iter().try_fold(1u64, |acc, &dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| Error::InvalidFormat("attribute element count overflow".into()))
            })
        }
    }

    /// Get total data size in bytes.
    pub fn data_size(&self) -> Result<usize> {
        let elements = usize::try_from(self.num_elements()?)
            .map_err(|_| Error::InvalidFormat("attribute element count overflow".into()))?;
        elements
            .checked_mul(self.datatype.size as usize)
            .ok_or_else(|| Error::InvalidFormat("attribute data size overflow".into()))
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

fn checked_window<'a>(raw: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = checked_end(pos, len, context)?;
    raw.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u16_le_at(raw: &[u8], pos: usize, context: &str) -> Result<u16> {
    let bytes = checked_window(raw, pos, 2, context)?;
    let bytes: [u8; 2] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u16::from_le_bytes(bytes))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = checked_end(*pos, len, context)?;
    Ok(())
}

fn checked_end(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn align8(len: usize, context: &str) -> Result<usize> {
    len.checked_add(7)
        .map(|value| value & !7)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} padded size overflow")))
}

#[cfg(test)]
mod tests {
    use super::{advance_pos, align8, read_u16_le_at};

    #[test]
    fn attribute_padding_rejects_overflow() {
        let err = align8(usize::MAX, "attribute").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn attribute_cursor_advance_rejects_overflow() {
        let mut pos = usize::MAX;
        let err = advance_pos(&mut pos, 1, "attribute").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn attribute_u16_reader_rejects_offset_overflow() {
        let err = read_u16_le_at(&[], usize::MAX, "attribute test field").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
