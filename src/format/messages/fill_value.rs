use crate::error::{Error, Result};

/// Parsed Fill Value message (types 0x0004 old and 0x0005 new).
#[derive(Debug, Clone)]
pub struct FillValueMessage {
    pub version: u8,
    /// Whether a fill value is defined.
    pub defined: bool,
    /// The raw fill value bytes (if defined).
    pub value: Option<Vec<u8>>,
}

impl FillValueMessage {
    /// Decode fill value message (type 0x0005, version 2 or 3).
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidFormat("empty fill value message".into()));
        }

        let version = data[0];
        match version {
            1 | 2 => Self::decode_v2(data),
            3 => Self::decode_v3(data),
            _ => Err(Error::Unsupported(format!(
                "fill value message version {version}"
            ))),
        }
    }

    fn decode_v2(data: &[u8]) -> Result<Self> {
        // v2: version(1) + space_alloc_time(1) + fill_write_time(1) + fill_defined(1) + [size(4) + value]
        if data.len() < 4 {
            return Err(Error::InvalidFormat("fill value v2 too short".into()));
        }

        let defined = data[3] != 0;
        let value = if defined && data.len() > 4 {
            let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            if size > 0 && data.len() >= 8 + size {
                Some(data[8..8 + size].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            version: data[0],
            defined,
            value,
        })
    }

    fn decode_v3(data: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1) + [size(4) + value]
        if data.len() < 2 {
            return Err(Error::InvalidFormat("fill value v3 too short".into()));
        }

        let flags = data[1];
        let defined = flags & 0x20 != 0 || flags & 0x04 != 0;

        let value = if flags & 0x20 != 0 && data.len() > 2 {
            let size = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
            if size > 0 && data.len() >= 6 + size {
                Some(data[6..6 + size].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            version: 3,
            defined,
            value,
        })
    }

    /// Decode old-style fill value message (type 0x0004).
    pub fn decode_old(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat("old fill value too short".into()));
        }

        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let value = if size > 0 && data.len() >= 4 + size {
            Some(data[4..4 + size].to_vec())
        } else {
            None
        };

        Ok(Self {
            version: 0,
            defined: size > 0,
            value,
        })
    }
}
