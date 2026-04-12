use crate::error::{Error, Result};

/// Filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

/// A single filter in the pipeline.
#[derive(Debug, Clone)]
pub struct FilterDesc {
    pub id: u16,
    pub name: Option<String>,
    pub flags: u16,
    pub client_data: Vec<u32>,
}

/// Parsed Filter Pipeline message (type 0x000B).
#[derive(Debug, Clone)]
pub struct FilterPipelineMessage {
    pub version: u8,
    pub filters: Vec<FilterDesc>,
}

impl FilterPipelineMessage {
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat(
                "filter pipeline message too short".into(),
            ));
        }

        let version = data[0];
        let nfilters = data[1] as usize;

        match version {
            1 => Self::decode_v1(data, nfilters),
            2 => Self::decode_v2(data, nfilters),
            _ => Err(Error::Unsupported(format!(
                "filter pipeline version {version}"
            ))),
        }
    }

    fn decode_v1(data: &[u8], nfilters: usize) -> Result<Self> {
        // v1: version(1) + nfilters(1) + reserved(6)
        let mut pos = 8;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            if pos + 8 > data.len() {
                break;
            }

            let id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            let flags = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
            let cd_nelmts = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            // Name (null-terminated, padded to 8-byte boundary)
            let name = if name_len > 0 {
                let name_end = pos + name_len;
                let null_pos = data[pos..name_end]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(name_len);
                let n = String::from_utf8_lossy(&data[pos..pos + null_pos]).to_string();
                // Pad to 8-byte boundary
                let padded = (name_len + 7) & !7;
                pos += padded;
                Some(n)
            } else {
                None
            };

            // Client data values
            let mut client_data = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
                client_data.push(val);
            }

            // Pad cd_nelmts to even number in v1
            if cd_nelmts % 2 != 0 {
                pos += 4;
            }

            filters.push(FilterDesc {
                id,
                name,
                flags,
                client_data,
            });
        }

        Ok(Self {
            version: 1,
            filters,
        })
    }

    fn decode_v2(data: &[u8], nfilters: usize) -> Result<Self> {
        // v2: version(1) + nfilters(1), no reserved bytes
        let mut pos = 2;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            if pos + 2 > data.len() {
                break;
            }

            let id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;

            // v2: name_length and name are OMITTED for known filter IDs (< 256)
            let name = if id >= 256 {
                let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                if name_len > 0 {
                    let null_pos = data[pos..pos + name_len]
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(name_len);
                    let n = String::from_utf8_lossy(&data[pos..pos + null_pos]).to_string();
                    pos += name_len;
                    Some(n)
                } else {
                    None
                }
            } else {
                None
            };

            let flags = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
            let cd_nelmts = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            let mut client_data = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
                client_data.push(val);
            }

            filters.push(FilterDesc {
                id,
                name,
                flags,
                client_data,
            });
        }

        Ok(Self {
            version: 2,
            filters,
        })
    }
}
