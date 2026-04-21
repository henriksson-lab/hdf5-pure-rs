use crate::error::{Error, Result};

const MAX_FILTERS: usize = 32;
const MAX_FILTER_CLIENT_VALUES: usize = 1024;

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
        let result = Self::decode_impl(data);

        #[cfg(feature = "tracehash")]
        if let Ok(message) = &result {
            let mut th = tracehash::th_call!("hdf5.filter_pipeline.decode");
            th.input_bytes(data);
            th.output_value(&(true));
            th.output_u64(message.version as u64);
            th.output_u64(message.filters.len() as u64);
            th.finish();
        }

        result
    }

    fn decode_impl(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat(
                "filter pipeline message too short".into(),
            ));
        }

        let version = data[0];
        let nfilters = data[1] as usize;
        if nfilters > MAX_FILTERS {
            return Err(Error::InvalidFormat(format!(
                "filter pipeline has too many filters: {nfilters}"
            )));
        }

        let result = match version {
            1 => Self::decode_v1(data, nfilters),
            2 => Self::decode_v2(data, nfilters),
            _ => Err(Error::InvalidFormat(format!(
                "filter pipeline version {version}"
            ))),
        };

        result
    }

    fn decode_v1(data: &[u8], nfilters: usize) -> Result<Self> {
        // v1: version(1) + nfilters(1) + reserved(6)
        ensure_available(data, 0, 8, "filter pipeline v1 header")?;
        let mut pos = 8;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            let id = read_u16_le(data, &mut pos, "filter pipeline v1 filter id")?;
            let name_len = read_u16_le(data, &mut pos, "filter pipeline v1 name length")? as usize;
            // The v1 spec requires the name length (including null terminator
            // and 8-byte padding) to itself be a multiple of eight; matches
            // upstream `H5O__pline_decode`.
            if name_len % 8 != 0 {
                return Err(Error::InvalidFormat(format!(
                    "filter pipeline v1 name length {name_len} is not a multiple of eight"
                )));
            }
            let flags = read_u16_le(data, &mut pos, "filter pipeline v1 flags")?;
            let cd_nelmts =
                read_u16_le(data, &mut pos, "filter pipeline v1 client data count")? as usize;
            if cd_nelmts > MAX_FILTER_CLIENT_VALUES {
                return Err(Error::InvalidFormat(format!(
                    "filter pipeline v1 client data count {cd_nelmts} exceeds supported maximum {MAX_FILTER_CLIENT_VALUES}"
                )));
            }

            // Name (null-terminated, padded to 8-byte boundary)
            let name = if name_len > 0 {
                ensure_available(data, pos, name_len, "filter pipeline v1 name")?;
                let name_end = pos + name_len;
                let null_pos = data[pos..name_end]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(name_len);
                let n = String::from_utf8_lossy(&data[pos..pos + null_pos]).to_string();
                // Pad to 8-byte boundary
                let padded = (name_len + 7) & !7;
                ensure_available(data, pos, padded, "filter pipeline v1 padded name")?;
                pos += padded;
                Some(n)
            } else {
                None
            };

            // Client data values
            let mut client_data = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                let val = read_u32_le(data, &mut pos, "filter pipeline v1 client data")?;
                client_data.push(val);
            }

            // Pad cd_nelmts to even number in v1
            if cd_nelmts % 2 != 0 {
                ensure_available(data, pos, 4, "filter pipeline v1 client data padding")?;
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
            let id = read_u16_le(data, &mut pos, "filter pipeline v2 filter id")?;

            // v2: name_length and name are OMITTED for known filter IDs (< 256)
            let name = if id >= 256 {
                let name_len =
                    read_u16_le(data, &mut pos, "filter pipeline v2 name length")? as usize;
                if name_len > 0 {
                    ensure_available(data, pos, name_len, "filter pipeline v2 name")?;
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

            let flags = read_u16_le(data, &mut pos, "filter pipeline v2 flags")?;
            let cd_nelmts =
                read_u16_le(data, &mut pos, "filter pipeline v2 client data count")? as usize;
            if cd_nelmts > MAX_FILTER_CLIENT_VALUES {
                return Err(Error::InvalidFormat(format!(
                    "filter pipeline v2 client data count {cd_nelmts} exceeds supported maximum {MAX_FILTER_CLIENT_VALUES}"
                )));
            }

            let mut client_data = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                let val = read_u32_le(data, &mut pos, "filter pipeline v2 client data")?;
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

fn ensure_available(data: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn read_u16_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u16> {
    ensure_available(data, *pos, 2, context)?;
    let value = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(value)
}

fn read_u32_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u32> {
    ensure_available(data, *pos, 4, context)?;
    let value = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(value)
}
