use crate::error::{Error, Result};

pub const FILL_TIME_NEVER: u8 = 1;

/// Parsed Fill Value message (types 0x0004 old and 0x0005 new).
#[derive(Debug, Clone)]
pub struct FillValueMessage {
    pub version: u8,
    /// Allocation-time policy encoded by the fill-value message.
    pub alloc_time: u8,
    /// Fill-write-time policy encoded by the fill-value message.
    pub fill_time: u8,
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
            _ => Err(Error::InvalidFormat(format!(
                "fill value message version {version}"
            ))),
        }
    }

    fn decode_v2(data: &[u8]) -> Result<Self> {
        // v2: version(1) + space_alloc_time(1) + fill_write_time(1) + fill_defined(1) + [size(4) + value]
        if data.len() < 4 {
            return Err(Error::InvalidFormat("fill value v2 too short".into()));
        }

        let alloc_time = data[1];
        let fill_time = data[2];
        let defined = data[3] != 0;
        let value = if defined {
            if data.len() < 8 {
                return Err(Error::InvalidFormat(
                    "fill value v2 missing value size".into(),
                ));
            }
            let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
            if data.len() < 8 + size {
                return Err(Error::InvalidFormat(
                    "fill value v2 value is truncated".into(),
                ));
            }
            if size > 0 {
                Some(data[8..8 + size].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        let message = Self {
            version: data[0],
            alloc_time,
            fill_time,
            defined,
            value,
        };
        trace_fill_value(
            data,
            message.version,
            alloc_time,
            fill_time,
            defined,
            &message.value,
        );
        Ok(message)
    }

    fn decode_v3(data: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1) + [size(4) + value]
        if data.len() < 2 {
            return Err(Error::InvalidFormat("fill value v3 too short".into()));
        }

        let flags = data[1];
        let alloc_time = flags & 0x03;
        let fill_time = (flags >> 2) & 0x03;
        let undefined = flags & 0x10 != 0;
        let have_value = flags & 0x20 != 0;
        if undefined && have_value {
            return Err(Error::InvalidFormat(
                "fill value v3 has both undefined and value-present flags".into(),
            ));
        }
        let defined = !undefined;

        let value = if have_value {
            if data.len() < 6 {
                return Err(Error::InvalidFormat(
                    "fill value v3 missing value size".into(),
                ));
            }
            let size = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
            if data.len() < 6 + size {
                return Err(Error::InvalidFormat(
                    "fill value v3 value is truncated".into(),
                ));
            }
            if size > 0 {
                Some(data[6..6 + size].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        let message = Self {
            version: 3,
            alloc_time,
            fill_time,
            defined,
            value,
        };
        trace_fill_value(
            data,
            message.version,
            alloc_time,
            fill_time,
            defined,
            &message.value,
        );
        Ok(message)
    }

    /// Decode old-style fill value message (type 0x0004).
    pub fn decode_old(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat("old fill value too short".into()));
        }

        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let value = if size > 0 {
            if data.len() < 4 + size {
                return Err(Error::InvalidFormat("old fill value is truncated".into()));
            }
            Some(data[4..4 + size].to_vec())
        } else {
            None
        };

        let message = Self {
            version: 0,
            alloc_time: 2,
            fill_time: 2,
            defined: size > 0,
            value,
        };
        trace_fill_value(data, 0, 2, 2, message.defined, &message.value);
        Ok(message)
    }
}

#[cfg(feature = "tracehash")]
fn trace_fill_value(
    data: &[u8],
    version: u8,
    alloc_time: u8,
    fill_time: u8,
    defined: bool,
    value: &Option<Vec<u8>>,
) {
    let mut th = tracehash::th_call!("hdf5.fill_value.decode");
    th.input_bytes(data);
    th.output_bool(true);
    th.output_u64(version as u64);
    th.output_u64(alloc_time as u64);
    th.output_u64(fill_time as u64);
    th.output_bool(defined);
    if let Some(value) = value {
        th.output_bool(true);
        th.output_u64(value.len() as u64);
        th.output_bytes(value);
    } else {
        th.output_bool(false);
        th.output_u64(0);
    }
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_fill_value(
    _data: &[u8],
    _version: u8,
    _alloc_time: u8,
    _fill_time: u8,
    _defined: bool,
    _value: &Option<Vec<u8>>,
) {
}
