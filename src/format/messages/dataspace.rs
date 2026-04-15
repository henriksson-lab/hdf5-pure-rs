use crate::error::{Error, Result};

/// Dataspace type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataspaceType {
    Scalar,
    Simple,
    Null,
}

/// Parsed Dataspace message (type 0x0001).
#[derive(Debug, Clone)]
pub struct DataspaceMessage {
    pub version: u8,
    pub space_type: DataspaceType,
    pub ndims: u8,
    /// Current dimension sizes.
    pub dims: Vec<u64>,
    /// Maximum dimension sizes (None means same as current).
    pub max_dims: Option<Vec<u64>>,
}

impl DataspaceMessage {
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat("dataspace message too short".into()));
        }

        let version = data[0];
        let ndims = data[1];

        match version {
            1 => Self::decode_v1(data, ndims),
            2 => Self::decode_v2(data, ndims),
            _ => {
                // Some embedded dataspaces (in attribute messages) may not have
                // a standard version byte. Try to parse as v2 with what we have.
                // If ndims is 0 and the data is very short, it might be a scalar.
                if data.len() <= 4 {
                    Ok(Self {
                        version,
                        space_type: DataspaceType::Scalar,
                        ndims: 0,
                        dims: vec![],
                        max_dims: None,
                    })
                } else {
                    // Try as v1
                    Self::decode_v1(data, ndims)
                }
            }
        }
    }

    fn decode_v1(data: &[u8], ndims: u8) -> Result<Self> {
        ensure_available(data, 0, 8, "dataspace v1 header")?;
        let flags = data[2];
        // v1: reserved bytes [3..8], then dimensions
        let has_max = flags & 0x01 != 0;

        let mut pos = 8; // skip 5 reserved bytes after version(1)+ndims(1)+flags(1)

        let dims = read_dims(data, &mut pos, ndims as usize, "dataspace v1 dimensions")?;
        let max_dims = if has_max {
            Some(read_dims(
                data,
                &mut pos,
                ndims as usize,
                "dataspace v1 max dimensions",
            )?)
        } else {
            None
        };

        // Permutation indices (deprecated, skip if present)
        // flags & 0x02 != 0 means permutation present

        let space_type = if ndims == 0 {
            DataspaceType::Scalar
        } else {
            DataspaceType::Simple
        };

        Ok(Self {
            version: 1,
            space_type,
            ndims,
            dims,
            max_dims,
        })
    }

    fn decode_v2(data: &[u8], ndims: u8) -> Result<Self> {
        let flags = data[2];
        let space_type_val = data[3];

        let space_type = match space_type_val {
            0 => DataspaceType::Scalar,
            1 => DataspaceType::Simple,
            2 => DataspaceType::Null,
            _ => {
                return Err(Error::InvalidFormat(format!(
                    "unknown dataspace type {space_type_val}"
                )))
            }
        };

        let has_max = flags & 0x01 != 0;
        let mut pos = 4;

        let dims = read_dims(data, &mut pos, ndims as usize, "dataspace v2 dimensions")?;
        let max_dims = if has_max {
            Some(read_dims(
                data,
                &mut pos,
                ndims as usize,
                "dataspace v2 max dimensions",
            )?)
        } else {
            None
        };

        Ok(Self {
            version: 2,
            space_type,
            ndims,
            dims,
            max_dims,
        })
    }
}

fn read_dims(data: &[u8], pos: &mut usize, count: usize, context: &str) -> Result<Vec<u64>> {
    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        let val = read_le_u64(data, pos, 8, context)?;
        dims.push(val);
    }
    Ok(dims)
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

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    ensure_available(data, *pos, size, context)?;
    let mut val = 0u64;
    for i in 0..size {
        val |= (data[*pos + i] as u64) << (i * 8);
    }
    *pos += size;
    Ok(val)
}
