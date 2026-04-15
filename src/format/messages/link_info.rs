use crate::error::{Error, Result};
use crate::io::reader::UNDEF_ADDR;

/// Parsed Link Info message (type 0x0002).
/// Contains pointers to dense link storage structures.
#[derive(Debug, Clone)]
pub struct LinkInfoMessage {
    /// Version of the link info message.
    pub version: u8,
    /// Flags.
    pub flags: u8,
    /// Maximum creation order index (if tracked).
    pub max_creation_index: Option<u64>,
    /// Address of fractal heap for storing link data.
    pub fractal_heap_addr: u64,
    /// Address of v2 B-tree for name index.
    pub name_btree_addr: u64,
    /// Address of v2 B-tree for creation order index (if indexed).
    pub corder_btree_addr: Option<u64>,
}

impl LinkInfoMessage {
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        let mut pos = 0;
        let sa = sizeof_addr as usize;

        let version = read_u8(data, &mut pos, "link info message version")?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "link info message version {version}"
            )));
        }

        let flags = read_u8(data, &mut pos, "link info message flags")?;

        let has_max_crt_order = flags & 0x01 != 0;
        let has_corder_btree = flags & 0x02 != 0;

        let max_creation_index = if has_max_crt_order {
            let val = read_le_u64(data, &mut pos, 8, "link info max creation index")?;
            Some(val)
        } else {
            None
        };

        let fractal_heap_addr = read_le_u64(data, &mut pos, sa, "link info fractal heap address")?;

        let name_btree_addr = read_le_u64(data, &mut pos, sa, "link info name btree address")?;

        let corder_btree_addr = if has_corder_btree {
            let addr = read_le_u64(data, &mut pos, sa, "link info creation order btree address")?;
            Some(addr)
        } else {
            None
        };

        Ok(Self {
            version,
            flags,
            max_creation_index,
            fractal_heap_addr,
            name_btree_addr,
            corder_btree_addr,
        })
    }

    /// Whether this group has dense link storage (fractal heap).
    pub fn has_dense_storage(&self) -> bool {
        self.fractal_heap_addr != UNDEF_ADDR
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

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    *pos += 1;
    Ok(value)
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
