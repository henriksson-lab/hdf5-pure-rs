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
        if data.is_empty() {
            return Err(Error::InvalidFormat("empty link info message".into()));
        }

        let mut pos = 0;
        let sa = sizeof_addr as usize;

        let version = data[pos];
        pos += 1;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "link info message version {version}"
            )));
        }

        let flags = data[pos];
        pos += 1;

        let has_max_crt_order = flags & 0x01 != 0;
        let has_corder_btree = flags & 0x02 != 0;

        let max_creation_index = if has_max_crt_order {
            let val = read_le_u64(&data[pos..], 8);
            pos += 8;
            Some(val)
        } else {
            None
        };

        let fractal_heap_addr = read_le_u64(&data[pos..], sa);
        pos += sa;

        let name_btree_addr = read_le_u64(&data[pos..], sa);
        pos += sa;

        let corder_btree_addr = if has_corder_btree {
            let addr = read_le_u64(&data[pos..], sa);
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

fn read_le_u64(data: &[u8], size: usize) -> u64 {
    let mut val = 0u64;
    for i in 0..size.min(8).min(data.len()) {
        val |= (data[i] as u64) << (i * 8);
    }
    val
}
