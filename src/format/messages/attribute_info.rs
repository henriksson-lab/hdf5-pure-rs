use crate::error::{Error, Result};
use crate::io::reader::UNDEF_ADDR;

/// Parsed Attribute Info message (type 0x0015).
#[derive(Debug, Clone)]
pub struct AttributeInfoMessage {
    pub version: u8,
    pub flags: u8,
    pub max_creation_index: Option<u16>,
    pub fractal_heap_addr: u64,
    pub name_btree_addr: u64,
    pub corder_btree_addr: Option<u64>,
}

impl AttributeInfoMessage {
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat(
                "attribute info message too short".into(),
            ));
        }

        let sa = sizeof_addr as usize;
        let mut pos = 0;

        let version = data[pos];
        pos += 1;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "attribute info version {version}"
            )));
        }

        let flags = data[pos];
        pos += 1;

        let has_max_crt_order = flags & 0x01 != 0;
        let has_corder_btree = flags & 0x02 != 0;

        let max_creation_index = if has_max_crt_order {
            let val = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
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
