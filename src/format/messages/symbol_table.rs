use crate::error::Result;

/// Parsed Symbol Table message (type 0x0011).
/// Points to a v1 B-tree and local heap for group membership.
#[derive(Debug, Clone)]
pub struct SymbolTableMessage {
    /// Address of the v1 B-tree for group nodes.
    pub btree_addr: u64,
    /// Address of the local heap for link names.
    pub name_heap_addr: u64,
}

impl SymbolTableMessage {
    /// Decode from raw message bytes. `sizeof_addr` determines address width.
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        let sa = sizeof_addr as usize;
        if data.len() < sa * 2 {
            return Err(crate::error::Error::InvalidFormat(
                "symbol table message too short".into(),
            ));
        }

        let btree_addr = read_addr(data, 0, sizeof_addr);
        let name_heap_addr = read_addr(data, sa, sizeof_addr);

        Ok(Self {
            btree_addr,
            name_heap_addr,
        })
    }
}

fn read_addr(data: &[u8], offset: usize, size: u8) -> u64 {
    let mut val = 0u64;
    for i in 0..size as usize {
        val |= (data[offset + i] as u64) << (i * 8);
    }
    val
}
