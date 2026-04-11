/// File creation properties.
#[derive(Debug, Clone)]
pub struct FileCreate {
    /// Superblock version.
    pub superblock_version: u8,
    /// Size of file addresses in bytes.
    pub sizeof_addr: u8,
    /// Size of file lengths in bytes.
    pub sizeof_size: u8,
    /// Symbol table leaf node K value (v0/v1 only).
    pub sym_leaf_k: u16,
    /// B-tree internal node K value (v0/v1 only).
    pub btree_k: u16,
    /// Chunk B-tree K value.
    pub chunk_btree_k: u16,
}

impl FileCreate {
    /// Extract file creation properties from a File.
    pub fn from_file(f: &crate::hl::file::File) -> Self {
        let sb = f.superblock();
        Self {
            superblock_version: sb.version,
            sizeof_addr: sb.sizeof_addr,
            sizeof_size: sb.sizeof_size,
            sym_leaf_k: sb.sym_leaf_k,
            btree_k: sb.snode_btree_k,
            chunk_btree_k: sb.chunk_btree_k,
        }
    }
}
