use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// v1 B-tree node magic: "TREE"
const BTREE_MAGIC: [u8; 4] = [b'T', b'R', b'E', b'E'];

/// B-tree node types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeType {
    /// Type 0: Group nodes (symbol table nodes).
    Group,
    /// Type 1: Raw data chunks.
    RawData,
}

/// A v1 B-tree node.
#[derive(Debug, Clone)]
pub struct BTreeV1Node {
    pub node_type: BTreeType,
    pub level: u8,
    pub entries_used: u16,
    pub left_sibling: u64,
    pub right_sibling: u64,
    /// For group B-trees: child node addresses.
    pub children: Vec<u64>,
    /// For group B-trees: keys (symbol name offsets in local heap).
    pub keys: Vec<u64>,
}

impl BTreeV1Node {
    /// Read a v1 B-tree node at the given address.
    pub fn read_at<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<Self> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != BTREE_MAGIC {
            return Err(Error::InvalidFormat("invalid v1 B-tree magic".into()));
        }

        let node_type_val = reader.read_u8()?;
        let node_type = match node_type_val {
            0 => BTreeType::Group,
            1 => BTreeType::RawData,
            _ => {
                return Err(Error::Unsupported(format!(
                    "B-tree node type {node_type_val}"
                )))
            }
        };

        let level = reader.read_u8()?;
        let entries_used = reader.read_u16()?;
        let left_sibling = reader.read_addr()?;
        let right_sibling = reader.read_addr()?;

        let mut keys = Vec::new();
        let mut children = Vec::new();

        match node_type {
            BTreeType::Group => {
                // Group B-tree: keys are (heap_offset, obj_header_addr) pairs for internal nodes,
                // or just heap_offset for comparison.
                // Actually for group B-trees, keys are just the symbol name heap offset (sizeof_size).
                // Structure: key[0], child[0], key[1], child[1], ..., key[n]
                // So there are entries_used children and entries_used+1 keys.

                for _i in 0..entries_used as usize {
                    // Key
                    let key = reader.read_length()?;
                    keys.push(key);

                    // Child pointer
                    let child = reader.read_addr()?;
                    children.push(child);
                }
                // Final key
                let final_key = reader.read_length()?;
                keys.push(final_key);
            }
            BTreeType::RawData => {
                // Raw data chunk B-tree: keys are chunk coordinates + filter mask.
                // We'll implement this in Phase 3.
                // For now just skip.
            }
        }

        Ok(Self {
            node_type,
            level,
            entries_used,
            left_sibling,
            right_sibling,
            children,
            keys,
        })
    }

    /// Collect all leaf-level symbol table node addresses from a group B-tree.
    /// Recursively traverses internal nodes to reach leaves.
    pub fn collect_symbol_table_addrs<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        btree_addr: u64,
    ) -> Result<Vec<u64>> {
        let node = Self::read_at(reader, btree_addr)?;

        if node.node_type != BTreeType::Group {
            return Err(Error::InvalidFormat(
                "expected group B-tree".into(),
            ));
        }

        if node.level == 0 {
            // Leaf node: children are symbol table node addresses
            Ok(node.children)
        } else {
            // Internal node: recurse into children
            let mut result = Vec::new();
            for &child_addr in &node.children {
                let mut child_addrs = Self::collect_symbol_table_addrs(reader, child_addr)?;
                result.append(&mut child_addrs);
            }
            Ok(result)
        }
    }
}
