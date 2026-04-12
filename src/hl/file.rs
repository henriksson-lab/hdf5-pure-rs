use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v1::BTreeV1Node;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::link::LinkMessage;
use crate::format::object_header::{self, RawMessage};
use crate::format::superblock::Superblock;
use crate::format::symbol_table::SymbolTableNode;
use crate::hl::group::Group;
use crate::io::reader::HdfReader;

/// Represents the type of an HDF5 object as determined by its object header messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Group,
    Dataset,
    NamedDatatype,
    Unknown,
}

/// Internal state of an open HDF5 file.
pub(crate) struct FileInner<R: Read + Seek> {
    pub reader: HdfReader<R>,
    pub superblock: Superblock,
}

/// An open HDF5 file.
pub struct File {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    superblock: Superblock,
}

impl File {
    /// Open an HDF5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = fs::File::open(path.as_ref()).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("failed to open {}: {e}", path.as_ref().display()),
            ))
        })?;

        let buf = BufReader::new(f);
        let mut reader = HdfReader::new(buf);

        let superblock = Superblock::read(&mut reader)?;

        let inner = Arc::new(Mutex::new(FileInner { reader, superblock: superblock.clone() }));

        Ok(File { inner, superblock })
    }

    /// Get the superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Get the root group.
    pub fn root_group(&self) -> Result<Group> {
        Group::open(self.inner.clone(), "/", self.superblock.root_addr)
    }

    /// List all member names in the root group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        self.root_group()?.member_names()
    }

    /// Open a group by path (starting from root).
    pub fn group(&self, path: &str) -> Result<Group> {
        let root = self.root_group()?;
        if path == "/" || path.is_empty() {
            return Ok(root);
        }

        let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        let mut current = root;
        for part in parts {
            if part.is_empty() {
                continue;
            }
            current = current.open_group(part)?;
        }
        Ok(current)
    }

    /// List attribute names on the root group.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        crate::hl::attribute::attr_names(&self.inner, self.superblock.root_addr)
    }

    /// Get an attribute by name on the root group.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.superblock.root_addr, name)
    }

    /// Open a dataset by path from the root group.
    pub fn dataset(&self, path: &str) -> Result<crate::hl::dataset::Dataset> {
        let path = path.trim_start_matches('/');
        if let Some(last_slash) = path.rfind('/') {
            let group_path = &path[..last_slash];
            let ds_name = &path[last_slash + 1..];
            let group = self.group(group_path)?;
            group.open_dataset(ds_name)
        } else {
            self.root_group()?.open_dataset(path)
        }
    }
}

/// Determine object type from an object header's messages.
pub(crate) fn object_type_from_messages(messages: &[RawMessage]) -> ObjectType {
    let has_dataspace = messages.iter().any(|m| m.msg_type == object_header::MSG_DATASPACE);
    let has_layout = messages.iter().any(|m| m.msg_type == object_header::MSG_LAYOUT);
    let has_datatype = messages.iter().any(|m| m.msg_type == object_header::MSG_DATATYPE);
    let has_stab = messages.iter().any(|m| m.msg_type == object_header::MSG_SYMBOL_TABLE);
    let has_link = messages.iter().any(|m| m.msg_type == object_header::MSG_LINK);
    let has_link_info = messages.iter().any(|m| m.msg_type == object_header::MSG_LINK_INFO);

    if has_layout || (has_dataspace && has_datatype && !has_stab && !has_link && !has_link_info) {
        ObjectType::Dataset
    } else if has_stab || has_link || has_link_info {
        ObjectType::Group
    } else if has_datatype && !has_dataspace {
        ObjectType::NamedDatatype
    } else if messages.is_empty() {
        // Empty object header -- likely an empty group (v2 format)
        ObjectType::Group
    } else {
        ObjectType::Unknown
    }
}

/// Collect group member names from a v1 symbol table.
pub(crate) fn collect_v1_group_members<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    btree_addr: u64,
    heap_addr: u64,
) -> Result<Vec<(String, u64)>> {
    let heap = LocalHeap::read_at(reader, heap_addr)?;
    let snod_addrs = BTreeV1Node::collect_symbol_table_addrs(reader, btree_addr)?;

    let mut members = Vec::new();

    for snod_addr in snod_addrs {
        let snod = SymbolTableNode::read_at(reader, snod_addr)?;
        for entry in &snod.entries {
            if let Some(name) = heap.get_string(entry.name_offset as usize) {
                if !name.is_empty() {
                    members.push((name, entry.obj_header_addr));
                }
            }
        }
    }

    Ok(members)
}

/// Collect group member names from v2 link messages in an object header.
pub(crate) fn collect_v2_link_members(
    messages: &[RawMessage],
    sizeof_addr: u8,
) -> Vec<(String, u64)> {
    let mut members = Vec::new();

    for msg in messages {
        if msg.msg_type == object_header::MSG_LINK {
            if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                // Include all link types; use hard_link_addr or 0 for soft/external
                let addr = link.hard_link_addr.unwrap_or(0);
                members.push((link.name, addr));
            }
        }
    }

    members
}
