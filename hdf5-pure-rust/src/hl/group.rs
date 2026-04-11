use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v2;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::link::{LinkMessage, LinkType};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::object_header::{self, ObjectHeader};
use crate::hl::dataset::Dataset;
use crate::hl::file::{
    collect_v1_group_members, collect_v2_link_members, object_type_from_messages, FileInner,
    ObjectType,
};

/// An HDF5 group.
pub struct Group {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    name: String,
    addr: u64,
}

impl Group {
    pub(crate) fn open(
        inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
        name: &str,
        addr: u64,
    ) -> Result<Self> {
        Ok(Self {
            inner,
            name: name.to_string(),
            addr,
        })
    }

    /// Get the group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the object header address.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// List all member names in this group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        let members = self.members()?;
        Ok(members.into_iter().map(|(name, _)| name).collect())
    }

    /// List all members as (name, object_header_addr) pairs.
    pub fn members(&self) -> Result<Vec<(String, u64)>> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        // Check for v1 symbol table message
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                return collect_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                );
            }
        }

        // V2: collect from link messages
        let members = collect_v2_link_members(&oh.messages, sizeof_addr);
        if !members.is_empty() {
            return Ok(members);
        }

        // V2 dense storage: link info message with fractal heap + v2 B-tree
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    return Self::read_dense_links(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                    );
                }
            }
        }

        Ok(Vec::new())
    }

    /// Read dense links from fractal heap + v2 B-tree.
    fn read_dense_links<R: std::io::Read + std::io::Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        sizeof_addr: u8,
    ) -> Result<Vec<(String, u64)>> {
        let heap = FractalHeapHeader::read_at(reader, link_info.fractal_heap_addr)?;

        // Read all records from the name B-tree
        let records = btree_v2::collect_all_records(reader, link_info.name_btree_addr)?;

        let mut members = Vec::new();

        for record in &records {
            // Record format for type 5 (link name): hash(4) + heap_id(heap_id_len)
            if record.len() < 4 + heap.heap_id_len as usize {
                continue;
            }

            let heap_id = &record[4..4 + heap.heap_id_len as usize];

            // Read the link message from the fractal heap
            match heap.read_managed_object(reader, heap_id) {
                Ok(link_data) => {
                    // The data in the heap is a serialized link message
                    match LinkMessage::decode(&link_data, sizeof_addr) {
                        Ok(link) => {
                            if let Some(addr) = link.hard_link_addr {
                                members.push((link.name, addr));
                            }
                        }
                        Err(e) => {
                            eprintln!("Warning: failed to decode link: {e}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: failed to read heap object: {e}");
                }
            }
        }

        Ok(members)
    }

    /// Open a sub-group by name.
    pub fn open_group(&self, name: &str) -> Result<Group> {
        let members = self.members()?;

        for (member_name, addr) in &members {
            if member_name == name {
                // Verify it's a group
                let mut guard = self.inner.lock();
                let oh = ObjectHeader::read_at(&mut guard.reader, *addr)?;
                let obj_type = object_type_from_messages(&oh.messages);
                drop(guard);

                if obj_type == ObjectType::Group {
                    let full_name = if self.name == "/" {
                        format!("/{name}")
                    } else {
                        format!("{}/{name}", self.name)
                    };
                    return Group::open(self.inner.clone(), &full_name, *addr);
                } else {
                    return Err(Error::InvalidFormat(format!(
                        "'{name}' is not a group (type: {obj_type:?})"
                    )));
                }
            }
        }

        Err(Error::InvalidFormat(format!(
            "group member '{name}' not found"
        )))
    }

    /// Get the number of members in this group.
    pub fn len(&self) -> Result<usize> {
        Ok(self.members()?.len())
    }

    /// Check if the group is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Get the type of a member object.
    pub fn member_type(&self, name: &str) -> Result<ObjectType> {
        let members = self.members()?;

        for (member_name, addr) in &members {
            if member_name == name {
                let mut guard = self.inner.lock();
                let oh = ObjectHeader::read_at(&mut guard.reader, *addr)?;
                return Ok(object_type_from_messages(&oh.messages));
            }
        }

        Err(Error::InvalidFormat(format!(
            "member '{name}' not found"
        )))
    }

    /// List attribute names.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        crate::hl::attribute::attr_names(&self.inner, self.addr)
    }

    /// Get an attribute by name.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.addr, name)
    }

    /// Get the link type of a member by name.
    pub fn link_type(&self, name: &str) -> Result<LinkType> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    if link.name == name {
                        return Ok(link.link_type);
                    }
                }
            }
        }
        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    /// Get the target path of a soft link.
    pub fn soft_link_target(&self, name: &str) -> Result<String> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    if link.name == name {
                        return link.soft_link_target.ok_or_else(|| {
                            Error::InvalidFormat(format!("'{name}' is not a soft link"))
                        });
                    }
                }
            }
        }
        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    /// Get the target (filename, object_path) of an external link.
    pub fn external_link_target(&self, name: &str) -> Result<(String, String)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    if link.name == name {
                        return link.external_link.ok_or_else(|| {
                            Error::InvalidFormat(format!("'{name}' is not an external link"))
                        });
                    }
                }
            }
        }
        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    /// Check if a named member (link) exists in this group.
    pub fn link_exists(&self, name: &str) -> Result<bool> {
        crate::hl::location::link_exists(self, name)
    }

    /// Open a dataset by name.
    pub fn open_dataset(&self, name: &str) -> Result<Dataset> {
        let members = self.members()?;

        for (member_name, addr) in &members {
            if member_name == name {
                let mut guard = self.inner.lock();
                let oh = ObjectHeader::read_at(&mut guard.reader, *addr)?;
                let obj_type = object_type_from_messages(&oh.messages);
                drop(guard);

                if obj_type == ObjectType::Dataset {
                    let full_name = if self.name == "/" {
                        format!("/{name}")
                    } else {
                        format!("{}/{name}", self.name)
                    };
                    return Ok(Dataset::new(self.inner.clone(), &full_name, *addr));
                } else {
                    return Err(Error::InvalidFormat(format!(
                        "'{name}' is not a dataset (type: {obj_type:?})"
                    )));
                }
            }
        }

        Err(Error::InvalidFormat(format!(
            "dataset '{name}' not found"
        )))
    }
}
