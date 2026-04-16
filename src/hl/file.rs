use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v1::BTreeV1Node;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::link::{LinkMessage, LinkType};
use crate::format::object_header::{self, RawMessage};
use crate::format::superblock::Superblock;
use crate::format::symbol_table::SymbolTableNode;
use crate::hl::dataset::Dataset;
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
    pub path: Option<PathBuf>,
}

/// An open HDF5 file.
pub struct File {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    superblock: Superblock,
}

impl File {
    const MAX_SOFT_LINK_TRAVERSALS: usize = 40;

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

        let inner = Arc::new(Mutex::new(FileInner {
            reader,
            superblock: superblock.clone(),
            path: Some(path.as_ref().to_path_buf()),
        }));

        Ok(File { inner, superblock })
    }

    /// Get the superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    #[cfg(feature = "tracehash")]
    pub(crate) fn inner_arc(&self) -> Arc<Mutex<FileInner<BufReader<fs::File>>>> {
        self.inner.clone()
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
        let resolved = self.resolve_path(path)?;
        if resolved.object_type != ObjectType::Group {
            return Err(Error::InvalidFormat(format!(
                "'{path}' is not a group (type: {:?})",
                resolved.object_type
            )));
        }
        Group::open(resolved.inner, &resolved.path, resolved.addr)
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
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        let resolved = self.resolve_path(path)?;
        if resolved.object_type != ObjectType::Dataset {
            return Err(Error::InvalidFormat(format!(
                "'{path}' is not a dataset (type: {:?})",
                resolved.object_type
            )));
        }
        Ok(Dataset::new(resolved.inner, &resolved.path, resolved.addr))
    }

    fn resolve_path(&self, path: &str) -> Result<ResolvedObject> {
        let mut path = canonical_path(path);
        let mut traversals = 0usize;

        'resolve: loop {
            if path == "/" {
                return Ok(ResolvedObject {
                    inner: self.inner.clone(),
                    path,
                    addr: self.superblock.root_addr,
                    object_type: ObjectType::Group,
                });
            }

            let parts: Vec<String> = path
                .trim_start_matches('/')
                .split('/')
                .filter(|part| !part.is_empty())
                .map(str::to_string)
                .collect();
            let mut current = self.root_group()?;
            let mut current_path = String::from("/");

            for (idx, part) in parts.iter().enumerate() {
                let is_last = idx + 1 == parts.len();
                let link = match current.find_link_by_name(part) {
                    Ok(link) => link,
                    Err(link_err) => {
                        if let Some((_, addr)) = current
                            .members()?
                            .into_iter()
                            .find(|(member_name, _)| member_name == part)
                        {
                            let object_type = self.object_type_at(addr)?;
                            let next_path = join_absolute_path(&current_path, part);

                            if is_last {
                                return Ok(ResolvedObject {
                                    inner: self.inner.clone(),
                                    path: next_path,
                                    addr,
                                    object_type,
                                });
                            }

                            if object_type != ObjectType::Group {
                                return Err(Error::InvalidFormat(format!(
                                    "'{next_path}' is not a group (type: {object_type:?})"
                                )));
                            }
                            current = Group::open(self.inner.clone(), &next_path, addr)?;
                            current_path = next_path;
                            continue;
                        }
                        return Err(link_err);
                    }
                };
                match link.link_type {
                    LinkType::Hard => {
                        let addr = link.hard_link_addr.ok_or_else(|| {
                            Error::InvalidFormat(format!(
                                "hard link '{}' is missing object address",
                                link.name
                            ))
                        })?;
                        let object_type = self.object_type_at(addr)?;
                        let next_path = join_absolute_path(&current_path, part);

                        if is_last {
                            return Ok(ResolvedObject {
                                inner: self.inner.clone(),
                                path: next_path,
                                addr,
                                object_type,
                            });
                        }

                        if object_type != ObjectType::Group {
                            return Err(Error::InvalidFormat(format!(
                                "'{next_path}' is not a group (type: {object_type:?})"
                            )));
                        }
                        current = Group::open(self.inner.clone(), &next_path, addr)?;
                        current_path = next_path;
                    }
                    LinkType::Soft => {
                        traversals += 1;
                        if traversals > Self::MAX_SOFT_LINK_TRAVERSALS {
                            return Err(Error::InvalidFormat(
                                "soft link traversal limit exceeded".into(),
                            ));
                        }
                        let target = link.soft_link_target.ok_or_else(|| {
                            Error::InvalidFormat(format!(
                                "soft link '{}' is missing target path",
                                link.name
                            ))
                        })?;
                        let remaining = parts[idx + 1..].join("/");
                        path = resolve_soft_target(&current_path, &target, &remaining);
                        continue 'resolve;
                    }
                    LinkType::External => {
                        let (filename, object_path) = link.external_link.ok_or_else(|| {
                            Error::InvalidFormat(format!(
                                "external link '{}' is missing target path",
                                link.name
                            ))
                        })?;
                        let remaining = parts[idx + 1..].join("/");
                        let target_path = if remaining.is_empty() {
                            canonical_path(&object_path)
                        } else {
                            canonical_path(&join_absolute_path(&object_path, &remaining))
                        };
                        let file_path = self.resolve_external_file_path(&filename)?;
                        let external_file = File::open(file_path)?;
                        return external_file.resolve_path(&target_path);
                    }
                    LinkType::UserDefined(kind) => {
                        return Err(Error::Unsupported(format!(
                            "user-defined link traversal is not supported for link type {kind}"
                        )));
                    }
                }
            }
        }
    }

    fn object_type_at(&self, addr: u64) -> Result<ObjectType> {
        let mut guard = self.inner.lock();
        let oh = object_header::ObjectHeader::read_at(&mut guard.reader, addr)?;
        Ok(object_type_from_messages(&oh.messages))
    }

    fn resolve_external_file_path(&self, filename: &str) -> Result<PathBuf> {
        let path = PathBuf::from(filename);
        if path.is_absolute() {
            return Ok(path);
        }
        let base = self
            .inner
            .lock()
            .path
            .as_ref()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .ok_or_else(|| {
                Error::InvalidFormat("relative external link has no base file path".into())
            })?;
        Ok(base.join(path))
    }
}

struct ResolvedObject {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    path: String,
    addr: u64,
    object_type: ObjectType,
}

fn canonical_path(path: &str) -> String {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", parts.join("/"))
    }
}

fn join_absolute_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{child}")
    } else {
        format!("{parent}/{child}")
    }
}

fn resolve_soft_target(parent: &str, target: &str, remaining: &str) -> String {
    let base = if target.starts_with('/') {
        canonical_path(target)
    } else {
        canonical_path(&join_absolute_path(parent, target))
    };
    if remaining.is_empty() {
        base
    } else {
        canonical_path(&join_absolute_path(&base, remaining))
    }
}

/// Determine object type from an object header's messages.
pub(crate) fn object_type_from_messages(messages: &[RawMessage]) -> ObjectType {
    let has_dataspace = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_DATASPACE);
    let has_layout = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LAYOUT);
    let has_datatype = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_DATATYPE);
    let has_stab = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_SYMBOL_TABLE);
    let has_link = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LINK);
    let has_link_info = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LINK_INFO);

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
            let name = heap.get_string(entry.name_offset as usize)?;
            if !name.is_empty() {
                members.push((name, entry.obj_header_addr));
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
