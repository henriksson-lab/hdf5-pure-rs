use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};

use crate::engine::allocator::FileAllocator;
use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::object_header::*;
use crate::format::superblock::Superblock;
use crate::io::reader::UNDEF_ADDR;

/// A writable HDF5 file under construction.
pub struct HdfFileWriter<W: Write + Seek> {
    writer: W,
    allocator: FileAllocator,
    sizeof_addr: u8,
    sizeof_size: u8,
    /// Map of group path -> object header address.
    groups: HashMap<String, u64>,
    /// Pending links: (parent_path, child_name, child_addr).
    links: Vec<(String, String, u64)>,
    /// Pending pre-encoded attribute messages for the root group.
    pending_root_attrs: Vec<(u16, Vec<u8>)>,
    /// Pending root attributes added through the typed writer API.
    pending_root_attr_specs: Vec<OwnedAttrSpec>,
    /// Pre-encoded link messages (for soft/external links): (parent_path, encoded_link_msg).
    special_links: Vec<(String, Vec<u8>)>,
}

/// Describes an attribute to attach.
pub struct AttrSpec<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub dtype: DtypeSpec,
    pub data: &'a [u8],
}

#[derive(Clone)]
struct OwnedAttrSpec {
    name: String,
    shape: Vec<u64>,
    dtype: DtypeSpec,
    data: Vec<u8>,
}

impl OwnedAttrSpec {
    fn as_attr_spec(&self) -> AttrSpec<'_> {
        AttrSpec {
            name: &self.name,
            shape: &self.shape,
            dtype: self.dtype.clone(),
            data: &self.data,
        }
    }
}

/// Describes a dataset to create.
pub struct DatasetSpec<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub dtype: DtypeSpec,
    pub data: &'a [u8],
}

/// Describes the dataset fill-value message to write.
#[derive(Debug, Clone, Copy)]
pub struct FillValueSpec<'a> {
    pub alloc_time: u8,
    pub fill_time: u8,
    pub value: Option<&'a [u8]>,
}

impl<'a> FillValueSpec<'a> {
    pub fn undefined(alloc_time: u8, fill_time: u8) -> Self {
        Self {
            alloc_time,
            fill_time,
            value: None,
        }
    }

    pub fn with_value(alloc_time: u8, fill_time: u8, value: &'a [u8]) -> Self {
        Self {
            alloc_time,
            fill_time,
            value: Some(value),
        }
    }
}

/// Describes a compound datatype field.
#[derive(Debug, Clone)]
pub struct CompoundFieldSpec {
    pub name: String,
    pub offset: u32,
    pub dtype: DtypeSpec,
}

/// Describes a datatype.
#[derive(Debug, Clone)]
pub enum DtypeSpec {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    FixedAsciiString {
        len: u32,
        padding: u8,
    },
    FixedUtf8String {
        len: u32,
        padding: u8,
    },
    VarLenUtf8String,
    Compound {
        size: u32,
        fields: Vec<CompoundFieldSpec>,
    },
    Enum {
        base: Box<DtypeSpec>,
        members: Vec<(String, u64)>,
    },
    Opaque {
        size: u32,
        tag: String,
    },
    Array {
        dims: Vec<u32>,
        base: Box<DtypeSpec>,
    },
}

impl DtypeSpec {
    pub fn size(&self) -> u32 {
        match self {
            DtypeSpec::F64 | DtypeSpec::I64 | DtypeSpec::U64 => 8,
            DtypeSpec::F32 | DtypeSpec::I32 | DtypeSpec::U32 => 4,
            DtypeSpec::I16 | DtypeSpec::U16 => 2,
            DtypeSpec::I8 | DtypeSpec::U8 => 1,
            DtypeSpec::FixedAsciiString { len, .. } | DtypeSpec::FixedUtf8String { len, .. } => {
                *len
            }
            DtypeSpec::VarLenUtf8String => 16,
            DtypeSpec::Compound { size, .. } => *size,
            DtypeSpec::Enum { base, .. } => base.size(),
            DtypeSpec::Opaque { size, .. } => *size,
            DtypeSpec::Array { dims, base } => dims
                .iter()
                .try_fold(base.size(), |acc, dim| acc.checked_mul(*dim))
                .unwrap_or(u32::MAX),
        }
    }

    /// Encode as HDF5 datatype message bytes.
    pub fn encode(&self) -> Vec<u8> {
        self.encode_with_padding(true)
    }

    fn encode_embedded(&self) -> Vec<u8> {
        self.encode_with_padding(false)
    }

    fn encode_with_padding(&self, pad_top_level: bool) -> Vec<u8> {
        let mut buf = match self {
            DtypeSpec::F32 | DtypeSpec::F64 => self.encode_floating_point(),
            DtypeSpec::FixedAsciiString { len, padding } => {
                Self::encode_fixed_string(*len, *padding, false)
            }
            DtypeSpec::FixedUtf8String { len, padding } => {
                Self::encode_fixed_string(*len, *padding, true)
            }
            DtypeSpec::VarLenUtf8String => Self::encode_vlen_utf8_string(),
            DtypeSpec::Compound { size, fields } => Self::encode_compound(*size, fields),
            DtypeSpec::Enum { base, members } => Self::encode_enum(base, members),
            DtypeSpec::Opaque { size, tag } => Self::encode_opaque(*size, tag),
            DtypeSpec::Array { dims, base } => Self::encode_array(self.size(), dims, base),
            _ => self.encode_fixed_point(),
        };

        if pad_top_level
            && matches!(
                self,
                DtypeSpec::Compound { .. }
                    | DtypeSpec::Enum { .. }
                    | DtypeSpec::Opaque { .. }
                    | DtypeSpec::Array { .. }
                    | DtypeSpec::VarLenUtf8String
            )
        {
            while buf.len() % 8 != 0 {
                buf.push(0);
            }
        }

        buf
    }

    fn encode_floating_point(&self) -> Vec<u8> {
        let size = self.size();
        let mut buf = Vec::new();
        let class_and_version = 0x11u8;
        buf.push(class_and_version);
        if size == 4 {
            buf.extend_from_slice(&[0x20, 31, 0x00]);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.extend_from_slice(&32u16.to_le_bytes());
            buf.push(23);
            buf.push(8);
            buf.push(0);
            buf.push(23);
            buf.extend_from_slice(&127u32.to_le_bytes());
        } else {
            buf.extend_from_slice(&[0x20, 63, 0x00]);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.extend_from_slice(&64u16.to_le_bytes());
            buf.push(52);
            buf.push(11);
            buf.push(0);
            buf.push(52);
            buf.extend_from_slice(&1023u32.to_le_bytes());
        }
        buf
    }

    fn encode_fixed_string(len: u32, padding: u8, utf8: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x13);
        buf.push((padding & 0x0f) | if utf8 { 0x10 } else { 0x00 });
        buf.extend_from_slice(&[0x00, 0x00]);
        buf.extend_from_slice(&len.to_le_bytes());
        buf
    }

    fn encode_vlen_utf8_string() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x19);
        buf.extend_from_slice(&[0x01, 0x01, 0x00]);
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&[
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        buf
    }

    fn encode_compound(size: u32, fields: &[CompoundFieldSpec]) -> Vec<u8> {
        let mut buf = Self::encode_compound_header(size, fields.len());
        for field in fields {
            Self::encode_compound_field(&mut buf, field);
        }
        buf
    }

    fn encode_compound_header(size: u32, field_count: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x16);
        buf.push((field_count & 0xff) as u8);
        buf.push(((field_count >> 8) & 0xff) as u8);
        buf.push(0);
        buf.extend_from_slice(&size.to_le_bytes());
        buf
    }

    fn encode_compound_field(buf: &mut Vec<u8>, field: &CompoundFieldSpec) {
        let name_start = buf.len();
        buf.extend_from_slice(field.name.as_bytes());
        buf.push(0);
        let padded_name_len = (buf.len() - name_start + 7) & !7;
        while buf.len() < name_start + padded_name_len {
            buf.push(0);
        }
        buf.extend_from_slice(&field.offset.to_le_bytes());
        buf.extend_from_slice(&[0; 28]);
        buf.extend_from_slice(&field.dtype.encode_embedded());
    }

    fn encode_enum(base: &DtypeSpec, members: &[(String, u64)]) -> Vec<u8> {
        let base_bytes = base.encode_embedded();
        let base_size = base.size();
        let mut buf = Self::encode_enum_header(base_size, members.len(), &base_bytes);
        Self::encode_enum_names(&mut buf, members);
        Self::encode_enum_values(&mut buf, base_size as usize, members);
        buf
    }

    fn encode_enum_header(base_size: u32, member_count: usize, base_bytes: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x18);
        buf.push((member_count & 0xff) as u8);
        buf.push(((member_count >> 8) & 0xff) as u8);
        buf.push(0);
        buf.extend_from_slice(&base_size.to_le_bytes());
        buf.extend_from_slice(base_bytes);
        buf
    }

    fn encode_enum_names(buf: &mut Vec<u8>, members: &[(String, u64)]) {
        for (name, _) in members {
            Self::encode_padded_name(buf, name);
        }
    }

    fn encode_enum_values(buf: &mut Vec<u8>, value_size: usize, members: &[(String, u64)]) {
        for (_, value) in members {
            let encoded = value.to_le_bytes();
            buf.extend_from_slice(&encoded[..value_size.min(encoded.len())]);
            if value_size > encoded.len() {
                buf.resize(buf.len() + (value_size - encoded.len()), 0);
            }
        }
    }

    fn encode_opaque(size: u32, tag: &str) -> Vec<u8> {
        let mut buf = Self::encode_opaque_header(size, tag);
        buf.extend_from_slice(tag.as_bytes());
        buf.push(0);
        while buf.len() % 8 != 0 {
            buf.push(0);
        }
        buf
    }

    fn encode_opaque_header(size: u32, tag: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x15);
        let padded_tag_len = ((tag.len() + 1 + 7) & !7).min(255) as u8;
        buf.extend_from_slice(&[padded_tag_len, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf
    }

    fn encode_array(size: u32, dims: &[u32], base: &DtypeSpec) -> Vec<u8> {
        let mut buf = Self::encode_array_header(size, dims);
        buf.extend_from_slice(&base.encode_embedded());
        buf
    }

    fn encode_array_header(size: u32, dims: &[u32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x4a);
        buf.extend_from_slice(&[0x00, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.push(dims.len() as u8);
        for dim in dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf
    }

    fn encode_fixed_point(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let size = self.size();
        let is_signed = matches!(
            self,
            DtypeSpec::I8 | DtypeSpec::I16 | DtypeSpec::I32 | DtypeSpec::I64
        );
        let bf0 = if is_signed { 0x08u8 } else { 0x00u8 };
        buf.push(0x10u8);
        buf.extend_from_slice(&[bf0, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&(size as u16 * 8).to_le_bytes());
        buf
    }

    fn encode_padded_name(buf: &mut Vec<u8>, name: &str) {
        let name_start = buf.len();
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        let padded_name_len = (buf.len() - name_start + 7) & !7;
        while buf.len() < name_start + padded_name_len {
            buf.push(0);
        }
    }
}

/// Encode a dataspace message.
fn encode_dataspace(shape: &[u64]) -> Vec<u8> {
    let mut buf = Vec::new();

    if shape.is_empty() {
        // Scalar
        buf.push(2); // version 2
        buf.push(0); // ndims
        buf.push(0); // flags
        buf.push(0); // type = scalar
    } else {
        buf.push(2); // version 2
        buf.push(shape.len() as u8); // ndims
        buf.push(0); // flags (no max dims)
        buf.push(1); // type = simple
        for &d in shape {
            buf.extend_from_slice(&d.to_le_bytes());
        }
    }

    buf
}

/// Encode a dense hard link message. Libhdf5 stores ASCII dense-link
/// names without the optional character-set field when possible.
fn encode_dense_link_message(name: &str, target_addr: u64, sizeof_addr: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();
    let (size_flag, len_bytes) = if name_len < 256 {
        (0u8, 1usize)
    } else if name_len < 65536 {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };
    buf.push(1);
    buf.push(size_flag);
    match len_bytes {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&target_addr.to_le_bytes()[..sizeof_addr as usize]);
    buf
}

/// Encode a link message (v1, hard link).
fn encode_link_message(name: &str, target_addr: u64, sizeof_addr: u8) -> Vec<u8> {
    let mut buf = Vec::new();

    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    // Determine size-of-length encoding
    let (size_flag, len_bytes) = if name_len < 256 {
        (0u8, 1usize)
    } else if name_len < 65536 {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };

    // Version
    buf.push(1);

    // Flags: size_flag | has_char_encoding(0x10)
    let flags = size_flag | 0x10;
    buf.push(flags);

    // Character encoding: UTF-8 = 1
    buf.push(1);

    // Name length
    match len_bytes {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    // Name
    buf.extend_from_slice(name_bytes);

    // Hard link target address
    let addr_bytes = target_addr.to_le_bytes();
    buf.extend_from_slice(&addr_bytes[..sizeof_addr as usize]);

    buf
}

/// Encode a soft link message (v1).
fn encode_soft_link_message(name: &str, target_path: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();
    let target_bytes = target_path.as_bytes();

    let (size_flag, len_bytes) = if name_len < 256 {
        (0u8, 1)
    } else if name_len < 65536 {
        (1u8, 2)
    } else {
        (2u8, 4)
    };

    buf.push(1); // version
    buf.push(size_flag | 0x08 | 0x10); // flags: size_flag + has_link_type + has_char_encoding

    buf.push(1); // link type = soft
    buf.push(1); // char encoding = UTF-8

    match len_bytes {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }
    buf.extend_from_slice(name_bytes);

    // Soft link value: target_length(2) + target_path
    buf.extend_from_slice(&(target_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(target_bytes);

    buf
}

/// Encode an external link message (v1).
fn encode_external_link_message(name: &str, filename: &str, obj_path: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    let (size_flag, len_bytes) = if name_len < 256 {
        (0u8, 1)
    } else if name_len < 65536 {
        (1u8, 2)
    } else {
        (2u8, 4)
    };

    buf.push(1); // version
    buf.push(size_flag | 0x08 | 0x10); // flags: size_flag + has_link_type + has_char_encoding

    buf.push(64); // link type = external
    buf.push(1); // char encoding = UTF-8

    match len_bytes {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }
    buf.extend_from_slice(name_bytes);

    // External link value: info_length(2) + version(1) + filename(null-term) + obj_path(null-term)
    // Version 0: no flags byte
    let info_len = 1 + filename.len() + 1 + obj_path.len() + 1;
    buf.extend_from_slice(&(info_len as u16).to_le_bytes());
    buf.push(0); // ext version = 0 (no flags byte)
    buf.extend_from_slice(filename.as_bytes());
    buf.push(0); // null terminator
    buf.extend_from_slice(obj_path.as_bytes());
    buf.push(0); // null terminator

    buf
}

/// Encode a data layout message (v3, contiguous).
fn encode_contiguous_layout(
    data_addr: u64,
    data_size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(3); // version 3
    buf.push(1); // layout class = contiguous

    let addr_bytes = data_addr.to_le_bytes();
    buf.extend_from_slice(&addr_bytes[..sizeof_addr as usize]);

    let size_bytes = data_size.to_le_bytes();
    buf.extend_from_slice(&size_bytes[..sizeof_size as usize]);

    buf
}

/// Encode a data layout message (v3, chunked).
fn encode_chunked_layout_v3(
    btree_addr: u64,
    chunk_dims: &[u64],
    element_size: u32,
    sizeof_addr: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(3); // version 3
    buf.push(2); // layout class = chunked

    // ndims = chunk_dims.len() + 1 (extra dim for element size)
    let ndims = chunk_dims.len() + 1;
    buf.push(ndims as u8);

    // B-tree address
    let addr_bytes = btree_addr.to_le_bytes();
    buf.extend_from_slice(&addr_bytes[..sizeof_addr as usize]);

    // Chunk dimensions (each 4 bytes) + element size as last dim
    for &d in chunk_dims {
        buf.extend_from_slice(&(d as u32).to_le_bytes());
    }
    buf.extend_from_slice(&element_size.to_le_bytes());

    buf
}

/// Encode a filter pipeline message.
fn encode_filter_pipeline(compression_level: Option<u32>, shuffle: bool) -> Vec<u8> {
    let mut filters = Vec::new();

    if shuffle {
        filters.push((2u16, Vec::<u32>::new())); // SHUFFLE, no params
    }
    if let Some(level) = compression_level {
        filters.push((1u16, vec![level])); // DEFLATE, 1 param = level
    }

    if filters.is_empty() {
        return Vec::new();
    }

    let mut buf = Vec::new();
    buf.push(2); // version 2
    buf.push(filters.len() as u8); // number of filters

    for (id, params) in &filters {
        buf.extend_from_slice(&id.to_le_bytes()); // filter ID
                                                  // v2: skip name_length for known filter IDs (< 256)
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&(params.len() as u16).to_le_bytes()); // number of client data values
        for &p in params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
    }

    buf
}

/// Encode an attribute message (v3).
fn encode_attribute_message(name: &str, dtype: &DtypeSpec, shape: &[u64], data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    let dtype_bytes = dtype.encode();
    let ds_bytes = encode_dataspace(shape);
    let name_bytes = name.as_bytes();
    let name_with_null = name_bytes.len() + 1; // include null terminator

    buf.push(3); // version 3
    buf.push(0); // flags
    buf.extend_from_slice(&(name_with_null as u16).to_le_bytes());
    buf.extend_from_slice(&(dtype_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(ds_bytes.len() as u16).to_le_bytes());
    buf.push(0); // character encoding: ASCII

    buf.extend_from_slice(name_bytes);
    buf.push(0); // null terminator

    buf.extend_from_slice(&dtype_bytes);
    buf.extend_from_slice(&ds_bytes);
    buf.extend_from_slice(data);

    buf
}

fn encode_link_info_message(heap_addr: u64, name_btree_addr: u64, sizeof_addr: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&heap_addr.to_le_bytes()[..sizeof_addr as usize]);
    buf.extend_from_slice(&name_btree_addr.to_le_bytes()[..sizeof_addr as usize]);
    buf
}

fn encode_attr_info_message(heap_addr: u64, name_btree_addr: u64, sizeof_addr: u8) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&heap_addr.to_le_bytes()[..sizeof_addr as usize]);
    buf.extend_from_slice(&name_btree_addr.to_le_bytes()[..sizeof_addr as usize]);
    buf
}

fn dense_name_hash(name: &str) -> u32 {
    crate::format::checksum::checksum_lookup3(name.as_bytes(), 0)
}

/// Build a v2 object header from a list of messages.
fn build_v2_object_header(messages: &[(u16, &[u8])], flags: u8) -> Vec<u8> {
    // Calculate chunk 0 data size
    let mut chunk_data_size: usize = 0;
    for (_, data) in messages {
        // Message header: type(1) + size(2) + flags(1) = 4
        chunk_data_size += 4 + data.len();
    }

    // Determine chunk0 size encoding
    let (chunk0_flag, chunk0_bytes) = if chunk_data_size < 256 {
        (0u8, 1usize)
    } else if chunk_data_size < 65536 {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };

    let oh_flags = flags | chunk0_flag;

    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(b"OHDR");
    // Version
    buf.push(2);
    // Flags
    buf.push(oh_flags);

    // Optional timestamps (if HDR_STORE_TIMES)
    if oh_flags & HDR_STORE_TIMES != 0 {
        let now = 0u32; // placeholder
        buf.extend_from_slice(&now.to_le_bytes()); // atime
        buf.extend_from_slice(&now.to_le_bytes()); // mtime
        buf.extend_from_slice(&now.to_le_bytes()); // ctime
        buf.extend_from_slice(&now.to_le_bytes()); // btime
    }

    // Chunk 0 data size
    match chunk0_bytes {
        1 => buf.push(chunk_data_size as u8),
        2 => buf.extend_from_slice(&(chunk_data_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(chunk_data_size as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    // Messages
    for (msg_type, data) in messages {
        buf.push(*msg_type as u8); // type (1 byte in v2)
        buf.extend_from_slice(&(data.len() as u16).to_le_bytes()); // size
        let msg_flags = if *msg_type == MSG_GROUP_INFO { 0x01 } else { 0 };
        buf.push(msg_flags);
        buf.extend_from_slice(data);
    }

    // Checksum over everything so far
    let checksum = checksum_metadata(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());

    buf
}

fn encode_fill_value_message(fill: Option<FillValueSpec<'_>>) -> Result<Vec<u8>> {
    let Some(fill) = fill else {
        return Ok(vec![3u8, 0x09]);
    };
    if fill.alloc_time > 3 {
        return Err(Error::InvalidFormat(format!(
            "fill allocation time {} exceeds 2-bit field",
            fill.alloc_time
        )));
    }
    if fill.fill_time > 3 {
        return Err(Error::InvalidFormat(format!(
            "fill write time {} exceeds 2-bit field",
            fill.fill_time
        )));
    }

    let mut flags = fill.alloc_time | (fill.fill_time << 2);
    let mut buf = vec![3u8];
    if let Some(value) = fill.value {
        flags |= 0x20;
        buf.push(flags);
        buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        buf.extend_from_slice(value);
    } else {
        flags |= 0x10;
        buf.push(flags);
    }
    Ok(buf)
}

fn encode_global_heap_collection(objects: &[Vec<u8>], sizeof_size: u8) -> Result<Vec<u8>> {
    if sizeof_size != 8 {
        return Err(Error::Unsupported(
            "global heap writer currently requires 8-byte size fields".into(),
        ));
    }
    if objects.len() > u16::MAX as usize {
        return Err(Error::InvalidFormat(
            "too many global heap objects for vlen string dataset".into(),
        ));
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GCOL");
    buf.push(1);
    buf.extend_from_slice(&[0; 3]);
    buf.extend_from_slice(&0u64.to_le_bytes());

    for (idx, object) in objects.iter().enumerate() {
        let object_size = u64::try_from(object.len()).map_err(|_| {
            Error::InvalidFormat("global heap object length does not fit in u64".into())
        })?;
        let padded_size = object_size
            .checked_add(7)
            .map(|size| size & !7)
            .ok_or_else(|| Error::InvalidFormat("global heap object size overflow".into()))?;
        let padded_len = usize::try_from(padded_size).map_err(|_| {
            Error::InvalidFormat("global heap padded object length does not fit in usize".into())
        })?;

        buf.extend_from_slice(&((idx + 1) as u16).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(&object_size.to_le_bytes());
        buf.extend_from_slice(object);
        buf.resize(buf.len() + (padded_len - object.len()), 0);
    }

    let min_collection_size = 4096usize;
    let needed_with_free_header = buf
        .len()
        .checked_add(16)
        .ok_or_else(|| Error::InvalidFormat("global heap collection size overflow".into()))?;
    let target_size = needed_with_free_header
        .max(min_collection_size)
        .checked_add(4095)
        .map(|size| size & !4095)
        .ok_or_else(|| Error::InvalidFormat("global heap collection size overflow".into()))?;
    let free_size = target_size
        .checked_sub(buf.len() + 16)
        .ok_or_else(|| Error::InvalidFormat("global heap free object size overflow".into()))?;
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&[0; 4]);
    buf.extend_from_slice(&(free_size as u64).to_le_bytes());
    buf.resize(target_size, 0);

    let collection_size = u64::try_from(buf.len()).map_err(|_| {
        Error::InvalidFormat("global heap collection length does not fit in u64".into())
    })?;
    buf[8..16].copy_from_slice(&collection_size.to_le_bytes());
    Ok(buf)
}

fn shape_element_count(shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("dataset shape element count overflow".into()))
    })
}

impl<W: Write + Seek> HdfFileWriter<W> {
    /// Create a new HDF5 file writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            allocator: FileAllocator::new(0),
            sizeof_addr: 8,
            sizeof_size: 8,
            groups: HashMap::new(),
            links: Vec::new(),
            pending_root_attrs: Vec::new(),
            pending_root_attr_specs: Vec::new(),
            special_links: Vec::new(),
        }
    }

    /// Write the initial file structure: superblock placeholder.
    /// Call finalize() when done to write the superblock with correct EOF.
    pub fn begin(&mut self) -> Result<()> {
        // Reserve space for superblock (v2 with 8-byte addresses: 48 bytes)
        let sb_size = 12 + 4 * self.sizeof_addr as u64 + 4;
        self.allocator = FileAllocator::new(sb_size);

        // Write placeholder bytes for superblock
        let zeros = vec![0u8; sb_size as usize];
        self.write_at(0, &zeros)?;

        Ok(())
    }

    /// Write an empty group object header (will be rewritten with links in finalize).
    fn write_group_object_header(&mut self, extra_messages: &[(u16, &[u8])]) -> Result<u64> {
        let messages: Vec<(u16, &[u8])> = extra_messages.to_vec();
        let oh_bytes = build_v2_object_header(&messages, 0);
        let oh_addr = self.allocator.allocate(oh_bytes.len() as u64, 8);
        self.write_at(oh_addr, &oh_bytes)?;
        Ok(oh_addr)
    }

    /// Create the root group.
    pub fn create_root_group(&mut self) -> Result<u64> {
        let addr = self.write_group_object_header(&[])?;
        self.groups.insert("/".to_string(), addr);
        Ok(addr)
    }

    /// Create a sub-group.
    pub fn create_group(&mut self, parent: &str, name: &str) -> Result<u64> {
        let addr = self.write_group_object_header(&[])?;
        let full_path = if parent == "/" {
            format!("/{name}")
        } else {
            format!("{parent}/{name}")
        };
        self.groups.insert(full_path.clone(), addr);
        self.links
            .push((parent.to_string(), name.to_string(), addr));
        Ok(addr)
    }

    /// Create a soft link in a group.
    pub fn create_soft_link(&mut self, parent: &str, name: &str, target_path: &str) {
        let msg = encode_soft_link_message(name, target_path);
        self.special_links.push((parent.to_string(), msg));
    }

    /// Create an external link in a group.
    pub fn create_external_link(
        &mut self,
        parent: &str,
        name: &str,
        filename: &str,
        obj_path: &str,
    ) {
        let msg = encode_external_link_message(name, filename, obj_path);
        self.special_links.push((parent.to_string(), msg));
    }

    /// Create a dataset with compact storage (data embedded in the object header).
    /// Best for small datasets (< ~64KB).
    pub fn create_compact_dataset(&mut self, parent: &str, spec: &DatasetSpec) -> Result<u64> {
        self.create_compact_dataset_with_fill(parent, spec, None)
    }

    pub fn create_compact_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        let dtype_bytes = spec.dtype.encode();
        let ds_bytes = encode_dataspace(spec.shape);

        // Compact layout: version 3, class 0, size(2) + data
        let mut layout_bytes = Vec::new();
        layout_bytes.push(3); // version 3
        layout_bytes.push(0); // class = compact
        layout_bytes.extend_from_slice(&(spec.data.len() as u16).to_le_bytes());
        layout_bytes.extend_from_slice(spec.data);

        let fill_value_bytes = encode_fill_value_message(fill)?;

        let messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];

        let oh_bytes = build_v2_object_header(&messages, 0);
        let oh_addr = self.allocator.allocate(oh_bytes.len() as u64, 8);
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a dataset with contiguous storage.
    pub fn create_dataset(&mut self, parent: &str, spec: &DatasetSpec) -> Result<u64> {
        self.create_dataset_with_fill(parent, spec, None)
    }

    pub fn create_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        let dtype_bytes = spec.dtype.encode();
        let ds_bytes = encode_dataspace(spec.shape);

        // Allocate space for the data
        let data_size = spec.data.len() as u64;
        let data_addr = if data_size > 0 {
            let addr = self.allocator.allocate(data_size, 8);
            self.write_at(addr, spec.data)?;
            addr
        } else {
            UNDEF_ADDR
        };

        let layout_bytes =
            encode_contiguous_layout(data_addr, data_size, self.sizeof_addr, self.sizeof_size);

        let fill_value_bytes = encode_fill_value_message(fill)?;

        let messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];

        let oh_bytes = build_v2_object_header(&messages, 0);
        let oh_addr = self.allocator.allocate(oh_bytes.len() as u64, 8);
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a contiguous variable-length UTF-8 string dataset backed by a global heap.
    pub fn create_vlen_utf8_string_dataset(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
    ) -> Result<u64> {
        let expected_count = shape_element_count(shape)?;
        if expected_count != strings.len() as u64 {
            return Err(Error::InvalidFormat(format!(
                "vlen string data length {} does not match dataset shape element count {expected_count}",
                strings.len()
            )));
        }

        let mut heap_objects = Vec::new();
        let mut heap_indices = Vec::with_capacity(strings.len());
        for value in strings {
            if value.is_empty() {
                heap_indices.push(None);
                continue;
            }
            let mut object = value.as_bytes().to_vec();
            object.push(0);
            heap_objects.push(object);
            heap_indices.push(Some(heap_objects.len() as u32));
        }

        let heap_addr = if heap_objects.is_empty() {
            UNDEF_ADDR
        } else {
            let heap_bytes = encode_global_heap_collection(&heap_objects, self.sizeof_size)?;
            let addr = self.allocator.allocate(heap_bytes.len() as u64, 8);
            self.write_at(addr, &heap_bytes)?;
            addr
        };

        let mut data =
            Vec::with_capacity(strings.len() * DtypeSpec::VarLenUtf8String.size() as usize);
        for (value, heap_index) in strings.iter().zip(heap_indices) {
            if let Some(index) = heap_index {
                let len = u32::try_from(value.len() + 1)
                    .map_err(|_| Error::InvalidFormat("vlen string length exceeds u32".into()))?;
                data.extend_from_slice(&len.to_le_bytes());
                data.extend_from_slice(&heap_addr.to_le_bytes());
                data.extend_from_slice(&index.to_le_bytes());
            } else {
                data.extend_from_slice(&0u32.to_le_bytes());
                data.extend_from_slice(&0u64.to_le_bytes());
                data.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        let spec = DatasetSpec {
            name,
            shape,
            dtype: DtypeSpec::VarLenUtf8String,
            data: &data,
        };
        self.create_dataset(parent, &spec)
    }

    /// Create a dataset with attributes.
    pub fn create_dataset_with_attrs(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        let dtype_bytes = spec.dtype.encode();
        let ds_bytes = encode_dataspace(spec.shape);

        let data_size = spec.data.len() as u64;
        let data_addr = if data_size > 0 {
            let addr = self.allocator.allocate(data_size, 8);
            self.write_at(addr, spec.data)?;
            addr
        } else {
            UNDEF_ADDR
        };

        let layout_bytes =
            encode_contiguous_layout(data_addr, data_size, self.sizeof_addr, self.sizeof_size);
        let fill_value_bytes = encode_fill_value_message(None)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs.len() > 8 {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((
                MSG_ATTR_INFO,
                encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr),
            ));
        } else {
            for attr in attrs {
                let attr_bytes =
                    encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data);
                messages.push((MSG_ATTRIBUTE, attr_bytes));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_bytes = build_v2_object_header(&msg_refs, 0);
        let oh_addr = self.allocator.allocate(oh_bytes.len() as u64, 8);
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Add attributes to the root group (call before finalize).
    pub fn set_root_attrs(&mut self, attrs: Vec<(u16, Vec<u8>)>) {
        // Store as pending attribute messages for the root group
        // These will be included when finalize rewrites the root group
        for (msg_type, data) in attrs {
            self.pending_root_attrs.push((msg_type, data));
        }
    }

    /// Create a root group attribute from spec.
    pub fn add_root_attr(&mut self, attr: &AttrSpec) {
        self.pending_root_attr_specs.push(OwnedAttrSpec {
            name: attr.name.to_string(),
            shape: attr.shape.to_vec(),
            dtype: attr.dtype.clone(),
            data: attr.data.to_vec(),
        });
    }

    /// Create a chunked dataset with optional compression.
    pub fn create_chunked_dataset(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
    ) -> Result<u64> {
        self.create_chunked_dataset_with_fill(
            parent,
            spec,
            chunk_dims,
            compression_level,
            shuffle,
            None,
        )
    }

    pub fn create_chunked_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        let dtype_bytes = spec.dtype.encode();
        let ds_bytes = encode_dataspace(spec.shape);
        let element_size = spec.dtype.size() as usize;
        let ndims = spec.shape.len();

        // Split data into chunks, apply filters, write each chunk
        let chunk_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();
        let chunk_raw_bytes = chunk_elements * element_size;

        // Calculate number of chunks per dimension
        let mut n_chunks_per_dim = Vec::with_capacity(ndims);
        for i in 0..ndims {
            n_chunks_per_dim.push(
                ((spec.shape[i] as usize) + chunk_dims[i] as usize - 1) / chunk_dims[i] as usize,
            );
        }
        let total_chunks: usize = n_chunks_per_dim.iter().product();

        // Write each chunk and collect (coords, addr, compressed_size)
        let mut chunk_entries: Vec<(Vec<u64>, u64, u32)> = Vec::new();

        for chunk_idx in 0..total_chunks {
            // Calculate chunk coordinates
            let mut coords = vec![0u64; ndims];
            let mut rem = chunk_idx;
            for d in (0..ndims).rev() {
                coords[d] = (rem % n_chunks_per_dim[d]) as u64 * chunk_dims[d];
                rem /= n_chunks_per_dim[d];
            }

            // Extract chunk data from the source array
            let mut chunk_buf = vec![0u8; chunk_raw_bytes];
            self.extract_chunk(
                spec.data,
                spec.shape,
                &coords,
                chunk_dims,
                element_size,
                &mut chunk_buf,
            );

            // Apply filters in forward order
            let mut filtered = chunk_buf;
            if shuffle {
                filtered = crate::filters::shuffle::shuffle(&filtered, element_size)?;
            }
            if let Some(level) = compression_level {
                filtered = crate::filters::deflate::compress(&filtered, level)?;
            }

            let compressed_size = filtered.len() as u32;
            let addr = self.allocator.allocate(filtered.len() as u64, 1);
            self.write_at(addr, &filtered)?;

            chunk_entries.push((coords, addr, compressed_size));
        }

        // Write v1 B-tree for chunk index
        let btree_addr = self.write_chunk_btree_v1(&chunk_entries, ndims, element_size as u32)?;

        // Encode layout message (v3 chunked)
        let layout_bytes = encode_chunked_layout_v3(
            btree_addr,
            chunk_dims,
            element_size as u32,
            self.sizeof_addr,
        );

        // Encode filter pipeline message
        let pipeline_bytes = encode_filter_pipeline(compression_level, shuffle);

        let fill_value_bytes = encode_fill_value_message(fill)?;

        let mut messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];
        if !pipeline_bytes.is_empty() {
            messages.push((MSG_FILTER_PIPELINE, &pipeline_bytes));
        }

        let oh_bytes = build_v2_object_header(&messages, 0);
        let oh_addr = self.allocator.allocate(oh_bytes.len() as u64, 8);
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Extract chunk data from a flat array.
    fn extract_chunk(
        &self,
        data: &[u8],
        shape: &[u64],
        chunk_start: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        out: &mut [u8],
    ) {
        let ndims = shape.len();
        if ndims == 1 {
            // Fast path for 1D
            let start = chunk_start[0] as usize;
            let chunk_len = chunk_dims[0] as usize;
            let data_len = shape[0] as usize;
            let copy_len = chunk_len.min(data_len - start);
            let src_start = start * element_size;
            let src_end = src_start + copy_len * element_size;
            out[..copy_len * element_size].copy_from_slice(&data[src_start..src_end]);
        } else {
            // General N-D: iterate over elements
            let chunk_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();
            let mut idx = vec![0usize; ndims];

            for elem in 0..chunk_elements {
                // Convert linear index within chunk to N-D
                let mut rem = elem;
                for d in (0..ndims).rev() {
                    idx[d] = rem % chunk_dims[d] as usize;
                    rem /= chunk_dims[d] as usize;
                }

                // Global position
                let mut in_bounds = true;
                let mut src_linear = 0usize;
                let mut stride = 1usize;
                for d in (0..ndims).rev() {
                    let global = chunk_start[d] as usize + idx[d];
                    if global >= shape[d] as usize {
                        in_bounds = false;
                        break;
                    }
                    src_linear += global * stride;
                    stride *= shape[d] as usize;
                }

                if in_bounds {
                    let src_offset = src_linear * element_size;
                    let dst_offset = elem * element_size;
                    if src_offset + element_size <= data.len()
                        && dst_offset + element_size <= out.len()
                    {
                        out[dst_offset..dst_offset + element_size]
                            .copy_from_slice(&data[src_offset..src_offset + element_size]);
                    }
                }
            }
        }
    }

    /// Write a v1 B-tree leaf node for chunk index.
    fn write_chunk_btree_v1(
        &mut self,
        chunks: &[(Vec<u64>, u64, u32)],
        ndims: usize,
        element_size: u32,
    ) -> Result<u64> {
        let sa = self.sizeof_addr as usize;
        let nchunks = chunks.len();

        // Key = chunk_size(4) + filter_mask(4) + (ndims+1)*8
        let key_size = 4 + 4 + (ndims + 1) * 8;

        // The C library allocates B-tree node read buffer based on max capacity (2*K+1 keys, 2*K children).
        // Default chunk btree K = 32, so max entries = 64.
        // Max node size = header(4+1+1+2+sa*2) + (2K+1)*key_size + (2K)*sa
        let btree_k: usize = 32; // HDF5_BTREE_CHUNK_IK_DEF
        let max_entries = 2 * btree_k;
        let header_size = 4 + 1 + 1 + 2 + sa * 2;
        let max_node_size = header_size + (max_entries + 1) * key_size + max_entries * sa;

        let mut buf = Vec::with_capacity(max_node_size);

        // Magic
        buf.extend_from_slice(b"TREE");
        // Type = 1 (raw data chunks)
        buf.push(1);
        // Level = 0 (leaf)
        buf.push(0);
        // Number of entries
        buf.extend_from_slice(&(nchunks as u16).to_le_bytes());
        // Left sibling = UNDEF
        let undef = UNDEF_ADDR.to_le_bytes();
        buf.extend_from_slice(&undef[..sa]);
        // Right sibling = UNDEF
        buf.extend_from_slice(&undef[..sa]);

        // Entries: key + child alternating
        for (coords, addr, compressed_size) in chunks {
            // Key: chunk_size(4) + filter_mask(4) + coords(ndims*8) + extra_dim(8)
            buf.extend_from_slice(&compressed_size.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask = 0 (all filters applied)
            for &c in coords {
                buf.extend_from_slice(&c.to_le_bytes());
            }
            buf.extend_from_slice(&0u64.to_le_bytes()); // extra dimension = 0

            // Child: chunk address
            let addr_bytes = addr.to_le_bytes();
            buf.extend_from_slice(&addr_bytes[..sa]);
        }

        // Final key (sentinel)
        buf.extend_from_slice(&0u32.to_le_bytes()); // chunk_size = 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask = 0
                                                    // Final coords = last chunk's coords (or data shape end)
        if let Some((last_coords, _, _)) = chunks.last() {
            for &c in last_coords {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        } else {
            for _ in 0..ndims {
                buf.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        // Extra dim for final key = element size (required by C library)
        buf.extend_from_slice(&(element_size as u64).to_le_bytes());

        // Pad to max node size so the C library can read the full expected buffer
        buf.resize(max_node_size, 0);

        let btree_addr = self.allocator.allocate(buf.len() as u64, 8);
        self.write_at(btree_addr, &buf)?;

        Ok(btree_addr)
    }

    fn write_dense_link_storage(&mut self, links: &[(String, u64)]) -> Result<(u64, u64)> {
        let mut payloads = Vec::with_capacity(links.len());
        let mut managed_size = 0u64;
        for (name, addr) in links {
            let link_bytes = encode_dense_link_message(name, *addr, self.sizeof_addr);
            managed_size = managed_size
                .checked_add(link_bytes.len() as u64)
                .ok_or_else(|| Error::InvalidFormat("dense link heap size overflow".into()))?;
            payloads.push((name.as_str(), link_bytes));
        }

        let (heap_addr, heap_ids) = self.write_managed_fractal_heap(&payloads, 7)?;
        let mut records = Vec::with_capacity(payloads.len());
        for ((name, _), heap_id) in payloads.iter().zip(heap_ids) {
            let mut record = Vec::with_capacity(4 + heap_id.len());
            record.extend_from_slice(&dense_name_hash(name).to_le_bytes());
            record.extend_from_slice(&heap_id);
            records.push(record);
        }
        records
            .sort_by_key(|record| u32::from_le_bytes([record[0], record[1], record[2], record[3]]));

        let btree_addr = self.write_dense_name_btree(5, &records)?;
        Ok((heap_addr, btree_addr))
    }

    fn write_dense_attribute_storage(&mut self, attrs: &[AttrSpec<'_>]) -> Result<(u64, u64)> {
        let mut payloads = Vec::with_capacity(attrs.len());
        for attr in attrs {
            let attr_bytes =
                encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data);
            payloads.push((attr.name, attr_bytes));
        }

        let (heap_addr, heap_ids) = self.write_managed_fractal_heap(&payloads, 8)?;
        let mut records = Vec::with_capacity(payloads.len());
        for (creation_order, ((name, _), heap_id)) in payloads.iter().zip(heap_ids).enumerate() {
            let mut record = Vec::with_capacity(heap_id.len() + 9);
            record.extend_from_slice(&heap_id);
            record.push(0);
            record.extend_from_slice(&(creation_order as u32).to_le_bytes());
            record.extend_from_slice(&dense_name_hash(name).to_le_bytes());
            records.push(record);
        }
        records.sort_by_key(|record| {
            let hash_pos = record.len() - 4;
            u32::from_le_bytes([
                record[hash_pos],
                record[hash_pos + 1],
                record[hash_pos + 2],
                record[hash_pos + 3],
            ])
        });

        let btree_addr = self.write_dense_name_btree(8, &records)?;
        Ok((heap_addr, btree_addr))
    }

    fn write_managed_fractal_heap(
        &mut self,
        payloads: &[(&str, Vec<u8>)],
        heap_id_len: u16,
    ) -> Result<(u64, Vec<Vec<u8>>)> {
        let offset_bytes = 4usize;
        let length_bytes = heap_id_len
            .checked_sub(1 + offset_bytes as u16)
            .ok_or_else(|| Error::InvalidFormat("managed heap ID length is too short".into()))?
            as usize;
        if length_bytes == 0 || length_bytes > 8 {
            return Err(Error::Unsupported(format!(
                "managed heap ID length {heap_id_len} leaves unsupported length byte count {length_bytes}"
            )));
        }
        let needed_block_size = 25
            + payloads
                .iter()
                .map(|(_, payload)| payload.len())
                .sum::<usize>();
        let block_size = 512usize.max(needed_block_size.next_power_of_two());
        let heap_header_len = self.minimal_fractal_heap_header_len();
        let heap_addr = self.allocator.allocate(heap_header_len as u64, 8);
        let direct_addr = self.allocator.allocate(block_size as u64, 8);

        let mut direct = Vec::with_capacity(block_size);
        direct.extend_from_slice(b"FHDB");
        direct.push(0);
        direct.extend_from_slice(&heap_addr.to_le_bytes()[..self.sizeof_addr as usize]);
        direct.extend_from_slice(&0u32.to_le_bytes());
        direct.extend_from_slice(&0u32.to_le_bytes());

        let mut heap_ids = Vec::with_capacity(payloads.len());
        for (_, payload) in payloads {
            let offset = direct.len() as u32;
            direct.extend_from_slice(payload);
            let len = payload.len() as u64;
            let max_len = if length_bytes == 8 {
                u64::MAX
            } else {
                (1u64 << (length_bytes * 8)) - 1
            };
            if len > max_len {
                return Err(Error::Unsupported(format!(
                    "dense heap payload length {len} exceeds {length_bytes}-byte managed heap ID length"
                )));
            }
            let mut heap_id = Vec::with_capacity(heap_id_len as usize);
            heap_id.push(0);
            heap_id.extend_from_slice(&offset.to_le_bytes()[..offset_bytes]);
            heap_id.extend_from_slice(&len.to_le_bytes()[..length_bytes]);
            heap_ids.push(heap_id);
        }
        direct.resize(block_size, 0);
        let checksum = checksum_metadata(&direct);
        direct[17..21].copy_from_slice(&checksum.to_le_bytes());
        self.write_at(direct_addr, &direct)?;

        let used_managed_space =
            direct.iter().rposition(|byte| *byte != 0).unwrap_or(20) as u64 + 1;
        let heap = self.encode_minimal_fractal_heap(
            heap_id_len,
            payloads.len(),
            used_managed_space,
            block_size as u64,
            direct_addr,
        );
        debug_assert_eq!(heap.len(), heap_header_len);
        self.write_at(heap_addr, &heap)?;
        Ok((heap_addr, heap_ids))
    }

    fn minimal_fractal_heap_header_len(&self) -> usize {
        4 + 1
            + 2
            + 2
            + 1
            + 4
            + self.sizeof_size as usize
            + self.sizeof_addr as usize
            + self.sizeof_size as usize
            + self.sizeof_addr as usize
            + self.sizeof_size as usize * 8
            + 2
            + self.sizeof_size as usize * 2
            + 2
            + 2
            + self.sizeof_addr as usize
            + 2
            + 4
    }

    fn encode_minimal_fractal_heap(
        &self,
        heap_id_len: u16,
        managed_nobjs: usize,
        _managed_size: u64,
        managed_alloc_size: u64,
        root_block_addr: u64,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        let undef = UNDEF_ADDR.to_le_bytes();
        let sa = self.sizeof_addr as usize;
        let ss = self.sizeof_size as usize;
        let free_space = 0u64;

        buf.extend_from_slice(b"FRHP");
        buf.push(0);
        buf.extend_from_slice(&heap_id_len.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0x02);
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&undef[..sa]);
        buf.extend_from_slice(&free_space.to_le_bytes()[..ss]);
        buf.extend_from_slice(&undef[..sa]);
        buf.extend_from_slice(&managed_alloc_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&managed_alloc_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&(managed_nobjs as u64).to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&managed_alloc_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&65536u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&32u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&root_block_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&0u16.to_le_bytes());
        let checksum = checksum_metadata(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    fn write_dense_name_btree(&mut self, tree_type: u8, records: &[Vec<u8>]) -> Result<u64> {
        let record_size = records
            .first()
            .map(|record| record.len())
            .ok_or_else(|| Error::InvalidFormat("cannot write empty dense name B-tree".into()))?;
        if records.iter().any(|record| record.len() != record_size) {
            return Err(Error::InvalidFormat(
                "dense name B-tree records have inconsistent sizes".into(),
            ));
        }
        if records.len() > u16::MAX as usize {
            return Err(Error::Unsupported(
                "dense name B-tree writer supports at most 65535 records".into(),
            ));
        }

        let node_size = 512usize.max(10 + records.len() * record_size);
        let mut leaf = Vec::with_capacity(6 + records.len() * record_size + 4);
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(tree_type);
        for record in records {
            leaf.extend_from_slice(record);
        }
        let leaf_checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&leaf_checksum.to_le_bytes());
        let root_addr = self.allocator.allocate(leaf.len() as u64, 8);
        self.write_at(root_addr, &leaf)?;

        let mut header = Vec::new();
        header.extend_from_slice(b"BTHD");
        header.push(0);
        header.push(tree_type);
        header.extend_from_slice(&(node_size as u32).to_le_bytes());
        header.extend_from_slice(&(record_size as u16).to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.push(100);
        header.push(40);
        header.extend_from_slice(&root_addr.to_le_bytes()[..self.sizeof_addr as usize]);
        header.extend_from_slice(&(records.len() as u16).to_le_bytes());
        header
            .extend_from_slice(&(records.len() as u64).to_le_bytes()[..self.sizeof_size as usize]);
        let checksum = checksum_metadata(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        let header_addr = self.allocator.allocate(header.len() as u64, 8);
        self.write_at(header_addr, &header)?;
        Ok(header_addr)
    }

    /// Finalize the file: update root group with links and write superblock.
    pub fn finalize(&mut self) -> Result<()> {
        // Sort groups by depth (deepest first) so child groups are written
        // before their parents, and parent links point to correct addresses.
        let mut group_paths: Vec<String> = self.groups.keys().cloned().collect();
        group_paths.sort_by(|a, b| {
            let depth_a = a.split('/').filter(|s| !s.is_empty()).count();
            let depth_b = b.split('/').filter(|s| !s.is_empty()).count();
            depth_b.cmp(&depth_a) // deepest first
        });

        for path in group_paths {
            // Collect links for this group, using CURRENT addresses
            let group_links: Vec<(String, u64)> = self
                .links
                .iter()
                .filter(|(parent, _, _)| *parent == path)
                .map(|(_, name, addr)| {
                    // If the target is a group, use its updated address
                    let target_path = if path == "/" {
                        format!("/{name}")
                    } else {
                        format!("{path}/{name}")
                    };
                    let current_addr = self.groups.get(&target_path).copied().unwrap_or(*addr);
                    (name.clone(), current_addr)
                })
                .collect();

            if group_links.is_empty() && path != "/" {
                continue;
            }

            // Build messages: link info + link messages. Groups above the
            // compact threshold use dense link storage, backed by a v2 B-tree
            // name index and heap IDs that point directly at link payloads.
            let mut messages: Vec<(u16, Vec<u8>)> = Vec::new();

            if group_links.len() > 8 {
                let (heap_addr, btree_addr) = self.write_dense_link_storage(&group_links)?;
                messages.push((MSG_GROUP_INFO, vec![0, 0]));
                messages.push((
                    MSG_LINK_INFO,
                    encode_link_info_message(heap_addr, btree_addr, self.sizeof_addr),
                ));
            } else {
                messages.push((MSG_GROUP_INFO, vec![0, 0]));
                let link_info = encode_link_info_message(UNDEF_ADDR, UNDEF_ADDR, self.sizeof_addr);
                messages.push((MSG_LINK_INFO, link_info));

                for (name, addr) in &group_links {
                    let link_bytes = encode_link_message(name, *addr, self.sizeof_addr);
                    messages.push((MSG_LINK, link_bytes));
                }
            }

            // Add special links (soft/external) for this group
            for (parent, link_data) in &self.special_links {
                if *parent == path {
                    messages.push((MSG_LINK, link_data.clone()));
                }
            }

            // Add pending root attributes. Attributes added through the typed
            // API can spill to dense storage; pre-encoded messages remain compact.
            if path == "/" {
                for (msg_type, attr_data) in &self.pending_root_attrs {
                    messages.push((*msg_type, attr_data.clone()));
                }

                if self.pending_root_attr_specs.len() > 8 {
                    let owned_attrs = self.pending_root_attr_specs.clone();
                    let attr_specs: Vec<AttrSpec<'_>> = owned_attrs
                        .iter()
                        .map(OwnedAttrSpec::as_attr_spec)
                        .collect();
                    let (heap_addr, btree_addr) =
                        self.write_dense_attribute_storage(&attr_specs)?;
                    messages.push((
                        MSG_ATTR_INFO,
                        encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr),
                    ));
                } else {
                    for attr in &self.pending_root_attr_specs {
                        let attr_bytes = encode_attribute_message(
                            &attr.name,
                            &attr.dtype,
                            &attr.shape,
                            &attr.data,
                        );
                        messages.push((MSG_ATTRIBUTE, attr_bytes));
                    }
                }
            }

            let msg_refs: Vec<(u16, &[u8])> =
                messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();

            let oh_bytes = build_v2_object_header(&msg_refs, 0);
            let oh_addr = self.allocator.allocate(oh_bytes.len() as u64, 8);
            self.write_at(oh_addr, &oh_bytes)?;

            // Update the group address
            self.groups.insert(path.clone(), oh_addr);
        }

        // Write superblock
        let root_addr = *self
            .groups
            .get("/")
            .ok_or_else(|| Error::Other("no root group".into()))?;
        let eof = self.allocator.eof();

        let sb = Superblock {
            version: 2,
            sizeof_addr: self.sizeof_addr,
            sizeof_size: self.sizeof_size,
            status_flags: 0,
            base_addr: 0,
            ext_addr: UNDEF_ADDR,
            eof_addr: eof,
            root_addr,
            ..Default::default()
        };

        let mut sb_bytes = Vec::new();
        sb.write_v2(&mut sb_bytes);
        self.write_at(0, &sb_bytes)?;

        self.writer.flush()?;

        Ok(())
    }

    fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.writer.seek(SeekFrom::Start(offset))?;
        self.writer.write_all(data)?;
        Ok(())
    }
}
