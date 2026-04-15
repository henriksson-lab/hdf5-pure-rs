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
    /// Pending attribute messages for the root group.
    pending_root_attrs: Vec<(u16, Vec<u8>)>,
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

/// Describes a dataset to create.
pub struct DatasetSpec<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub dtype: DtypeSpec,
    pub data: &'a [u8],
}

/// Describes a datatype.
#[derive(Debug, Clone, Copy)]
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
}

impl DtypeSpec {
    pub fn size(&self) -> u32 {
        match self {
            DtypeSpec::F64 | DtypeSpec::I64 | DtypeSpec::U64 => 8,
            DtypeSpec::F32 | DtypeSpec::I32 | DtypeSpec::U32 => 4,
            DtypeSpec::I16 | DtypeSpec::U16 => 2,
            DtypeSpec::I8 | DtypeSpec::U8 => 1,
        }
    }

    /// Encode as HDF5 datatype message bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        match self {
            DtypeSpec::F32 | DtypeSpec::F64 => {
                let size = self.size();
                // class_and_version: version=1, class=1 (floating-point)
                let class_and_version = 0x11u8;
                // class bit fields:
                // bf0: bit0=byte order(0=LE), bits1-2=low pad, bits3-4=high pad, bits5-6=internal pad(2=mantissa norm implied 1)
                // bf1: sign bit location
                // bf2: reserved
                if size == 4 {
                    buf.push(class_and_version);
                    buf.extend_from_slice(&[0x20, 31, 0x00]); // bf0=0x20(norm=2,LE), bf1=31(sign bit), bf2=0
                    buf.extend_from_slice(&size.to_le_bytes());
                    // Properties: bit_offset(2) + bit_precision(2) + epos(1) + esize(1) + mpos(1) + msize(1) + ebias(4)
                    buf.extend_from_slice(&0u16.to_le_bytes()); // bit offset
                    buf.extend_from_slice(&32u16.to_le_bytes()); // bit precision
                    buf.push(23); // exponent position
                    buf.push(8); // exponent size
                    buf.push(0); // mantissa position
                    buf.push(23); // mantissa size
                    buf.extend_from_slice(&127u32.to_le_bytes()); // exponent bias
                } else {
                    buf.push(class_and_version);
                    buf.extend_from_slice(&[0x20, 63, 0x00]); // bf0=0x20(norm=2,LE), bf1=63(sign bit), bf2=0
                    buf.extend_from_slice(&size.to_le_bytes());
                    buf.extend_from_slice(&0u16.to_le_bytes()); // bit offset
                    buf.extend_from_slice(&64u16.to_le_bytes()); // bit precision
                    buf.push(52); // exponent position
                    buf.push(11); // exponent size
                    buf.push(0); // mantissa position
                    buf.push(52); // mantissa size
                    buf.extend_from_slice(&1023u32.to_le_bytes()); // exponent bias
                }
            }
            _ => {
                let size = self.size();
                let is_signed = matches!(
                    self,
                    DtypeSpec::I8 | DtypeSpec::I16 | DtypeSpec::I32 | DtypeSpec::I64
                );
                // class_and_version: version=1, class=0 (fixed-point)
                let class_and_version = 0x10u8; // version 1, class 0
                                                // class bit fields: byte order = LE (bit 0 = 0), sign = signed (bit 3)
                let bf0 = if is_signed { 0x08u8 } else { 0x00u8 };
                buf.push(class_and_version);
                buf.extend_from_slice(&[bf0, 0x00, 0x00]);
                buf.extend_from_slice(&size.to_le_bytes());

                // Properties: bit offset(2) + bit precision(2)
                buf.extend_from_slice(&0u16.to_le_bytes()); // bit offset
                buf.extend_from_slice(&(size as u16 * 8).to_le_bytes()); // bit precision
            }
        }

        buf
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
        8 => buf.extend_from_slice(&(chunk_data_size as u64).to_le_bytes()),
        _ => unreachable!(),
    }

    // Messages
    for (msg_type, data) in messages {
        buf.push(*msg_type as u8); // type (1 byte in v2)
        buf.extend_from_slice(&(data.len() as u16).to_le_bytes()); // size
        buf.push(0); // flags
        buf.extend_from_slice(data);
    }

    // Checksum over everything so far
    let checksum = checksum_metadata(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());

    buf
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
        let dtype_bytes = spec.dtype.encode();
        let ds_bytes = encode_dataspace(spec.shape);

        // Compact layout: version 3, class 0, size(2) + data
        let mut layout_bytes = Vec::new();
        layout_bytes.push(3); // version 3
        layout_bytes.push(0); // class = compact
        layout_bytes.extend_from_slice(&(spec.data.len() as u16).to_le_bytes());
        layout_bytes.extend_from_slice(spec.data);

        let fill_value_bytes = vec![3u8, 0x09];

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

        // Fill value message (v3, "undefined" fill value)
        let fill_value_bytes = vec![3u8, 0x09]; // version 3, flags=0x09 (never write, undefined)

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
        let fill_value_bytes = vec![3u8, 0x09];

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        for attr in attrs {
            let attr_bytes =
                encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data);
            messages.push((MSG_ATTRIBUTE, attr_bytes));
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
        let attr_bytes = encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data);
        self.pending_root_attrs.push((MSG_ATTRIBUTE, attr_bytes));
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

        // Fill value message
        let fill_value_bytes = vec![3u8, 0x09];

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

            // Build messages: link info + link messages
            let mut messages: Vec<(u16, Vec<u8>)> = Vec::new();

            // Link info message (v0): flags=0, fractal_heap=UNDEF, name_btree=UNDEF
            let mut link_info = Vec::new();
            link_info.push(0); // version
            link_info.push(0); // flags
            let undef_bytes = UNDEF_ADDR.to_le_bytes();
            link_info.extend_from_slice(&undef_bytes[..self.sizeof_addr as usize]); // fractal heap
            link_info.extend_from_slice(&undef_bytes[..self.sizeof_addr as usize]); // name btree
            messages.push((MSG_LINK_INFO, link_info));

            for (name, addr) in &group_links {
                let link_bytes = encode_link_message(name, *addr, self.sizeof_addr);
                messages.push((MSG_LINK, link_bytes));
            }

            // Add special links (soft/external) for this group
            for (parent, link_data) in &self.special_links {
                if *parent == path {
                    messages.push((MSG_LINK, link_data.clone()));
                }
            }

            // Add pending root attributes
            if path == "/" {
                for (msg_type, attr_data) in &self.pending_root_attrs {
                    messages.push((*msg_type, attr_data.clone()));
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
