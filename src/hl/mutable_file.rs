use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v2::BTreeV2Header;
use crate::format::checksum::checksum_metadata;
use crate::format::messages::data_layout::{ChunkIndexType, LayoutClass};
use crate::format::messages::filter_pipeline::{FILTER_DEFLATE, FILTER_SHUFFLE};
use crate::format::object_header::{
    self, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES,
};
use crate::format::superblock::Superblock;
use crate::hl::dataset::Dataset;
use crate::hl::file::FileInner;
use crate::hl::group::Group;
use crate::io::reader::HdfReader;

#[derive(Debug, Clone)]
struct ChunkBTreeEntry {
    coords: Vec<u64>,
    chunk_size: u32,
    filter_mask: u32,
    child_addr: u64,
}

#[derive(Debug, Clone)]
struct MutableExtensibleArrayHeader {
    class_id: u8,
    raw_element_size: usize,
    index_block_elements: u8,
    data_block_min_elements: usize,
    super_block_min_data_ptrs: usize,
    max_data_block_page_elements: usize,
    max_index_set: u64,
    realized_elements: u64,
    index_block_addr: u64,
    array_offset_size: u8,
    index_block_super_block_addrs: usize,
    checksum_pos: u64,
    data_block_count_pos: u64,
    data_block_size_pos: u64,
    max_index_set_pos: u64,
    realized_elements_pos: u64,
}

/// A mutable HDF5 file opened for read-write access.
///
/// Supports resizing chunked datasets and writing new chunks.
pub struct MutableFile {
    /// Read path (for parsing)
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    /// Write path (for modifying)
    write_handle: fs::File,
    superblock: Superblock,
    path: PathBuf,
}

impl MutableFile {
    /// Open an existing HDF5 file for read-write access.
    pub fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open for reading
        let read_file = fs::File::open(&path)?;
        let mut reader = HdfReader::new(BufReader::new(read_file));
        let superblock = Superblock::read(&mut reader)?;

        let inner = Arc::new(Mutex::new(FileInner {
            reader,
            superblock: superblock.clone(),
            path: Some(path.clone()),
        }));

        // Open separately for writing
        let write_handle = fs::OpenOptions::new().write(true).open(&path)?;

        Ok(Self {
            inner,
            write_handle,
            superblock,
            path,
        })
    }

    /// Get the root group (read-only access).
    pub fn root_group(&self) -> Result<Group> {
        Group::open(self.inner.clone(), "/", self.superblock.root_addr)
    }

    /// List member names in the root group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        self.root_group()?.member_names()
    }

    /// Open a dataset by path.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        let path_str = path.trim_start_matches('/');
        if let Some(last_slash) = path_str.rfind('/') {
            let group_path = &path_str[..last_slash];
            let ds_name = &path_str[last_slash + 1..];
            let root = self.root_group()?;
            let mut current = root;
            for part in group_path.split('/').filter(|s| !s.is_empty()) {
                current = current.open_group(part)?;
            }
            current.open_dataset(ds_name)
        } else {
            self.root_group()?.open_dataset(path_str)
        }
    }

    /// Resize a chunked dataset to new dimensions.
    ///
    /// The new dimensions must not exceed the dataset's maximum dimensions.
    /// Only the dataspace message is rewritten; no data is moved or deleted.
    /// New regions are filled with the default fill value (zeros).
    pub fn resize_dataset(&mut self, path: &str, new_dims: &[u64]) -> Result<()> {
        let ds = self.dataset(path)?;
        let info = ds.info()?;

        // Verify chunked
        if info.layout.layout_class != LayoutClass::Chunked {
            return Err(Error::InvalidFormat(
                "can only resize chunked datasets".into(),
            ));
        }

        // Verify dimension count matches
        if new_dims.len() != info.dataspace.dims.len() {
            return Err(Error::InvalidFormat(format!(
                "dimension count mismatch: dataset has {} dims, new shape has {}",
                info.dataspace.dims.len(),
                new_dims.len()
            )));
        }

        // Check max dims if present
        if let Some(ref max_dims) = info.dataspace.max_dims {
            for (i, (&new_d, &max_d)) in new_dims.iter().zip(max_dims.iter()).enumerate() {
                if max_d != u64::MAX && new_d > max_d {
                    return Err(Error::InvalidFormat(format!(
                        "dimension {i}: new size {new_d} exceeds max {max_d}"
                    )));
                }
            }
        }

        // Find the dataspace message location in the file
        let ds_addr = ds.addr();
        let (msg_data_offset, msg_data_len, oh_start, oh_check_len) =
            self.find_message_in_oh(ds_addr, object_header::MSG_DATASPACE)?;

        // Build new dataspace message bytes
        let new_ds_bytes = self.encode_dataspace_v2(new_dims, info.dataspace.max_dims.as_deref());

        if new_ds_bytes.len() != msg_data_len {
            return Err(Error::InvalidFormat(format!(
                "dataspace message size changed ({} -> {}); in-place resize not possible",
                msg_data_len,
                new_ds_bytes.len()
            )));
        }

        // Write the new dataspace message data
        self.write_handle.seek(SeekFrom::Start(msg_data_offset))?;
        self.write_handle.write_all(&new_ds_bytes)?;

        // Recompute and write the OH checksum (v2 only)
        self.rewrite_oh_checksum(oh_start, oh_check_len)?;

        self.write_handle.flush()?;

        // Reopen the read handle to pick up changes
        self.reopen_reader()?;

        Ok(())
    }

    /// Find a message of the given type in an object header.
    /// Returns (message_data_file_offset, message_data_len, oh_start, oh_checksum_data_len).
    fn find_message_in_oh(
        &self,
        oh_addr: u64,
        target_msg_type: u16,
    ) -> Result<(u64, usize, u64, usize)> {
        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let first_bytes = reader.read_bytes(4)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::InvalidFormat(
                "expected v2 object header for resize".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "resize only supported for v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start + chunk0_data_size;

        // The checksum covers oh_addr .. chunk0_data_end
        let oh_check_len = (chunk0_data_end - oh_addr) as usize;

        // Scan messages
        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos + 4 > chunk0_data_end {
                break;
            }

            let msg_type = reader.read_u8()? as u16;
            let msg_size = reader.read_u16()? as usize;
            let _msg_flags = reader.read_u8()?;

            // Check for creation index
            let has_crt_order = flags & 0x04 != 0; // HDR_ATTR_CRT_ORDER_TRACKED
            if has_crt_order {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;

            if msg_type == target_msg_type {
                return Ok((msg_data_offset, msg_size, oh_addr, oh_check_len));
            }

            reader.skip(msg_size as u64)?;
        }

        Err(Error::InvalidFormat(format!(
            "message type {target_msg_type:#06x} not found in object header"
        )))
    }

    /// Encode a v2 dataspace message with the given dims and optional max_dims.
    fn encode_dataspace_v2(&self, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
        let mut buf = Vec::new();
        let has_max = max_dims.is_some();

        buf.push(2); // version 2
        buf.push(dims.len() as u8); // ndims
        buf.push(if has_max { 0x01 } else { 0x00 }); // flags
        buf.push(1); // type = simple

        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }

        if let Some(max) = max_dims {
            for &d in max {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }

        buf
    }

    /// Recompute and rewrite the v2 OH checksum.
    fn rewrite_oh_checksum(&mut self, oh_start: u64, check_len: usize) -> Result<()> {
        // Read the OH data (minus checksum)
        let mut guard = self.inner.lock();
        guard.reader.seek(oh_start)?;
        let oh_data = guard.reader.read_bytes(check_len)?;
        drop(guard);

        let checksum = checksum_metadata(&oh_data);

        self.write_handle
            .seek(SeekFrom::Start(oh_start + check_len as u64))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;

        Ok(())
    }

    /// Reopen the read handle to pick up file changes.
    fn reopen_reader(&mut self) -> Result<()> {
        let read_file = fs::File::open(&self.path)?;
        let mut reader = HdfReader::new(BufReader::new(read_file));
        let superblock = Superblock::read(&mut reader)?;
        self.superblock = superblock.clone();
        *self.inner.lock() = FileInner {
            reader,
            superblock,
            path: Some(self.path.clone()),
        };
        Ok(())
    }

    /// Write a full uncompressed chunk and update a leaf v1 chunk B-tree.
    ///
    /// This supports chunked datasets written by this crate, where the chunk
    /// index is a padded leaf v1 B-tree with spare entry capacity.
    pub fn write_chunk(
        &mut self,
        dataset_path: &str,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> Result<()> {
        let ds = self.dataset(dataset_path)?;
        let info = ds.info()?;

        if info.layout.layout_class != LayoutClass::Chunked {
            return Err(Error::InvalidFormat(
                "write_chunk only supports chunked datasets".into(),
            ));
        }
        if info.layout.version > 3
            && !matches!(
                info.layout.chunk_index_type,
                Some(ChunkIndexType::BTreeV1)
                    | Some(ChunkIndexType::FixedArray)
                    | Some(ChunkIndexType::ExtensibleArray)
                    | Some(ChunkIndexType::BTreeV2)
            )
        {
            return Err(Error::Unsupported(
                "write_chunk currently supports only v1 B-tree, fixed-array, simple extensible-array, and simple v2 B-tree chunk indexes".into(),
            ));
        }

        let chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked layout missing chunk dims".into()))?;
        let chunk_data_dims: Vec<u64> = if chunk_dims.len() == info.dataspace.dims.len() + 1 {
            chunk_dims[..info.dataspace.dims.len()].to_vec()
        } else if chunk_dims.len() == info.dataspace.dims.len() {
            chunk_dims.clone()
        } else {
            return Err(Error::InvalidFormat(format!(
                "chunk dimension rank {} does not match dataset rank {}",
                chunk_dims.len(),
                info.dataspace.dims.len()
            )));
        };
        if chunk_coords.len() != chunk_data_dims.len() {
            return Err(Error::InvalidFormat(format!(
                "chunk coordinate rank {} does not match dataset rank {}",
                chunk_coords.len(),
                chunk_data_dims.len()
            )));
        }
        for ((idx, &coord), &chunk) in chunk_coords.iter().enumerate().zip(&chunk_data_dims) {
            if chunk == 0 || coord % chunk != 0 {
                return Err(Error::InvalidFormat(format!(
                    "chunk coordinate {idx}={coord} is not aligned to chunk size {chunk}"
                )));
            }
        }

        let element_size = info.datatype.size as usize;
        let chunk_elements = chunk_data_dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize))
            .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))?;
        let expected_len = chunk_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?;
        if data.len() != expected_len {
            return Err(Error::InvalidFormat(format!(
                "chunk data has {} bytes, expected {expected_len}",
                data.len()
            )));
        }

        let mut filtered = data.to_vec();
        if let Some(ref pipeline) = info.filter_pipeline {
            for filter in &pipeline.filters {
                match filter.id {
                    FILTER_SHUFFLE => {
                        filtered = crate::filters::shuffle::shuffle(&filtered, element_size)?;
                    }
                    FILTER_DEFLATE => {
                        let level = filter.client_data.first().copied().unwrap_or(6);
                        filtered = crate::filters::deflate::compress(&filtered, level)?;
                    }
                    other => {
                        return Err(Error::Unsupported(format!(
                            "write_chunk cannot encode filter {other}"
                        )));
                    }
                }
            }
        }

        let index_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing B-tree address".into()))?;

        let chunk_addr = self.write_handle.seek(SeekFrom::End(0))?;
        self.write_handle.write_all(&filtered)?;
        match info.layout.chunk_index_type {
            Some(ChunkIndexType::FixedArray) => {
                self.rewrite_fixed_array_chunk(
                    index_addr,
                    &info,
                    chunk_coords,
                    &chunk_data_dims,
                    filtered.len() as u64,
                    chunk_addr,
                    expected_len,
                )?;
            }
            Some(ChunkIndexType::ExtensibleArray) => {
                self.rewrite_extensible_array_chunk(
                    index_addr,
                    &info,
                    chunk_coords,
                    &chunk_data_dims,
                    filtered.len() as u64,
                    chunk_addr,
                    expected_len,
                )?;
            }
            Some(ChunkIndexType::BTreeV2) => {
                self.rewrite_btree_v2_chunk(
                    index_addr,
                    &info,
                    chunk_coords,
                    &chunk_data_dims,
                    filtered.len() as u64,
                    chunk_addr,
                    expected_len,
                )?;
            }
            _ => {
                self.rewrite_leaf_chunk_btree(
                    index_addr,
                    chunk_coords,
                    filtered.len() as u32,
                    chunk_addr,
                    element_size as u32,
                )?;
            }
        }
        self.write_handle.flush()?;
        self.reopen_reader()?;

        Ok(())
    }

    fn rewrite_extensible_array_chunk(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        let element_index =
            Self::linear_chunk_index(chunk_coords, &info.dataspace.dims, chunk_dims)?;
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info.layout.version,
                unfiltered_chunk_bytes,
                self.superblock.sizeof_size,
            )
        } else {
            0
        };

        let mut guard = self.inner.lock();
        let header = Self::read_extensible_array_header(
            &mut guard.reader,
            index_addr,
            filtered,
            chunk_size_len,
        )?;
        let element_count = usize::try_from(header.max_index_set).map_err(|_| {
            Error::InvalidFormat("extensible array element count does not fit usize".into())
        })?;
        let direct_count = header.index_block_elements as usize;
        if element_index < element_count {
            let element_pos = Self::locate_extensible_array_element(
                &mut guard.reader,
                index_addr,
                &header,
                element_index,
            )?;
            drop(guard);
            self.write_extensible_array_element(
                element_pos,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )?;
            return Ok(());
        }
        if element_index != element_count {
            return Err(Error::Unsupported(
                "write_chunk can append only the next extensible-array chunk index".into(),
            ));
        }
        if element_index < direct_count {
            let element_pos = Self::locate_extensible_array_element(
                &mut guard.reader,
                index_addr,
                &header,
                element_index,
            )?;
            drop(guard);
            self.write_extensible_array_element(
                element_pos,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )?;
            self.rewrite_extensible_array_header_counts(
                index_addr,
                &header,
                header.max_index_set + 1,
                header.realized_elements.max(header.max_index_set + 1),
                None,
            )?;
            return Ok(());
        }
        if element_index == direct_count {
            drop(guard);
            self.append_first_extensible_array_data_block(
                index_addr,
                &header,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )?;
            return Ok(());
        }

        Err(Error::Unsupported(
            "write_chunk cannot grow extensible-array indexes beyond the first data block yet"
                .into(),
        ))
    }

    fn read_extensible_array_header<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<MutableExtensibleArrayHeader> {
        reader.seek(addr)?;
        if reader.read_bytes(4)? != b"EAHD" {
            return Err(Error::InvalidFormat(
                "invalid extensible array header magic".into(),
            ));
        }
        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "extensible array header version {version}"
            )));
        }
        let class_id = reader.read_u8()?;
        let expected_class = if filtered { 1 } else { 0 };
        if class_id != expected_class {
            return Err(Error::InvalidFormat(format!(
                "extensible array class {class_id} does not match filtered={filtered}"
            )));
        }
        let raw_element_size = reader.read_u8()? as usize;
        let expected_element_size = if filtered {
            reader.sizeof_addr() as usize + chunk_size_len + 4
        } else {
            reader.sizeof_addr() as usize
        };
        if raw_element_size != expected_element_size {
            return Err(Error::InvalidFormat(format!(
                "extensible array raw element size {raw_element_size} does not match expected {expected_element_size}"
            )));
        }

        let max_elements_bits = reader.read_u8()?;
        let index_block_elements = reader.read_u8()?;
        let data_block_min_elements = reader.read_u8()? as usize;
        let super_block_min_data_ptrs = reader.read_u8()? as usize;
        let max_data_block_page_elements_bits = reader.read_u8()?;

        let _super_block_count = reader.read_length()?;
        let _super_block_size = reader.read_length()?;
        let data_block_count_pos = reader.position()?;
        let _data_block_count = reader.read_length()?;
        let data_block_size_pos = reader.position()?;
        let _data_block_size = reader.read_length()?;
        let max_index_set_pos = reader.position()?;
        let max_index_set = reader.read_length()?;
        let realized_elements_pos = reader.position()?;
        let realized_elements = reader.read_length()?;
        let index_block_addr = reader.read_addr()?;
        let checksum_pos = reader.position()?;
        let stored_checksum = reader.read_u32()?;

        let check_len = usize::try_from(checksum_pos - addr).map_err(|_| {
            Error::InvalidFormat("extensible array header checksum span is too large".into())
        })?;
        reader.seek(addr)?;
        let check_data = reader.read_bytes(check_len)?;
        let computed = checksum_metadata(&check_data);
        if stored_checksum != computed {
            return Err(Error::InvalidFormat(format!(
                "extensible array header checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}"
            )));
        }
        reader.seek(checksum_pos + 4)?;

        if index_block_elements == 0
            || data_block_min_elements == 0
            || !data_block_min_elements.is_power_of_two()
            || !super_block_min_data_ptrs.is_power_of_two()
        {
            return Err(Error::InvalidFormat(
                "invalid extensible array block parameters".into(),
            ));
        }
        let array_offset_size = max_elements_bits.div_ceil(8);
        let log_data_min = data_block_min_elements.trailing_zeros() as usize;
        let super_block_count = 1usize
            + (max_elements_bits as usize)
                .checked_sub(log_data_min)
                .ok_or_else(|| {
                    Error::InvalidFormat("invalid extensible array block parameters".into())
                })?;
        let index_block_super_blocks = 2 * (super_block_min_data_ptrs.trailing_zeros() as usize);
        let index_block_super_block_addrs = super_block_count
            .checked_sub(index_block_super_blocks)
            .ok_or_else(|| {
                Error::InvalidFormat("invalid extensible array super block layout".into())
            })?;
        let max_data_block_page_elements = 1usize
            .checked_shl(max_data_block_page_elements_bits as u32)
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array page element count overflow".into())
            })?;

        Ok(MutableExtensibleArrayHeader {
            class_id,
            raw_element_size,
            index_block_elements,
            data_block_min_elements,
            super_block_min_data_ptrs,
            max_data_block_page_elements,
            max_index_set,
            realized_elements,
            index_block_addr,
            array_offset_size,
            index_block_super_block_addrs,
            checksum_pos,
            data_block_count_pos,
            data_block_size_pos,
            max_index_set_pos,
            realized_elements_pos,
        })
    }

    fn locate_extensible_array_element<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        element_index: usize,
    ) -> Result<u64> {
        if crate::io::reader::is_undef_addr(header.index_block_addr) {
            return Err(Error::Unsupported(
                "cannot update extensible-array chunk entry without an index block".into(),
            ));
        }
        Self::verify_extensible_array_index_block(reader, header_addr, header)?;
        let direct_count = header.index_block_elements as usize;
        let index_prefix_size = 4 + 1 + 1 + reader.sizeof_addr() as usize;
        if element_index < direct_count {
            return Ok(header.index_block_addr
                + index_prefix_size as u64
                + (element_index * header.raw_element_size) as u64);
        }

        let data_block_index = element_index - direct_count;
        if data_block_index < header.data_block_min_elements {
            let data_block_addr_pos = header.index_block_addr
                + index_prefix_size as u64
                + (direct_count * header.raw_element_size) as u64;
            reader.seek(data_block_addr_pos)?;
            let data_block_addr = reader.read_addr()?;
            if crate::io::reader::is_undef_addr(data_block_addr) {
                return Err(Error::Unsupported(
                    "cannot update unallocated extensible-array data block".into(),
                ));
            }
            return Self::locate_extensible_array_data_block_element(
                reader,
                header_addr,
                header,
                data_block_addr,
                direct_count as u64,
                data_block_index,
            );
        }

        Err(Error::Unsupported(
            "write_chunk cannot update extensible-array super-block entries yet".into(),
        ))
    }

    fn locate_extensible_array_data_block_element<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        data_block_addr: u64,
        block_offset: u64,
        element_index: usize,
    ) -> Result<u64> {
        reader.seek(data_block_addr)?;
        if reader.read_bytes(4)? != b"EADB" {
            return Err(Error::InvalidFormat(
                "invalid extensible array data block magic".into(),
            ));
        }
        let version = reader.read_u8()?;
        let class_id = reader.read_u8()?;
        let owner = reader.read_addr()?;
        let stored_offset = reader.read_uint(header.array_offset_size)?;
        if version != 0
            || class_id != header.class_id
            || owner != header_addr
            || stored_offset != block_offset
        {
            return Err(Error::InvalidFormat(
                "extensible array data block header does not match index".into(),
            ));
        }
        if header.data_block_min_elements > header.max_data_block_page_elements {
            return Err(Error::Unsupported(
                "write_chunk cannot update paged extensible-array data blocks yet".into(),
            ));
        }
        Ok(data_block_addr
            + (4 + 1 + 1 + reader.sizeof_addr() as usize + header.array_offset_size as usize)
                as u64
            + (element_index * header.raw_element_size) as u64)
    }

    fn verify_extensible_array_index_block<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
    ) -> Result<()> {
        reader.seek(header.index_block_addr)?;
        if reader.read_bytes(4)? != b"EAIB" {
            return Err(Error::InvalidFormat(
                "invalid extensible array index block magic".into(),
            ));
        }
        let version = reader.read_u8()?;
        let class_id = reader.read_u8()?;
        let owner = reader.read_addr()?;
        if version != 0 || class_id != header.class_id || owner != header_addr {
            return Err(Error::InvalidFormat(
                "extensible array index block header does not match array header".into(),
            ));
        }
        Ok(())
    }

    fn append_first_extensible_array_data_block(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        if crate::io::reader::is_undef_addr(header.index_block_addr) {
            return Err(Error::Unsupported(
                "write_chunk cannot create a missing extensible-array index block yet".into(),
            ));
        }
        if header.data_block_min_elements > header.max_data_block_page_elements {
            return Err(Error::Unsupported(
                "write_chunk cannot create paged extensible-array data blocks yet".into(),
            ));
        }

        let data_block_elements = header.data_block_min_elements;
        let data_block_size = 4
            + 1
            + 1
            + self.superblock.sizeof_addr as usize
            + header.array_offset_size as usize
            + data_block_elements * header.raw_element_size
            + 4;
        let data_block_addr = self.append_aligned_zeros(data_block_size, 8)?;
        let block_offset = header.index_block_elements as u64;
        let mut block = Vec::with_capacity(data_block_size - 4);
        block.extend_from_slice(b"EADB");
        block.push(0);
        block.push(header.class_id);
        block.extend_from_slice(&header_addr.to_le_bytes()[..self.superblock.sizeof_addr as usize]);
        block.extend_from_slice(&block_offset.to_le_bytes()[..header.array_offset_size as usize]);
        block.extend_from_slice(&chunk_addr.to_le_bytes()[..self.superblock.sizeof_addr as usize]);
        if filtered {
            block.extend_from_slice(&chunk_size.to_le_bytes()[..chunk_size_len]);
            block.extend_from_slice(&0u32.to_le_bytes());
        }
        let fill_addr = crate::io::reader::UNDEF_ADDR.to_le_bytes();
        for _ in 1..data_block_elements {
            block.extend_from_slice(&fill_addr[..self.superblock.sizeof_addr as usize]);
            if filtered {
                block.extend_from_slice(&0u64.to_le_bytes()[..chunk_size_len]);
                block.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        let checksum = checksum_metadata(&block);
        self.write_handle.seek(SeekFrom::Start(data_block_addr))?;
        self.write_handle.write_all(&block)?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;

        let data_block_addr_pos = header.index_block_addr
            + (4 + 1 + 1 + self.superblock.sizeof_addr as usize) as u64
            + (header.index_block_elements as usize * header.raw_element_size) as u64;
        self.write_handle
            .seek(SeekFrom::Start(data_block_addr_pos))?;
        self.write_handle
            .write_all(&data_block_addr.to_le_bytes()[..self.superblock.sizeof_addr as usize])?;
        self.rewrite_extensible_array_index_block_checksum(header, Some((0, data_block_addr)))?;

        let new_max_index_set = header.max_index_set + 1;
        let new_realized = block_offset + data_block_elements as u64;
        self.rewrite_extensible_array_header_counts(
            header_addr,
            header,
            new_max_index_set,
            new_realized,
            Some((1, data_block_size as u64)),
        )?;
        Ok(())
    }

    fn write_extensible_array_element(
        &mut self,
        element_pos: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        self.write_handle.seek(SeekFrom::Start(element_pos))?;
        self.write_handle
            .write_all(&chunk_addr.to_le_bytes()[..self.superblock.sizeof_addr as usize])?;
        if filtered {
            self.write_uint_le(chunk_size, chunk_size_len)?;
            self.write_handle.write_all(&0u32.to_le_bytes())?;
        }
        Ok(())
    }

    fn rewrite_extensible_array_index_block_checksum(
        &mut self,
        header: &MutableExtensibleArrayHeader,
        first_data_block_addr: Option<(usize, u64)>,
    ) -> Result<()> {
        let sa = self.superblock.sizeof_addr as usize;
        let index_prefix_size = 4 + 1 + 1 + sa;
        let check_len = index_prefix_size
            + header.index_block_elements as usize * header.raw_element_size
            + 2 * (header.super_block_min_data_ptrs - 1) * sa
            + header.index_block_super_block_addrs * sa;
        let mut guard = self.inner.lock();
        guard.reader.seek(header.index_block_addr)?;
        let mut index_bytes = guard.reader.read_bytes(check_len)?;
        drop(guard);
        if let Some((data_block_index, data_block_addr)) = first_data_block_addr {
            let data_block_addr_offset = index_prefix_size
                + header.index_block_elements as usize * header.raw_element_size
                + data_block_index * sa;
            index_bytes[data_block_addr_offset..data_block_addr_offset + sa]
                .copy_from_slice(&data_block_addr.to_le_bytes()[..sa]);
        }
        let checksum = checksum_metadata(&index_bytes);
        self.write_handle
            .seek(SeekFrom::Start(header.index_block_addr + check_len as u64))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    fn rewrite_extensible_array_header_counts(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        new_max_index_set: u64,
        new_realized_elements: u64,
        data_block_counts: Option<(u64, u64)>,
    ) -> Result<()> {
        let ss = self.superblock.sizeof_size as usize;
        if let Some((data_block_count, data_block_size)) = data_block_counts {
            self.write_handle
                .seek(SeekFrom::Start(header.data_block_count_pos))?;
            self.write_uint_le(data_block_count, ss)?;
            self.write_handle
                .seek(SeekFrom::Start(header.data_block_size_pos))?;
            self.write_uint_le(data_block_size, ss)?;
        }
        self.write_handle
            .seek(SeekFrom::Start(header.max_index_set_pos))?;
        self.write_uint_le(new_max_index_set, ss)?;
        self.write_handle
            .seek(SeekFrom::Start(header.realized_elements_pos))?;
        self.write_uint_le(new_realized_elements, ss)?;

        let check_len = usize::try_from(header.checksum_pos - header_addr).map_err(|_| {
            Error::InvalidFormat("extensible array header checksum span is too large".into())
        })?;
        let mut guard = self.inner.lock();
        guard.reader.seek(header_addr)?;
        let mut header_bytes = guard.reader.read_bytes(check_len)?;
        drop(guard);
        if let Some((data_block_count, data_block_size)) = data_block_counts {
            let offset = usize::try_from(header.data_block_count_pos - header_addr).unwrap();
            header_bytes[offset..offset + ss]
                .copy_from_slice(&data_block_count.to_le_bytes()[..ss]);
            let offset = usize::try_from(header.data_block_size_pos - header_addr).unwrap();
            header_bytes[offset..offset + ss].copy_from_slice(&data_block_size.to_le_bytes()[..ss]);
        }
        let offset = usize::try_from(header.max_index_set_pos - header_addr).unwrap();
        header_bytes[offset..offset + ss].copy_from_slice(&new_max_index_set.to_le_bytes()[..ss]);
        let offset = usize::try_from(header.realized_elements_pos - header_addr).unwrap();
        header_bytes[offset..offset + ss]
            .copy_from_slice(&new_realized_elements.to_le_bytes()[..ss]);
        let checksum = checksum_metadata(&header_bytes);
        self.write_handle
            .seek(SeekFrom::Start(header.checksum_pos))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    fn rewrite_btree_v2_chunk(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        let scaled_coords = Self::scaled_chunk_coords(chunk_coords, chunk_dims)?;
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info.layout.version,
                unfiltered_chunk_bytes,
                self.superblock.sizeof_size,
            )
        } else {
            0
        };
        let sa = self.superblock.sizeof_addr as usize;
        let expected_record_size =
            sa + (chunk_size_len + 4) * usize::from(filtered) + 8 * scaled_coords.len();

        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        let header = BTreeV2Header::read_at(reader, index_addr)?;
        if header.depth != 0 {
            return Err(Error::Unsupported(
                "write_chunk cannot append to multi-level v2 B-tree chunk indexes yet".into(),
            ));
        }
        if header.record_size as usize != expected_record_size {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree chunk record size {} does not match expected {expected_record_size}",
                header.record_size
            )));
        }
        let max_records = Self::btree_v2_leaf_capacity(&header)?;
        let root_nrecords = header.root_nrecords as usize;
        if root_nrecords > max_records {
            return Err(Error::InvalidFormat(
                "v2 B-tree root leaf has more records than its node capacity".into(),
            ));
        }
        if header.tree_type != 10 && header.tree_type != 11 {
            return Err(Error::Unsupported(format!(
                "write_chunk cannot update v2 B-tree type {} chunk indexes",
                header.tree_type
            )));
        }

        reader.seek(header.root_addr)?;
        if reader.read_bytes(4)? != [b'B', b'T', b'L', b'F'] {
            return Err(Error::InvalidFormat("invalid v2 B-tree leaf magic".into()));
        }
        let leaf_version = reader.read_u8()?;
        let leaf_type = reader.read_u8()?;
        if leaf_version != 0 || leaf_type != header.tree_type {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf header does not match tree header".into(),
            ));
        }

        let mut records = Vec::with_capacity(root_nrecords + 1);
        let mut replacing = false;
        for _ in 0..root_nrecords {
            let record = reader.read_bytes(header.record_size as usize)?;
            let existing_scaled = Self::decode_btree_v2_scaled_coords(
                &record,
                filtered,
                chunk_size_len,
                sa,
                scaled_coords.len(),
            )?;
            if existing_scaled == scaled_coords {
                records.push(Self::encode_btree_v2_chunk_record(
                    chunk_addr,
                    chunk_size,
                    &scaled_coords,
                    filtered,
                    chunk_size_len,
                    sa,
                )?);
                replacing = true;
            } else {
                records.push(record);
            }
        }
        drop(guard);

        if !replacing {
            if root_nrecords >= max_records {
                return Err(Error::Unsupported(
                    "write_chunk cannot split full v2 B-tree leaf nodes yet".into(),
                ));
            }
            records.push(Self::encode_btree_v2_chunk_record(
                chunk_addr,
                chunk_size,
                &scaled_coords,
                filtered,
                chunk_size_len,
                sa,
            )?);
            let mut sortable_records = Vec::with_capacity(records.len());
            for record in records {
                let record_scaled = Self::decode_btree_v2_scaled_coords(
                    &record,
                    filtered,
                    chunk_size_len,
                    sa,
                    scaled_coords.len(),
                )?;
                sortable_records.push((record_scaled, record));
            }
            sortable_records.sort_by(|a, b| a.0.cmp(&b.0));
            records = sortable_records
                .into_iter()
                .map(|(_, record)| record)
                .collect();
        }

        let new_nrecords = records.len();
        let mut leaf = Vec::with_capacity(6 + new_nrecords * header.record_size as usize);
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(header.tree_type);
        for record in &records {
            leaf.extend_from_slice(record);
        }
        let leaf_checksum = checksum_metadata(&leaf);

        self.write_handle.seek(SeekFrom::Start(header.root_addr))?;
        self.write_handle.write_all(&leaf)?;
        self.write_handle.write_all(&leaf_checksum.to_le_bytes())?;

        if !replacing {
            self.rewrite_btree_v2_header_counts(index_addr, &header, new_nrecords as u16)?;
        }

        Ok(())
    }

    fn btree_v2_leaf_capacity(header: &BTreeV2Header) -> Result<usize> {
        let node_size = header.node_size as usize;
        let record_size = header.record_size as usize;
        if node_size <= 10 || record_size == 0 {
            return Err(Error::InvalidFormat("invalid v2 B-tree node sizing".into()));
        }
        let capacity = (node_size - 10) / record_size;
        if capacity == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf cannot hold any records".into(),
            ));
        }
        Ok(capacity)
    }

    fn rewrite_btree_v2_header_counts(
        &mut self,
        header_addr: u64,
        header: &BTreeV2Header,
        new_root_nrecords: u16,
    ) -> Result<()> {
        let sa = self.superblock.sizeof_addr as usize;
        let ss = self.superblock.sizeof_size as usize;
        let root_nrecords_pos = header_addr + (4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + sa) as u64;
        let total_records_pos = root_nrecords_pos + 2;
        let checksum_pos = total_records_pos + ss as u64;
        let old_root_nrecords = header.root_nrecords as u64;
        let new_total_records = header
            .total_records
            .checked_add(new_root_nrecords as u64)
            .and_then(|value| value.checked_sub(old_root_nrecords))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record count overflow".into()))?;

        self.write_handle.seek(SeekFrom::Start(root_nrecords_pos))?;
        self.write_handle
            .write_all(&new_root_nrecords.to_le_bytes())?;
        self.write_handle.seek(SeekFrom::Start(total_records_pos))?;
        self.write_uint_le(new_total_records, ss)?;

        let check_len = usize::try_from(checksum_pos - header_addr).map_err(|_| {
            Error::InvalidFormat("v2 B-tree header checksum span is too large".into())
        })?;
        let mut guard = self.inner.lock();
        guard.reader.seek(header_addr)?;
        let mut header_bytes = guard.reader.read_bytes(check_len)?;
        drop(guard);
        let root_offset = usize::try_from(root_nrecords_pos - header_addr).unwrap();
        let total_offset = usize::try_from(total_records_pos - header_addr).unwrap();
        header_bytes[root_offset..root_offset + 2]
            .copy_from_slice(&new_root_nrecords.to_le_bytes());
        header_bytes[total_offset..total_offset + ss]
            .copy_from_slice(&new_total_records.to_le_bytes()[..ss]);
        let checksum = checksum_metadata(&header_bytes);
        self.write_handle.seek(SeekFrom::Start(checksum_pos))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;

        Ok(())
    }

    fn encode_btree_v2_chunk_record(
        addr: u64,
        chunk_size: u64,
        scaled_coords: &[u64],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
    ) -> Result<Vec<u8>> {
        let mut record = Vec::new();
        record.extend_from_slice(&addr.to_le_bytes()[..sizeof_addr]);
        if filtered {
            if chunk_size_len == 0 || chunk_size_len > 8 {
                return Err(Error::InvalidFormat(format!(
                    "invalid v2 B-tree chunk size length {chunk_size_len}"
                )));
            }
            record.extend_from_slice(&chunk_size.to_le_bytes()[..chunk_size_len]);
            record.extend_from_slice(&0u32.to_le_bytes());
        }
        for &coord in scaled_coords {
            record.extend_from_slice(&coord.to_le_bytes());
        }
        Ok(record)
    }

    fn decode_btree_v2_scaled_coords(
        record: &[u8],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
        ndims: usize,
    ) -> Result<Vec<u64>> {
        let mut pos = sizeof_addr;
        if record.len() < pos {
            return Err(Error::InvalidFormat(
                "truncated v2 B-tree chunk address".into(),
            ));
        }
        if filtered {
            pos = pos
                .checked_add(chunk_size_len)
                .and_then(|value| value.checked_add(4))
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree record offset overflow".into()))?;
        }
        if record.len() < pos + ndims * 8 {
            return Err(Error::InvalidFormat(
                "truncated v2 B-tree scaled chunk coordinates".into(),
            ));
        }

        let mut coords = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            coords.push(Self::read_le_uint(&record[pos..pos + 8])?);
            pos += 8;
        }
        Ok(coords)
    }

    fn scaled_chunk_coords(chunk_coords: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>> {
        if chunk_coords.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match chunk dimensions".into(),
            ));
        }
        chunk_coords
            .iter()
            .zip(chunk_dims)
            .map(|(&coord, &dim)| {
                if dim == 0 {
                    return Err(Error::InvalidFormat("chunk dimension is zero".into()));
                }
                if coord % dim != 0 {
                    return Err(Error::InvalidFormat(
                        "chunk coordinate is not aligned to chunk dimension".into(),
                    ));
                }
                Ok(coord / dim)
            })
            .collect()
    }

    fn read_le_uint(bytes: &[u8]) -> Result<u64> {
        if bytes.len() > 8 {
            return Err(Error::InvalidFormat(
                "little-endian integer is wider than u64".into(),
            ));
        }
        let mut value = 0u64;
        for (idx, byte) in bytes.iter().enumerate() {
            value |= (*byte as u64) << (idx * 8);
        }
        Ok(value)
    }

    fn rewrite_fixed_array_chunk(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        let element_index =
            Self::linear_chunk_index(chunk_coords, &info.dataspace.dims, chunk_dims)?;
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info.layout.version,
                unfiltered_chunk_bytes,
                self.superblock.sizeof_size,
            )
        } else {
            0
        };

        let mut guard = self.inner.lock();
        let element_pos = crate::format::fixed_array::locate_fixed_array_element(
            &mut guard.reader,
            index_addr,
            filtered,
            chunk_size_len,
            element_index,
        )?;
        drop(guard);

        let sa = self.superblock.sizeof_addr as usize;
        self.write_handle.seek(SeekFrom::Start(element_pos))?;
        self.write_handle
            .write_all(&chunk_addr.to_le_bytes()[..sa])?;
        if filtered {
            self.write_uint_le(chunk_size, chunk_size_len)?;
            self.write_handle.write_all(&0u32.to_le_bytes())?;
        }
        Ok(())
    }

    fn linear_chunk_index(
        chunk_coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
    ) -> Result<usize> {
        if chunk_coords.len() != data_dims.len() || chunk_dims.len() != data_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match dataset rank".into(),
            ));
        }

        let chunks_per_dim: Vec<u64> = data_dims
            .iter()
            .zip(chunk_dims)
            .map(|(&dim, &chunk)| (dim + chunk - 1) / chunk)
            .collect();
        let mut index = 0usize;
        for dim in 0..data_dims.len() {
            let scaled = chunk_coords[dim] / chunk_dims[dim];
            if scaled >= chunks_per_dim[dim] {
                return Err(Error::Unsupported(
                    "fixed-array chunk index updates can replace existing chunks only".into(),
                ));
            }
            index = index
                .checked_mul(chunks_per_dim[dim] as usize)
                .and_then(|value| value.checked_add(scaled as usize))
                .ok_or_else(|| Error::InvalidFormat("chunk index overflow".into()))?;
        }
        Ok(index)
    }

    fn filtered_chunk_size_len(
        layout_version: u8,
        unfiltered_chunk_bytes: usize,
        sizeof_size: u8,
    ) -> usize {
        if layout_version > 4 {
            return sizeof_size as usize;
        }
        let bits = if unfiltered_chunk_bytes == 0 {
            0
        } else {
            usize::BITS as usize - unfiltered_chunk_bytes.leading_zeros() as usize
        };
        (1 + ((bits + 8) / 8)).min(8)
    }

    fn write_uint_le(&mut self, value: u64, size: usize) -> Result<()> {
        self.write_handle.write_all(&value.to_le_bytes()[..size])?;
        Ok(())
    }

    fn rewrite_leaf_chunk_btree(
        &mut self,
        btree_addr: u64,
        chunk_coords: &[u64],
        chunk_size: u32,
        chunk_addr: u64,
        element_size: u32,
    ) -> Result<()> {
        let ndims = chunk_coords.len();
        let sa = self.superblock.sizeof_addr as usize;
        let key_size = 4 + 4 + (ndims + 1) * 8;
        let entry_size = key_size + sa;
        let header_size = 4 + 1 + 1 + 2 + sa * 2;
        let max_entries = 64usize;

        let mut guard = self.inner.lock();
        guard.reader.seek(btree_addr)?;
        if guard.reader.read_bytes(4)? != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }
        let node_type = guard.reader.read_u8()?;
        let level = guard.reader.read_u8()?;
        let entries_used = guard.reader.read_u16()? as usize;
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw-data chunk B-tree, got type {node_type}"
            )));
        }
        if level != 0 {
            drop(guard);
            let mut entries = self.collect_chunk_btree_entries(btree_addr, ndims)?;
            if let Some(entry) = entries
                .iter_mut()
                .find(|entry| entry.coords.as_slice() == chunk_coords)
            {
                entry.chunk_size = chunk_size;
                entry.filter_mask = 0;
                entry.child_addr = chunk_addr;
            } else {
                entries.push(ChunkBTreeEntry {
                    coords: chunk_coords.to_vec(),
                    chunk_size,
                    filter_mask: 0,
                    child_addr: chunk_addr,
                });
            }
            entries.sort_by(|a, b| a.coords.cmp(&b.coords));
            self.rebuild_chunk_btree_from_entries(btree_addr, &entries, element_size, sa)?;
            return Ok(());
        }
        let entries_start = btree_addr + header_size as u64;
        let mut entries = Vec::with_capacity(entries_used + 1);
        for entry_idx in 0..entries_used {
            let key_pos = entries_start + (entry_idx * entry_size) as u64;
            guard.reader.seek(key_pos)?;
            let existing_size = guard.reader.read_u32()?;
            let filter_mask = guard.reader.read_u32()?;
            let mut coords = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                coords.push(guard.reader.read_u64()?);
            }
            let _extra = guard.reader.read_u64()?;
            let child_addr = guard.reader.read_addr()?;
            if coords == chunk_coords {
                drop(guard);
                self.write_btree_entry(key_pos, chunk_coords, chunk_size, chunk_addr, sa)?;
                return Ok(());
            }
            entries.push(ChunkBTreeEntry {
                coords,
                chunk_size: existing_size,
                filter_mask,
                child_addr,
            });
        }
        drop(guard);

        if entries_used >= max_entries {
            entries.push(ChunkBTreeEntry {
                coords: chunk_coords.to_vec(),
                chunk_size,
                filter_mask: 0,
                child_addr: chunk_addr,
            });
            entries.sort_by(|a, b| a.coords.cmp(&b.coords));
            self.rebuild_chunk_btree_from_entries(btree_addr, &entries, element_size, sa)?;
            return Ok(());
        }

        self.write_handle.seek(SeekFrom::Start(btree_addr + 6))?;
        self.write_handle
            .write_all(&((entries_used + 1) as u16).to_le_bytes())?;

        let new_entry_pos = entries_start + (entries_used * entry_size) as u64;
        self.write_btree_entry(new_entry_pos, chunk_coords, chunk_size, chunk_addr, sa)?;

        let final_key_pos = entries_start + ((entries_used + 1) * entry_size) as u64;
        self.write_btree_final_key(final_key_pos, chunk_coords, element_size)?;

        Ok(())
    }

    fn collect_chunk_btree_entries(
        &mut self,
        node_addr: u64,
        ndims: usize,
    ) -> Result<Vec<ChunkBTreeEntry>> {
        let mut guard = self.inner.lock();
        Self::collect_chunk_btree_entries_with_reader(&mut guard.reader, node_addr, ndims)
    }

    fn collect_chunk_btree_entries_with_reader<R: std::io::Read + Seek>(
        reader: &mut HdfReader<R>,
        node_addr: u64,
        ndims: usize,
    ) -> Result<Vec<ChunkBTreeEntry>> {
        reader.seek(node_addr)?;
        if reader.read_bytes(4)? != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }
        let node_type = reader.read_u8()?;
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw-data chunk B-tree, got type {node_type}"
            )));
        }
        let level = reader.read_u8()?;
        let entries_used = reader.read_u16()? as usize;
        let _left_sibling = reader.read_addr()?;
        let _right_sibling = reader.read_addr()?;

        if level == 0 {
            let mut entries = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                let chunk_size = reader.read_u32()?;
                let filter_mask = reader.read_u32()?;
                let mut coords = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    coords.push(reader.read_u64()?);
                }
                let _extra = reader.read_u64()?;
                let child_addr = reader.read_addr()?;
                entries.push(ChunkBTreeEntry {
                    coords,
                    chunk_size,
                    filter_mask,
                    child_addr,
                });
            }
            Ok(entries)
        } else {
            let mut child_addrs = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                let _chunk_size = reader.read_u32()?;
                let _filter_mask = reader.read_u32()?;
                for _ in 0..=ndims {
                    let _ = reader.read_u64()?;
                }
                child_addrs.push(reader.read_addr()?);
            }

            let mut entries = Vec::new();
            for child_addr in child_addrs {
                entries.extend(Self::collect_chunk_btree_entries_with_reader(
                    reader, child_addr, ndims,
                )?);
            }
            Ok(entries)
        }
    }

    fn rebuild_chunk_btree_from_entries(
        &mut self,
        root_addr: u64,
        entries: &[ChunkBTreeEntry],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        let ndims = entries
            .first()
            .map(|entry| entry.coords.len())
            .ok_or_else(|| Error::InvalidFormat("cannot rebuild empty chunk B-tree".into()))?;
        let node_size = Self::chunk_btree_node_size(ndims, sizeof_addr);

        if entries.len() <= 64 {
            self.write_chunk_btree_node(root_addr, 0, entries, element_size, sizeof_addr)?;
            return Ok(());
        }

        let leaf_count = entries.len().div_ceil(64);
        if leaf_count > 64 {
            return Err(Error::Unsupported(
                "write_chunk cannot grow v1 chunk B-tree beyond a two-level root".into(),
            ));
        }

        let mut root_entries = Vec::with_capacity(leaf_count);
        for leaf_entries in entries.chunks(64) {
            let leaf_addr = self.append_aligned_zeros(node_size, 8)?;
            self.write_chunk_btree_node(leaf_addr, 0, leaf_entries, element_size, sizeof_addr)?;
            root_entries.push(ChunkBTreeEntry {
                coords: leaf_entries[0].coords.clone(),
                chunk_size: leaf_entries[0].chunk_size,
                filter_mask: leaf_entries[0].filter_mask,
                child_addr: leaf_addr,
            });
        }
        self.write_chunk_btree_node(root_addr, 1, &root_entries, element_size, sizeof_addr)?;

        Ok(())
    }

    fn chunk_btree_node_size(ndims: usize, sizeof_addr: usize) -> usize {
        let key_size = 4 + 4 + (ndims + 1) * 8;
        let max_entries = 64usize;
        let header_size = 4 + 1 + 1 + 2 + sizeof_addr * 2;
        header_size + (max_entries + 1) * key_size + max_entries * sizeof_addr
    }

    fn append_aligned_zeros(&mut self, size: usize, align: u64) -> Result<u64> {
        let mut pos = self.write_handle.seek(SeekFrom::End(0))?;
        let padding = (align - (pos % align)) % align;
        if padding != 0 {
            self.write_handle.write_all(&vec![0u8; padding as usize])?;
            pos += padding;
        }
        self.write_handle.write_all(&vec![0u8; size])?;
        Ok(pos)
    }

    fn write_chunk_btree_node(
        &mut self,
        pos: u64,
        level: u8,
        entries: &[ChunkBTreeEntry],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        if entries.len() > 64 {
            return Err(Error::InvalidFormat(
                "chunk B-tree node entry count exceeds v1 node capacity".into(),
            ));
        }
        let ndims = entries
            .first()
            .map(|entry| entry.coords.len())
            .ok_or_else(|| Error::InvalidFormat("cannot write empty chunk B-tree node".into()))?;
        let node_size = Self::chunk_btree_node_size(ndims, sizeof_addr);
        let mut buf = Vec::with_capacity(node_size);
        buf.extend_from_slice(b"TREE");
        buf.push(1);
        buf.push(level);
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        let undef = crate::io::reader::UNDEF_ADDR.to_le_bytes();
        buf.extend_from_slice(&undef[..sizeof_addr]);
        buf.extend_from_slice(&undef[..sizeof_addr]);

        for entry in entries {
            buf.extend_from_slice(&entry.chunk_size.to_le_bytes());
            buf.extend_from_slice(&entry.filter_mask.to_le_bytes());
            for &coord in &entry.coords {
                buf.extend_from_slice(&coord.to_le_bytes());
            }
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(&entry.child_addr.to_le_bytes()[..sizeof_addr]);
        }

        let final_coords = &entries[entries.len() - 1].coords;
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for &coord in final_coords {
            buf.extend_from_slice(&coord.to_le_bytes());
        }
        buf.extend_from_slice(&(element_size as u64).to_le_bytes());
        buf.resize(node_size, 0);

        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&buf)?;
        Ok(())
    }

    fn write_btree_entry(
        &mut self,
        pos: u64,
        chunk_coords: &[u64],
        chunk_size: u32,
        chunk_addr: u64,
        sizeof_addr: usize,
    ) -> Result<()> {
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&chunk_size.to_le_bytes())?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        for &coord in chunk_coords {
            self.write_handle.write_all(&coord.to_le_bytes())?;
        }
        self.write_handle.write_all(&0u64.to_le_bytes())?;
        self.write_handle
            .write_all(&chunk_addr.to_le_bytes()[..sizeof_addr])?;

        Ok(())
    }

    fn write_btree_final_key(
        &mut self,
        pos: u64,
        chunk_coords: &[u64],
        element_size: u32,
    ) -> Result<()> {
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        for &coord in chunk_coords {
            self.write_handle.write_all(&coord.to_le_bytes())?;
        }
        self.write_handle
            .write_all(&(element_size as u64).to_le_bytes())?;
        Ok(())
    }
}
