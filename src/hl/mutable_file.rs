use std::fs;
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
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
                Some(ChunkIndexType::BTreeV1) | Some(ChunkIndexType::FixedArray)
            )
        {
            return Err(Error::Unsupported(
                "write_chunk currently supports only v1 B-tree and fixed-array chunk indexes"
                    .into(),
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
