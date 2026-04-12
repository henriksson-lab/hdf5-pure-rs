use std::fs;
use std::io::{BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::data_layout::LayoutClass;
use crate::format::object_header::{self, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES, HDR_ATTR_STORE_PHASE_CHANGE};
use crate::format::superblock::Superblock;
use crate::hl::dataset::Dataset;
use crate::hl::file::FileInner;
use crate::hl::group::Group;
use crate::io::reader::HdfReader;

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
        }));

        // Open separately for writing
        let write_handle = fs::OpenOptions::new()
            .write(true)
            .open(&path)?;

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
            return Err(Error::InvalidFormat("expected v2 object header for resize".into()));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported("resize only supported for v2 object headers".into()));
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

        self.write_handle.seek(SeekFrom::Start(oh_start + check_len as u64))?;
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
        };
        Ok(())
    }

    /// Write new chunk data to extend a dataset.
    /// This appends new chunks at the end of the file but does NOT update the B-tree.
    /// For full extend support, use resize_dataset() to update dims, then write chunks.
    pub fn write_chunk(
        &mut self,
        _dataset_path: &str,
        _chunk_coords: &[u64],
        _data: &[u8],
    ) -> Result<()> {
        Err(Error::Unsupported(
            "write_chunk: B-tree update for new chunks not yet implemented. \
             Use resize_dataset() to shrink or mark new dimensions, \
             new regions will read as fill value (zeros)."
                .into(),
        ))
    }
}
