//! Fractal heap — top-level public API. Mirrors libhdf5's `H5HF.c`
//! (the file-spanning entry points). Per-component code lives in sibling
//! modules:
//!   - `hdr`    → `H5HFhdr.c` + hdr-half of `H5HFcache.c`
//!   - `iblock` → `H5HFiblock.c` + iblock-half of `H5HFcache.c`
//!   - `dblock` → `H5HFdblock.c` + dblock-half of `H5HFcache.c`
//!   - `man`    → `H5HFman.c`
//!   - `huge`   → `H5HFhuge.c` + `H5HFbtree2.c`
//!   - `tiny`   → `H5HFtiny.c`
//!   - `dtable` → `H5HFdtable.c`
//!
//! Trace probes and small numeric helpers live here in `mod.rs` because
//! they cut across all the per-block files.

mod dblock;
mod dtable;
mod hdr;
mod huge;
mod iblock;
mod man;
mod tiny;

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::io::reader::HdfReader;

/// Fractal heap header magic: "FRHP"
pub(super) const FRHP_MAGIC: [u8; 4] = [b'F', b'R', b'H', b'P'];
/// Direct block magic: "FHDB" (kept for reference; we currently read by
/// offset rather than by magic).
#[allow(dead_code)]
pub(super) const FHDB_MAGIC: [u8; 4] = [b'F', b'H', b'D', b'B'];
/// Indirect block magic: "FHIB"
pub(super) const FHIB_MAGIC: [u8; 4] = [b'F', b'H', b'I', b'B'];
const MAX_HEAP_OBJECT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Fractal heap header.
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    pub heap_addr: u64,
    pub heap_id_len: u16,
    pub io_filter_len: u16,
    pub flags: u8,
    pub max_managed_obj_size: u32,

    pub table_width: u16,
    pub start_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size: u16,
    pub start_root_rows: u16,
    pub root_block_addr: u64,
    pub current_root_rows: u16,
    pub num_managed_objects: u64,
    pub has_checksum: bool,
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
    pub huge_btree_addr: u64,
    pub root_direct_filtered_size: Option<u64>,
    pub root_direct_filter_mask: u32,
    pub filter_pipeline: Option<FilterPipelineMessage>,
}

impl FractalHeapHeader {
    /// Read a managed object from the fractal heap by its heap ID.
    /// Mirrors libhdf5's `H5HF_op` / `H5HF_get_obj_len`: dispatches by
    /// the type bits (managed / huge / tiny) in the heap-ID byte 0.
    pub fn read_managed_object<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        if heap_id.is_empty() {
            return Err(Error::InvalidFormat("empty heap ID".into()));
        }

        // Heap ID byte 0 layout (per H5HFpkg.h): bits 6-7 = version,
        // bits 4-5 = type, bits 0-3 = reserved (or tiny-length). Only
        // version 0 is currently defined; reject anything else, matching
        // libhdf5's `H5HF_get_obj_len` "incorrect heap ID version" check.
        let version = (heap_id[0] >> 6) & 0x03;
        if version != 0 {
            return Err(Error::InvalidFormat(format!(
                "unsupported fractal heap ID version {version}"
            )));
        }
        let id_type = (heap_id[0] >> 4) & 0x03;

        match id_type {
            0 => self.read_managed(reader, heap_id),
            1 => self.read_huge(reader, heap_id),
            2 => self.read_tiny(heap_id),
            _ => Err(Error::InvalidFormat(format!(
                "unknown heap ID type {id_type}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Trace probes — kept in one place because they're small, conditional on
// the `tracehash` feature, and called from sibling modules (dblock/huge/
// tiny). Each emits one tracehash event for the read it just performed.
// ---------------------------------------------------------------------------

impl FractalHeapHeader {
    #[cfg(feature = "tracehash")]
    pub(super) fn trace_managed_object(
        &self,
        block_addr: u64,
        block_size: u64,
        block_offset: u64,
        object_len: u64,
        _filter_mask: u32,
        filtered: bool,
    ) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.managed_object");
        th.input_u64(self.heap_addr);
        th.input_u64(block_addr);
        th.input_u64(block_offset);
        th.input_u64(object_len);
        th.output_value(&(true));
        th.output_u64(block_addr);
        th.output_u64(block_size);
        th.output_u64(block_offset);
        th.output_u64(object_len);
        th.output_u64(0);
        th.output_value(&(filtered));
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_managed_object(
        &self,
        _block_addr: u64,
        _block_size: u64,
        _block_offset: u64,
        _object_len: u64,
        _filter_mask: u32,
        _filtered: bool,
    ) {
    }

    #[cfg(feature = "tracehash")]
    pub(super) fn trace_huge_object(
        &self,
        heap_id: &[u8],
        addr: u64,
        stored_len: u64,
        object_len: u64,
        filter_mask: u32,
        filtered: bool,
    ) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.huge_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(stored_len);
        th.output_u64(object_len);
        th.output_u64(filter_mask as u64);
        th.output_value(&(filtered));
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_huge_object(
        &self,
        _heap_id: &[u8],
        _addr: u64,
        _stored_len: u64,
        _object_len: u64,
        _filter_mask: u32,
        _filtered: bool,
    ) {
    }

    #[cfg(feature = "tracehash")]
    pub(super) fn trace_tiny_object(&self, heap_id: &[u8], object_len: u64) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.tiny_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_value(&(true));
        th.output_u64(object_len);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_tiny_object(&self, _heap_id: &[u8], _object_len: u64) {}
}

// ---------------------------------------------------------------------------
// Internal numeric / checksum helpers shared across submodules.
// ---------------------------------------------------------------------------

pub(super) fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().take(8).enumerate() {
        value |= (*byte as u64) << (idx * 8);
    }
    value
}

pub(super) fn verify_metadata_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    check_len: u64,
    context: &str,
) -> Result<()> {
    let restore = reader.position()?;
    let check_len_usize = heap_object_len(check_len, context)?;
    reader.seek(start)?;
    let bytes = reader.read_bytes(check_len_usize)?;
    let stored = reader.read_u32()?;
    let computed = checksum_metadata(&bytes);
    reader.seek(restore)?;
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

pub(super) fn verify_direct_block_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    max_heap_size: u16,
    block_size: u64,
) -> Result<()> {
    let restore = reader.position()?;
    let block_offset_bytes = ((max_heap_size as usize) + 7) / 8;
    let check_len = 4usize
        .checked_add(1)
        .and_then(|n| n.checked_add(reader.sizeof_addr() as usize))
        .and_then(|n| n.checked_add(block_offset_bytes))
        .ok_or_else(|| {
            Error::InvalidFormat("fractal heap direct block checksum span overflow".into())
        })?;
    reader.seek(start)?;
    let bytes = reader.read_bytes(check_len)?;
    let stored = reader.read_u32()?;
    let payload_len = heap_object_len(block_size, "fractal heap direct block size")?;
    let payload = reader.read_bytes(payload_len)?;
    reader.seek(restore)?;

    let computed = checksum_metadata(&bytes);
    let mut with_zero_checksum = bytes.clone();
    with_zero_checksum.extend_from_slice(&0u32.to_le_bytes());
    let computed_with_zero_checksum = checksum_metadata(&with_zero_checksum);
    let mut whole_block = with_zero_checksum;
    whole_block.extend_from_slice(&payload);
    let computed_whole_block = checksum_metadata(&whole_block);
    if stored != computed && stored != computed_with_zero_checksum && stored != computed_whole_block
    {
        return Err(Error::InvalidFormat(format!(
            "fractal heap direct block checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

pub(super) fn heap_object_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_HEAP_OBJECT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_HEAP_OBJECT_BYTES}"
        )));
    }
    Ok(len)
}

pub(super) fn log2_power2(value: u64) -> u32 {
    debug_assert!(value.is_power_of_two());
    value.trailing_zeros()
}

pub(super) fn log2_floor(value: u64) -> u32 {
    63 - value.leading_zeros()
}
