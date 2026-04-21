//! Fractal heap indirect blocks — mirrors libhdf5's `H5HFiblock.c` plus
//! the iblock half of `H5HFcache.c` (`H5HF__cache_iblock_deserialize`).
//! Pure decoders for both unfiltered and filtered indirect blocks; the
//! traversal logic that consumes these decoded blocks lives in `man.rs`.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{read_le_uint, verify_metadata_checksum, FractalHeapHeader, FHIB_MAGIC};

/// Decoded fractal-heap indirect block: row count + flat list of child
/// pointers in (row, column) order. Output of `decode_indirect_block`,
/// consumed by `lookup_in_indirect_block`.
pub(super) struct IndirectBlock {
    pub(super) nrows: usize,
    pub(super) child_addrs: Vec<u64>,
}

/// Decoded *filtered* fractal-heap indirect block. Each direct-row entry
/// carries the `(addr, filtered_size, filter_mask)` triple read off disk;
/// indirect-row entries only carry the address (the other two fields are
/// 0 for those rows). Output of `decode_filtered_indirect_block`.
pub(super) struct FilteredIndirectEntry {
    pub(super) addr: u64,
    pub(super) filtered_size: u64,
    pub(super) filter_mask: u32,
}

pub(super) struct FilteredIndirectBlock {
    pub(super) nrows: usize,
    pub(super) block_offset_bytes: usize,
    pub(super) entries: Vec<FilteredIndirectEntry>,
}

impl FractalHeapHeader {
    /// Pure deserializer for a fractal-heap indirect block: validates the
    /// FHIB magic, reads the prefix, verifies the metadata checksum, and
    /// returns the table of child entries. Mirrors libhdf5's
    /// `H5HF__cache_iblock_deserialize` — no traversal of the listed
    /// addresses.
    pub(super) fn decode_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        nrows: usize,
    ) -> Result<IndirectBlock> {
        reader.seek(block_addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != FHIB_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid fractal heap indirect block magic".into(),
            ));
        }

        let _version = reader.read_u8()?;
        let _heap_header_addr = reader.read_addr()?;

        let block_offset_bytes = ((self.max_heap_size as usize) + 7) / 8;
        let _block_offset = read_le_uint(&reader.read_bytes(block_offset_bytes)?);
        if self.has_checksum {
            let checksum_span = 4usize
                .checked_add(1)
                .and_then(|n| n.checked_add(self.sizeof_addr as usize))
                .and_then(|n| n.checked_add(block_offset_bytes))
                .and_then(|n| {
                    n.checked_add(
                        nrows
                            .checked_mul(self.table_width as usize)?
                            .checked_mul(self.sizeof_addr as usize)?,
                    )
                })
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "fractal heap indirect block checksum span overflow".into(),
                    )
                })?;
            verify_metadata_checksum(
                reader,
                block_addr,
                checksum_span as u64,
                "fractal heap indirect block",
            )?;
        }

        let width = self.table_width as usize;
        let total_entries = nrows
            .checked_mul(width)
            .ok_or_else(|| Error::InvalidFormat("fractal heap entry count overflow".into()))?;
        let mut child_addrs = Vec::with_capacity(total_entries);
        for _ in 0..total_entries {
            child_addrs.push(reader.read_addr()?);
        }
        Ok(IndirectBlock { nrows, child_addrs })
    }

    /// Pure deserializer for a *filtered* fractal-heap indirect block.
    /// Each direct-row entry carries an extra (filtered_size, filter_mask)
    /// pair after the address; rows past `max_direct_rows` only carry
    /// addresses (and we still need to consume them to keep the reader
    /// aligned). Mirrors the filtered branch of
    /// `H5HF__cache_iblock_deserialize`.
    pub(super) fn decode_filtered_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
    ) -> Result<FilteredIndirectBlock> {
        reader.seek(block_addr)?;
        let magic = reader.read_bytes(4)?;
        if magic != FHIB_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid fractal heap indirect block magic".into(),
            ));
        }
        let _version = reader.read_u8()?;
        let _heap_header_addr = reader.read_addr()?;
        let block_offset_bytes = ((self.max_heap_size as usize) + 7) / 8;
        reader.skip(block_offset_bytes as u64)?;

        let nrows = self.current_root_rows as usize;
        let width = self.table_width as usize;
        let mut entries = Vec::with_capacity(
            nrows
                .checked_mul(width)
                .ok_or_else(|| Error::InvalidFormat("fractal heap entry count overflow".into()))?,
        );
        for row in 0..nrows {
            let block_size = if row < 2 {
                self.start_block_size
            } else {
                self.start_block_size * (1u64 << (row - 1))
            };
            let direct = block_size <= self.max_direct_block_size;
            for _ in 0..width {
                let addr = reader.read_addr()?;
                let (filtered_size, filter_mask) = if direct {
                    (reader.read_length()?, reader.read_u32()?)
                } else {
                    (0, 0)
                };
                entries.push(FilteredIndirectEntry {
                    addr,
                    filtered_size,
                    filter_mask,
                });
            }
        }
        Ok(FilteredIndirectBlock {
            nrows,
            block_offset_bytes,
            entries,
        })
    }
}
