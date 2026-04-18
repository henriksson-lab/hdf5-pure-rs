//! Fractal heap managed-object access — mirrors libhdf5's `H5HFman.c`.
//! Decode the heap-ID for managed objects, descend through the doubling
//! table (direct or indirect / filtered or unfiltered), and return the
//! object bytes. Composes with the iblock/dblock decoders.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::iblock::{FilteredIndirectBlock, IndirectBlock};
use super::FractalHeapHeader;

impl FractalHeapHeader {
    /// Read a managed (type 0) object.
    pub(super) fn read_managed<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        // Managed object heap ID:
        // byte 0: version(2 bits) + type(2 bits) + reserved(4 bits)
        // then: offset (ceil(max_heap_size/8) bytes) + length (remaining bytes)

        let offset_bytes = ((self.max_heap_size as usize) + 7) / 8;

        if heap_id.len() < 1 + offset_bytes {
            return Err(Error::InvalidFormat("heap ID too short for offset".into()));
        }

        let mut offset = 0u64;
        for i in 0..offset_bytes {
            offset |= (heap_id[1 + i] as u64) << (i * 8);
        }

        let len_start = 1 + offset_bytes;
        let mut length = 0u64;
        for i in 0..(heap_id.len() - len_start).min(8) {
            length |= (heap_id[len_start + i] as u64) << (i * 8);
        }

        // Bound checks matching libhdf5's `H5HF__man_op_real`:
        //  - offset must be < 2^max_heap_size (the heap's address space)
        //  - object size must fit in the managed object size limit
        if self.max_heap_size < 64 && offset >= (1u64 << self.max_heap_size) {
            return Err(Error::InvalidFormat(format!(
                "fractal heap object offset {offset} exceeds 2^{} address space",
                self.max_heap_size
            )));
        }
        if length > self.max_managed_obj_size as u64 {
            return Err(Error::InvalidFormat(format!(
                "fractal heap object size {length} exceeds max managed object size {}",
                self.max_managed_obj_size
            )));
        }

        if self.current_root_rows == 0 {
            // Root is a direct block -- offset is relative to block start
            self.read_from_direct_block(
                reader,
                self.root_block_addr,
                self.start_block_size,
                self.root_direct_filtered_size,
                self.root_direct_filter_mask,
                offset,
                length,
            )
        } else {
            // Root is an indirect block -- need to find which direct block contains the offset
            self.read_from_indirect_block(reader, self.root_block_addr, offset, length)
        }
    }

    pub(super) fn read_from_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if self.io_filter_len > 0 {
            return self.read_from_filtered_indirect_block(reader, block_addr, offset, length);
        }

        self.read_from_indirect_block_rows(
            reader,
            block_addr,
            self.current_root_rows as usize,
            0,
            offset,
            length,
        )
    }

    /// Walk a decoded indirect block to locate the heap object covering
    /// `offset`. Mirrors libhdf5's `H5HF__man_op_real` traversal: walks
    /// the row table, descending into nested indirect blocks once we leave
    /// the direct-row range.
    pub(super) fn lookup_in_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        iblock: &IndirectBlock,
        block_start: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let width = self.table_width as usize;
        let max_direct_rows = self.max_direct_rows();
        let mut current_heap_offset = block_start;
        let mut entry_index = 0usize;

        for row in 0..iblock.nrows {
            if row < max_direct_rows {
                let block_span = self.row_block_size(row);
                for _ in 0..width {
                    let child_addr = iblock.child_addrs[entry_index];
                    entry_index += 1;

                    if crate::io::reader::is_undef_addr(child_addr) {
                        current_heap_offset += block_span;
                        continue;
                    }

                    if offset >= current_heap_offset && offset < current_heap_offset + block_span {
                        let local_offset = offset - current_heap_offset;
                        return self.read_from_direct_block(
                            reader,
                            child_addr,
                            block_span,
                            None,
                            0,
                            local_offset,
                            length,
                        );
                    }

                    current_heap_offset += block_span;
                }
            } else {
                let child_rows = self.child_indirect_rows(row);
                let child_span = self.indirect_data_span(reader, child_rows)?;
                for _ in 0..width {
                    let child_addr = iblock.child_addrs[entry_index];
                    entry_index += 1;
                    if offset >= current_heap_offset && offset < current_heap_offset + child_span {
                        if crate::io::reader::is_undef_addr(child_addr) {
                            break;
                        }
                        return self.read_from_indirect_block_rows(
                            reader,
                            child_addr,
                            child_rows,
                            current_heap_offset,
                            offset,
                            length,
                        );
                    }
                    current_heap_offset += child_span;
                }
            }
        }

        Err(Error::InvalidFormat(format!(
            "fractal heap offset {offset} not found in indirect block"
        )))
    }

    /// Drive `decode_indirect_block` + `lookup_in_indirect_block` — the
    /// C-side composition is `H5HF__man_iblock_protect` (which loads &
    /// deserializes the iblock) followed by the lookup loop in
    /// `H5HF__man_op_real`.
    pub(super) fn read_from_indirect_block_rows<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        nrows: usize,
        block_start: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let iblock = self.decode_indirect_block(reader, block_addr, nrows)?;
        self.lookup_in_indirect_block(reader, &iblock, block_start, offset, length)
    }

    /// Walk a decoded filtered indirect block to locate the heap object
    /// covering `offset`. Mirrors the filtered traversal in
    /// `H5HF__man_op_real`.
    pub(super) fn lookup_in_filtered_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        iblock: &FilteredIndirectBlock,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let width = self.table_width as usize;
        let dblock_header_size = 5
            + self.sizeof_addr as u64
            + iblock.block_offset_bytes as u64
            + if self.has_checksum { 4 } else { 0 };
        let mut current_heap_offset = 0u64;
        let mut entry_index = 0usize;

        for row in 0..iblock.nrows {
            let block_size = if row < 2 {
                self.start_block_size
            } else {
                self.start_block_size * (1u64 << (row - 1))
            };

            if block_size > self.max_direct_block_size {
                entry_index += width; // indirect-row entries carry no payload to consume here
                continue;
            }

            let data_capacity = block_size - dblock_header_size;
            for _ in 0..width {
                let entry = &iblock.entries[entry_index];
                entry_index += 1;
                if crate::io::reader::is_undef_addr(entry.addr) {
                    current_heap_offset += data_capacity;
                    continue;
                }
                if offset >= current_heap_offset && offset < current_heap_offset + data_capacity {
                    return self.read_from_direct_block(
                        reader,
                        entry.addr,
                        block_size,
                        Some(entry.filtered_size),
                        entry.filter_mask,
                        offset - current_heap_offset,
                        length,
                    );
                }
                current_heap_offset += data_capacity;
            }
        }

        Err(Error::InvalidFormat(format!(
            "filtered fractal heap offset {offset} not found in indirect block"
        )))
    }

    /// Drive `decode_filtered_indirect_block` + `lookup_in_filtered_…` —
    /// C-side composition is `H5HF__man_iblock_protect` + the filtered
    /// branch of `H5HF__man_op_real`.
    pub(super) fn read_from_filtered_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let iblock = self.decode_filtered_indirect_block(reader, block_addr)?;
        self.lookup_in_filtered_indirect_block(reader, &iblock, offset, length)
    }
}
