//! Fractal heap direct blocks — mirrors libhdf5's `H5HFdblock.c` plus
//! the dblock half of `H5HFcache.c` (`H5HF__cache_dblock_deserialize`).
//! In the Rust port, direct-block reads are pull-style (no separate
//! cache layer), so this file just holds `read_from_direct_block`.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{
    heap_object_len, verify_direct_block_checksum, FractalHeapHeader,
};

impl FractalHeapHeader {
    pub(super) fn read_from_direct_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if let Some(filtered_size) = filtered_size {
            reader.seek(block_addr)?;
            let filtered = reader.read_bytes(heap_object_len(
                filtered_size,
                "filtered fractal heap block size",
            )?)?;
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("filtered fractal heap missing filter pipeline".into())
            })?;
            let data = crate::filters::apply_pipeline_reverse(&filtered, pipeline, 1)?;
            let start = heap_object_len(offset, "fractal heap object offset")?;
            let end = start
                .checked_add(heap_object_len(length, "fractal heap object length")?)
                .ok_or_else(|| Error::InvalidFormat("fractal heap object range overflow".into()))?;
            let out = data
                .get(start..end)
                .map(|slice| slice.to_vec())
                .ok_or_else(|| {
                    Error::InvalidFormat("fractal heap object exceeds filtered direct block".into())
                })?;
            self.trace_managed_object(block_addr, block_size, offset, length, filter_mask, true);
            return Ok(out);
        }

        if self.has_checksum && self.heap_addr == 0 {
            verify_direct_block_checksum(reader, block_addr, self.max_heap_size, block_size)?;
        }

        let addr = block_addr
            .checked_add(offset)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object address overflow".into()))?;
        reader.seek(addr)?;
        let data = reader.read_bytes(heap_object_len(length, "fractal heap object length")?)?;
        self.trace_managed_object(block_addr, block_size, offset, length, 0, false);
        Ok(data)
    }
}
