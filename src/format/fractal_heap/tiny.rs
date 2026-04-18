//! Fractal heap tiny-object access — mirrors libhdf5's `H5HFtiny.c`.
//! Tiny objects live entirely inside the heap-ID byte string; no I/O.

use crate::error::{Error, Result};

use super::FractalHeapHeader;

impl FractalHeapHeader {
    pub(super) fn read_tiny(&self, heap_id: &[u8]) -> Result<Vec<u8>> {
        let length = (heap_id[0] & 0x0F) as usize + 1;
        if heap_id.len() < 1 + length {
            return Err(Error::InvalidFormat("tiny heap ID too short".into()));
        }
        let data = heap_id[1..1 + length].to_vec();
        self.trace_tiny_object(heap_id, length as u64);
        Ok(data)
    }
}
