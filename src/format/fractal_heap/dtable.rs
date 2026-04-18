//! Fractal heap doubling-table geometry — mirrors libhdf5's
//! `H5HFdtable.c`. Pure-arithmetic helpers that compute block sizes,
//! row counts, and span totals from the heap header parameters.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{log2_floor, log2_power2, FractalHeapHeader};

impl FractalHeapHeader {
    pub(super) fn max_direct_rows(&self) -> usize {
        let start_bits = log2_power2(self.start_block_size);
        let max_direct_bits = log2_power2(self.max_direct_block_size);
        (max_direct_bits - start_bits + 2) as usize
    }

    pub(super) fn child_indirect_rows(&self, row: usize) -> usize {
        let first_row_bits =
            log2_power2(self.start_block_size) + log2_power2(self.table_width as u64);
        (log2_floor(self.row_block_size(row)) - first_row_bits + 1) as usize
    }

    pub(super) fn indirect_data_span<R: Read + Seek>(
        &self,
        reader: &HdfReader<R>,
        nrows: usize,
    ) -> Result<u64> {
        let width = self.table_width as u64;
        let max_direct_rows = self.max_direct_rows();
        let mut span = 0u64;

        for row in 0..nrows {
            if row < max_direct_rows {
                span = span
                    .checked_add(self.row_block_size(row) * width)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap span overflow".into()))?;
            } else {
                let child_rows = self.child_indirect_rows(row);
                span = span
                    .checked_add(self.indirect_data_span(reader, child_rows)? * width)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap span overflow".into()))?;
            }
        }

        Ok(span)
    }

    pub(super) fn row_block_size(&self, row: usize) -> u64 {
        if row == 0 {
            self.start_block_size
        } else {
            self.start_block_size * (1u64 << (row - 1))
        }
    }
}
