use std::io::{Seek, SeekFrom, Write};

use crate::error::Result;

use super::MutableFile;

impl MutableFile {
    pub(super) fn rewrite_fixed_array_chunk(
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
}
