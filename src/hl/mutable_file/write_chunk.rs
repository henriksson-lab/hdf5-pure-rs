use std::borrow::Cow;
use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::btree_v2::BTreeV2Header;
use crate::format::messages::data_layout::{ChunkIndexType, LayoutClass};
use crate::format::messages::filter_pipeline::{FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE};

use super::MutableFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WritableChunkIndexKind {
    BTreeV1,
    FixedArray,
    ExtensibleArray,
    BTreeV2,
}

impl MutableFile {
    /// Write a full uncompressed chunk and update the dataset's chunk index.
    ///
    /// This supports chunked datasets written by this crate where the chunk
    /// index variant is one of the currently mutable writer-side subsets.
    pub fn write_chunk(
        &mut self,
        dataset_path: &str,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> Result<()> {
        let ds = self.dataset(dataset_path)?;
        let info = ds.info()?;
        let index_kind = Self::writable_chunk_index_kind(&info);
        if self.superblock.base_addr != 0 && index_kind != WritableChunkIndexKind::BTreeV1 {
            return Err(Error::Unsupported(
                "write_chunk supports userblock files only for v1 B-tree chunk indexes".into(),
            ));
        }

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

        let chunk_data_dims = Self::chunk_data_dims(&info)?;
        Self::validate_chunk_coords(chunk_coords, &info.dataspace.dims, chunk_data_dims)?;

        let element_size = Self::u64_to_usize(u64::from(info.datatype.size), "datatype size")?;
        let expected_len = Self::expected_chunk_len(chunk_data_dims, element_size)?;
        Self::validate_chunk_write_len(data.len(), expected_len)?;
        let filtered = Self::encode_chunk_write_data(&info, data, element_size)?;

        let index_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing B-tree address".into()))?;

        self.validate_chunk_index_rewrite(
            index_kind,
            index_addr,
            &info,
            chunk_coords,
            chunk_data_dims,
            expected_len,
        )?;

        let chunk_physical_addr = self.write_handle.seek(SeekFrom::End(0))?;
        let chunk_addr =
            self.logical_addr_from_physical(chunk_physical_addr, "appended chunk address")?;
        self.write_handle.write_all(&filtered)?;
        self.rewrite_chunk_index(
            index_kind,
            index_addr,
            ds.addr(),
            &info,
            chunk_coords,
            chunk_data_dims,
            Self::usize_to_u64(filtered.len(), "filtered chunk size")?,
            chunk_addr,
            expected_len,
            element_size,
        )?;
        let physical_eof = self.write_handle.seek(SeekFrom::End(0))?;
        let logical_eof = physical_eof
            .checked_sub(self.superblock.base_addr)
            .ok_or_else(|| Error::InvalidFormat("file EOF is before HDF5 base address".into()))?;
        self.rewrite_superblock_eof(logical_eof)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;

        Ok(())
    }

    pub(super) fn chunk_data_dims(info: &crate::hl::dataset::DatasetInfo) -> Result<&[u64]> {
        let chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked layout missing chunk dims".into()))?;
        if chunk_dims.len() == info.dataspace.dims.len() + 1 {
            Ok(&chunk_dims[..info.dataspace.dims.len()])
        } else if chunk_dims.len() == info.dataspace.dims.len() {
            Ok(chunk_dims)
        } else {
            Err(Error::InvalidFormat(format!(
                "chunk dimension rank {} does not match dataset rank {}",
                chunk_dims.len(),
                info.dataspace.dims.len()
            )))
        }
    }

    fn validate_chunk_coords(
        chunk_coords: &[u64],
        data_dims: &[u64],
        chunk_data_dims: &[u64],
    ) -> Result<()> {
        if chunk_coords.len() != chunk_data_dims.len() {
            return Err(Error::InvalidFormat(format!(
                "chunk coordinate rank {} does not match dataset rank {}",
                chunk_coords.len(),
                chunk_data_dims.len()
            )));
        }
        if data_dims.len() != chunk_data_dims.len() {
            return Err(Error::InvalidFormat(format!(
                "chunk dimension rank {} does not match dataset rank {}",
                chunk_data_dims.len(),
                data_dims.len()
            )));
        }
        for (((idx, &coord), &dim), &chunk) in chunk_coords
            .iter()
            .enumerate()
            .zip(data_dims)
            .zip(chunk_data_dims)
        {
            if chunk == 0 || coord % chunk != 0 {
                return Err(Error::InvalidFormat(format!(
                    "chunk coordinate {idx}={coord} is not aligned to chunk size {chunk}"
                )));
            }
            if coord >= dim {
                return Err(Error::Unsupported(format!(
                    "write_chunk coordinate {idx}={coord} is outside dataset extent {dim}"
                )));
            }
        }
        Ok(())
    }

    fn expected_chunk_len(chunk_data_dims: &[u64], element_size: usize) -> Result<usize> {
        let chunk_elements = chunk_data_dims.iter().try_fold(1usize, |acc, &dim| {
            let dim = Self::u64_to_usize(dim, "chunk dimension")?;
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))
        })?;
        chunk_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))
    }

    fn validate_chunk_write_len(actual_len: usize, expected_len: usize) -> Result<()> {
        if actual_len != expected_len {
            return Err(Error::InvalidFormat(format!(
                "chunk data has {actual_len} bytes, expected {expected_len}",
            )));
        }
        Ok(())
    }

    fn encode_chunk_write_data<'a>(
        info: &crate::hl::dataset::DatasetInfo,
        data: &'a [u8],
        element_size: usize,
    ) -> Result<Cow<'a, [u8]>> {
        let mut filtered = Cow::Borrowed(data);
        let mut filter_buf = Vec::new();
        if let Some(ref pipeline) = info.filter_pipeline {
            for filter in &pipeline.filters {
                match filter.id {
                    FILTER_SHUFFLE => {
                        filter_buf.clear();
                        filter_buf.resize(filtered.len(), 0);
                        crate::filters::shuffle::shuffle_into(
                            filtered.as_ref(),
                            element_size,
                            &mut filter_buf,
                        )?;
                        Self::replace_encoded_filter_buffer(&mut filtered, &mut filter_buf);
                    }
                    FILTER_DEFLATE => {
                        let level = filter.client_data.first().copied().unwrap_or(6);
                        filter_buf.clear();
                        crate::filters::deflate::compress_into(
                            filtered.as_ref(),
                            level,
                            &mut filter_buf,
                        )?;
                        Self::replace_encoded_filter_buffer(&mut filtered, &mut filter_buf);
                    }
                    FILTER_FLETCHER32 => {
                        crate::filters::fletcher32::append_checksum_in_place(filtered.to_mut())?;
                    }
                    other => {
                        return Err(Error::Unsupported(format!(
                            "write_chunk cannot encode filter {other}"
                        )));
                    }
                }
            }
        }
        Ok(filtered)
    }

    fn replace_encoded_filter_buffer<'a>(filtered: &mut Cow<'a, [u8]>, filter_buf: &mut Vec<u8>) {
        let previous = std::mem::replace(filtered, Cow::Owned(std::mem::take(filter_buf)));
        if let Cow::Owned(mut previous) = previous {
            previous.clear();
            *filter_buf = previous;
        }
    }

    fn validate_chunk_index_rewrite(
        &mut self,
        index_kind: WritableChunkIndexKind,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_data_dims: &[u64],
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        match index_kind {
            WritableChunkIndexKind::FixedArray | WritableChunkIndexKind::ExtensibleArray => {
                Self::linear_chunk_index(chunk_coords, &info.dataspace.dims, chunk_data_dims)?;
                Ok(())
            }
            WritableChunkIndexKind::BTreeV1 => self.validate_btree_v1_chunk_index(index_addr),
            WritableChunkIndexKind::BTreeV2 => self.validate_btree_v2_chunk_index(
                index_addr,
                info,
                chunk_coords.len(),
                unfiltered_chunk_bytes,
            ),
        }
    }

    fn validate_btree_v1_chunk_index(&mut self, btree_addr: u64) -> Result<()> {
        let mut guard = self.inner.lock();
        guard.reader.seek(btree_addr)?;
        let mut magic = [0u8; 4];
        guard.reader.read_bytes_into(&mut magic)?;
        if magic != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }
        let node_type = guard.reader.read_u8()?;
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw-data chunk B-tree, got type {node_type}"
            )));
        }
        Ok(())
    }

    fn validate_btree_v2_chunk_index(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        ndims: usize,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
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
        let expected_record_size = Self::btree_v2_preflight_record_size(
            usize::from(self.superblock.sizeof_addr),
            filtered,
            chunk_size_len,
            ndims,
        )?;

        let mut guard = self.inner.lock();
        let header = BTreeV2Header::read_at(&mut guard.reader, index_addr)?;
        if usize::from(header.record_size) != expected_record_size {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree chunk record size {} does not match expected {expected_record_size}",
                header.record_size
            )));
        }
        if header.tree_type != 10 && header.tree_type != 11 {
            return Err(Error::Unsupported(format!(
                "write_chunk cannot update v2 B-tree type {} chunk indexes",
                header.tree_type
            )));
        }
        Ok(())
    }

    fn btree_v2_preflight_record_size(
        sizeof_addr: usize,
        filtered: bool,
        chunk_size_len: usize,
        ndims: usize,
    ) -> Result<usize> {
        let filter_bytes = if filtered {
            chunk_size_len
                .checked_add(4)
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))?
        } else {
            0
        };
        let coord_bytes = ndims
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))?;
        sizeof_addr
            .checked_add(filter_bytes)
            .and_then(|value| value.checked_add(coord_bytes))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn rewrite_chunk_index(
        &mut self,
        index_kind: WritableChunkIndexKind,
        index_addr: u64,
        dataset_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_data_dims: &[u64],
        filtered_len: u64,
        chunk_addr: u64,
        expected_len: usize,
        element_size: usize,
    ) -> Result<()> {
        match index_kind {
            WritableChunkIndexKind::FixedArray => self.rewrite_fixed_array_chunk(
                index_addr,
                dataset_addr,
                info,
                chunk_coords,
                chunk_data_dims,
                filtered_len,
                chunk_addr,
                expected_len,
            ),
            WritableChunkIndexKind::ExtensibleArray => self.rewrite_extensible_array_chunk(
                index_addr,
                info,
                chunk_coords,
                chunk_data_dims,
                filtered_len,
                chunk_addr,
                expected_len,
            ),
            WritableChunkIndexKind::BTreeV2 => self.rewrite_btree_v2_chunk(
                index_addr,
                info,
                chunk_coords,
                chunk_data_dims,
                filtered_len,
                chunk_addr,
                expected_len,
            ),
            WritableChunkIndexKind::BTreeV1 => self.rewrite_leaf_chunk_btree(
                index_addr,
                chunk_coords,
                chunk_data_dims,
                Self::u64_to_u32(filtered_len, "filtered chunk size")?,
                chunk_addr,
                Self::usize_to_u32(element_size, "datatype size")?,
            ),
        }
    }

    fn writable_chunk_index_kind(info: &crate::hl::dataset::DatasetInfo) -> WritableChunkIndexKind {
        match info.layout.chunk_index_type {
            Some(ChunkIndexType::FixedArray) => WritableChunkIndexKind::FixedArray,
            Some(ChunkIndexType::ExtensibleArray) => WritableChunkIndexKind::ExtensibleArray,
            Some(ChunkIndexType::BTreeV2) => WritableChunkIndexKind::BTreeV2,
            _ => WritableChunkIndexKind::BTreeV1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_chunk_len_rejects_element_count_overflow() {
        let err = MutableFile::expected_chunk_len(&[u64::MAX, 2], 1).unwrap_err();
        assert!(
            err.to_string().contains("chunk element count"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_chunk_coords_rejects_out_of_extent_chunk() {
        let err = MutableFile::validate_chunk_coords(&[10], &[10], &[5]).unwrap_err();
        assert!(
            err.to_string().contains("outside dataset extent"),
            "unexpected error: {err}"
        );
    }
}
