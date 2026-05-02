use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::superblock::Superblock;
use crate::hl::file::FileInner;
use crate::io::reader::HdfReader;

use super::MutableFile;

impl MutableFile {
    /// Recompute and rewrite the v2 OH checksum.
    pub(super) fn rewrite_oh_checksum(&mut self, oh_start: u64, check_len: usize) -> Result<()> {
        let mut guard = self.inner.lock();
        guard.reader.seek(oh_start)?;
        let oh_data = guard.reader.read_bytes(check_len)?;
        drop(guard);

        let checksum = checksum_metadata(&oh_data);

        let checksum_pos = oh_start
            .checked_add(check_len as u64)
            .ok_or_else(|| Error::InvalidFormat("object-header checksum offset overflow".into()))?;
        self.write_handle.seek(SeekFrom::Start(checksum_pos))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;

        Ok(())
    }

    /// Reopen the read handle to pick up file changes.
    pub(super) fn reopen_reader(&mut self) -> Result<()> {
        let read_file = fs::File::open(&self.path)?;
        let mut reader = HdfReader::new(BufReader::new(read_file));
        let superblock = Superblock::read(&mut reader)?;
        self.superblock = superblock.clone();
        *self.inner.lock() = FileInner {
            reader,
            superblock,
            path: Some(self.path.clone()),
            access_plist: crate::hl::plist::file_access::FileAccess::default(),
            dset_no_attrs_hint: false,
            open_objects: HashMap::new(),
            next_object_id: 1,
        };
        Ok(())
    }

    pub(super) fn linear_chunk_index(
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
            .map(|(&dim, &chunk)| {
                if chunk == 0 {
                    return Err(Error::InvalidFormat("zero chunk dimension".into()));
                }
                dim.checked_add(chunk - 1)
                    .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))
                    .map(|extent| extent / chunk)
            })
            .collect::<Result<_>>()?;
        let mut index = 0usize;
        for dim in 0..data_dims.len() {
            let scaled = chunk_coords[dim] / chunk_dims[dim];
            if scaled >= chunks_per_dim[dim] {
                return Err(Error::Unsupported(
                    "fixed-array chunk index updates can replace existing chunks only".into(),
                ));
            }
            let count = usize::try_from(chunks_per_dim[dim])
                .map_err(|_| Error::InvalidFormat("chunks per dimension overflow".into()))?;
            let scaled = usize::try_from(scaled)
                .map_err(|_| Error::InvalidFormat("chunk coordinate overflow".into()))?;
            index = index
                .checked_mul(count)
                .and_then(|value| value.checked_add(scaled))
                .ok_or_else(|| Error::InvalidFormat("chunk index overflow".into()))?;
        }
        Ok(index)
    }

    pub(super) fn filtered_chunk_size_len(
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

    pub(super) fn write_uint_le(&mut self, value: u64, size: usize) -> Result<()> {
        self.write_handle.write_all(&value.to_le_bytes()[..size])?;
        Ok(())
    }

    pub(super) fn read_fresh_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub(super) fn append_aligned_zeros(&mut self, size: usize, align: u64) -> Result<u64> {
        if align == 0 {
            return Err(Error::InvalidFormat("alignment cannot be zero".into()));
        }
        let mut pos = self.write_handle.seek(SeekFrom::End(0))?;
        let padding = (align - (pos % align)) % align;
        if padding != 0 {
            let padding = usize::try_from(padding)
                .map_err(|_| Error::InvalidFormat("alignment padding overflow".into()))?;
            self.write_handle.write_all(&vec![0u8; padding])?;
            pos = pos
                .checked_add(padding as u64)
                .ok_or_else(|| Error::InvalidFormat("aligned append offset overflow".into()))?;
        }
        self.write_handle.write_all(&vec![0u8; size])?;
        Ok(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_chunk_index_rejects_zero_chunk_dimension() {
        let err = MutableFile::linear_chunk_index(&[0], &[10], &[0]).unwrap_err();
        assert!(
            err.to_string().contains("zero chunk dimension"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn linear_chunk_index_rejects_chunk_count_overflow() {
        let err = MutableFile::linear_chunk_index(&[0], &[u64::MAX], &[2]).unwrap_err();
        assert!(
            err.to_string().contains("chunk count overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checksum_offset_addition_rejects_overflow() {
        let err = u64::MAX
            .checked_add(usize::MAX as u64)
            .ok_or_else(|| Error::InvalidFormat("object-header checksum offset overflow".into()))
            .unwrap_err();
        assert!(err.to_string().contains("checksum offset overflow"));
    }

    #[test]
    fn append_aligned_zeros_rejects_zero_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_alignment.h5");
        {
            let mut file = crate::WritableFile::create(&path).unwrap();
            file.flush().unwrap();
        }

        let mut file = MutableFile::open_rw(&path).unwrap();
        let err = file.append_aligned_zeros(1, 0).unwrap_err();
        assert!(
            err.to_string().contains("alignment cannot be zero"),
            "unexpected error: {err}"
        );
    }
}
