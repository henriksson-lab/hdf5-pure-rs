//! Fractal heap header — mirrors libhdf5's `H5HFhdr.c` plus the
//! header-half of `H5HFcache.c` (`H5HF__cache_hdr_deserialize`).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::io::reader::HdfReader;

use super::{FractalHeapHeader, FRHP_MAGIC};

impl FractalHeapHeader {
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != FRHP_MAGIC {
            return Err(Error::InvalidFormat("invalid fractal heap magic".into()));
        }

        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "fractal heap version {version}"
            )));
        }

        let heap_id_len = reader.read_u16()?;
        let io_filter_len = reader.read_u16()?;
        let flags = reader.read_u8()?;

        // "Huge" object info
        let max_managed_obj_size = reader.read_u32()?;
        let _next_huge_id = reader.read_length()?; // sizeof_size
        let huge_btree_addr = reader.read_addr()?; // sizeof_addr

        // Managed free space
        let _total_man_free = reader.read_length()?; // sizeof_size
        let _fs_addr = reader.read_addr()?; // sizeof_addr

        // Heap statistics
        let _man_size = reader.read_length()?;
        let _man_alloc_size = reader.read_length()?;
        let _man_iter_off = reader.read_length()?;
        let num_managed_objects = reader.read_length()?;
        let _huge_size = reader.read_length()?;
        let _huge_nobjs = reader.read_length()?;
        let _tiny_size = reader.read_length()?;
        let _tiny_nobjs = reader.read_length()?;

        // Doubling table info
        let table_width = reader.read_u16()?;
        let start_block_size = reader.read_length()?;
        let max_direct_block_size = reader.read_length()?;
        let max_heap_size = reader.read_u16()?;
        let start_root_rows = reader.read_u16()?;
        let root_block_addr = reader.read_addr()?;
        let current_root_rows = reader.read_u16()?;

        let has_checksum = flags & 0x02 != 0;

        // If I/O filters present, skip filter info
        let mut root_direct_filtered_size = None;
        let mut root_direct_filter_mask = 0;
        let mut filter_pipeline = None;
        if io_filter_len > 0 {
            root_direct_filtered_size = Some(reader.read_length()?);
            root_direct_filter_mask = reader.read_u32()?;
            let pipeline_bytes = reader.read_bytes(io_filter_len as usize)?;
            filter_pipeline = Some(FilterPipelineMessage::decode(&pipeline_bytes)?);
        }

        // Checksum
        let _checksum = reader.read_u32()?;

        validate_doubling_table_geometry(
            table_width,
            start_block_size,
            max_direct_block_size,
            max_heap_size,
        )?;

        Ok(Self {
            heap_id_len,
            heap_addr: addr,
            io_filter_len,
            flags,
            max_managed_obj_size,
            table_width,
            start_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            root_block_addr,
            current_root_rows,
            num_managed_objects,
            has_checksum,
            sizeof_addr: reader.sizeof_addr(),
            sizeof_size: reader.sizeof_size(),
            huge_btree_addr,
            root_direct_filtered_size,
            root_direct_filter_mask,
            filter_pipeline,
        })
    }
}

fn validate_doubling_table_geometry(
    table_width: u16,
    start_block_size: u64,
    max_direct_block_size: u64,
    max_heap_size: u16,
) -> Result<()> {
    if table_width == 0 {
        return Err(Error::InvalidFormat(
            "fractal heap table width must be nonzero".into(),
        ));
    }
    if start_block_size == 0 || !start_block_size.is_power_of_two() {
        return Err(Error::InvalidFormat(
            "fractal heap start block size must be a nonzero power of two".into(),
        ));
    }
    if max_direct_block_size == 0 || !max_direct_block_size.is_power_of_two() {
        return Err(Error::InvalidFormat(
            "fractal heap max direct block size must be a nonzero power of two".into(),
        ));
    }
    if max_direct_block_size < start_block_size {
        return Err(Error::InvalidFormat(
            "fractal heap max direct block size is smaller than start block size".into(),
        ));
    }
    if max_heap_size > 64 {
        return Err(Error::Unsupported(format!(
            "fractal heap max heap size {max_heap_size} exceeds 64-bit offsets"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_doubling_table_geometry;

    #[test]
    fn rejects_invalid_doubling_table_geometry() {
        assert!(validate_doubling_table_geometry(0, 8, 64, 32).is_err());
        assert!(validate_doubling_table_geometry(4, 0, 64, 32).is_err());
        assert!(validate_doubling_table_geometry(4, 12, 64, 32).is_err());
        assert!(validate_doubling_table_geometry(4, 64, 8, 32).is_err());
        assert!(validate_doubling_table_geometry(4, 8, 64, 65).is_err());
    }

    #[test]
    fn accepts_valid_doubling_table_geometry() {
        validate_doubling_table_geometry(4, 8, 64, 32).unwrap();
    }
}
