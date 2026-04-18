//! Extensible array header — mirrors libhdf5's `H5EAhdr.c` plus the
//! header-half of `H5EAcache.c` (`H5EA__cache_hdr_deserialize`).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

const MAX_EXTENSIBLE_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
pub(super) struct ExtensibleArrayHeader {
    pub(super) class_id: u8,
    pub(super) raw_element_size: usize,
    pub(super) index_block_elements: u8,
    pub(super) max_index_set: u64,
    pub(super) index_block_addr: u64,
    pub(super) array_offset_size: u8,
    pub(super) data_block_page_elements: usize,
    pub(super) index_block_super_blocks: usize,
    pub(super) index_block_data_block_addrs: usize,
    pub(super) index_block_super_block_addrs: usize,
    pub(super) super_block_info: Vec<SuperBlockInfo>,
}

#[derive(Debug, Clone)]
pub(super) struct SuperBlockInfo {
    pub(super) data_blocks: usize,
    pub(super) data_block_elements: usize,
    pub(super) start_data_block: u64,
}

pub(super) fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ExtensibleArrayHeader> {
    reader.seek(addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EAHD" {
        return Err(Error::InvalidFormat(
            "invalid extensible array header magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array header version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    let raw_element_size = reader.read_u8()? as usize;
    let max_elements_bits = reader.read_u8()?;
    let index_block_elements = reader.read_u8()?;
    let data_block_min_elements = reader.read_u8()?;
    let super_block_min_data_ptrs = reader.read_u8()?;
    let max_data_block_page_elements_bits = reader.read_u8()?;

    let _super_block_count = reader.read_length()?;
    let _super_block_size = reader.read_length()?;
    let _data_block_count = reader.read_length()?;
    let _data_block_size = reader.read_length()?;
    let max_index_set = reader.read_length()?;
    let max_index_count = super::usize_from_u64(max_index_set, "extensible array max index")?;
    if max_index_count > MAX_EXTENSIBLE_ARRAY_ELEMENTS {
        return Err(Error::InvalidFormat(format!(
            "extensible array max index {max_index_count} exceeds supported maximum {MAX_EXTENSIBLE_ARRAY_ELEMENTS}"
        )));
    }
    let _realized_elements = reader.read_length()?;
    let index_block_addr = reader.read_addr()?;
    verify_checksum(reader, addr, "extensible array header")?;

    let array_offset_size = max_elements_bits.div_ceil(8);
    let data_block_page_elements = 1usize
        .checked_shl(max_data_block_page_elements_bits as u32)
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array page element count overflow".into())
        })?;
    let super_block_count = 1usize
        + (max_elements_bits as usize)
            .checked_sub(super::log2_power2(data_block_min_elements as u64)?)
            .ok_or_else(|| {
                Error::InvalidFormat("invalid extensible array block parameters".into())
            })?;
    let index_block_super_blocks = 2 * super::log2_power2(super_block_min_data_ptrs as u64)?;
    let index_block_data_block_addrs = 2 * (super_block_min_data_ptrs as usize - 1);
    let index_block_super_block_addrs = super_block_count
        .checked_sub(index_block_super_blocks)
        .ok_or_else(|| {
            Error::InvalidFormat("invalid extensible array super block layout".into())
        })?;
    let super_block_info =
        build_super_block_info(super_block_count, data_block_min_elements as usize)?;

    Ok(ExtensibleArrayHeader {
        class_id,
        raw_element_size,
        index_block_elements,
        max_index_set,
        index_block_addr,
        array_offset_size,
        data_block_page_elements,
        index_block_super_blocks,
        index_block_data_block_addrs,
        index_block_super_block_addrs,
        super_block_info,
    })
}

pub(super) fn verify_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    context: &str,
) -> Result<()> {
    let checksum_pos = reader.position()?;
    let stored_checksum = reader.read_u32()?;
    let check_len = usize::try_from(checksum_pos - start)
        .map_err(|_| Error::InvalidFormat(format!("{context} checksum span is too large")))?;
    reader.seek(start)?;
    let check_data = reader.read_bytes(check_len)?;
    let computed = checksum_metadata(&check_data);
    if stored_checksum != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}"
        )));
    }
    reader.seek(checksum_pos + 4)?;
    Ok(())
}

fn build_super_block_info(
    count: usize,
    min_data_block_elements: usize,
) -> Result<Vec<SuperBlockInfo>> {
    let mut infos = Vec::with_capacity(count);
    let mut start_index = 0u64;
    let mut start_data_block = 0u64;
    for index in 0..count {
        let data_blocks = 1usize.checked_shl((index / 2) as u32).ok_or_else(|| {
            Error::InvalidFormat("extensible array data block count overflow".into())
        })?;
        let data_block_elements = min_data_block_elements
            .checked_mul(
                1usize
                    .checked_shl(index.div_ceil(2) as u32)
                    .ok_or_else(|| {
                        Error::InvalidFormat(
                            "extensible array data block element count overflow".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block size overflow".into())
            })?;
        infos.push(SuperBlockInfo {
            data_blocks,
            data_block_elements,
            start_data_block,
        });
        start_index = start_index
            .checked_add((data_blocks as u64) * (data_block_elements as u64))
            .ok_or_else(|| Error::InvalidFormat("extensible array start index overflow".into()))?;
        start_data_block = start_data_block
            .checked_add(data_blocks as u64)
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block index overflow".into())
            })?;
    }
    Ok(infos)
}
