//! Extensible array header — mirrors libhdf5's `H5EAhdr.c` plus the
//! header-half of `H5EAcache.c` (`H5EA__cache_hdr_deserialize`).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

const MAX_EXTENSIBLE_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
pub(crate) struct ParsedExtensibleArrayHeader {
    pub(crate) class_id: u8,
    pub(crate) raw_element_size: usize,
    pub(crate) index_block_elements: u8,
    pub(crate) data_block_min_elements: usize,
    pub(crate) super_block_min_data_ptrs: usize,
    pub(crate) super_block_count: u64,
    pub(crate) super_block_size: u64,
    pub(crate) data_block_count: u64,
    pub(crate) data_block_size: u64,
    pub(crate) max_index_set: u64,
    pub(crate) realized_elements: u64,
    pub(crate) index_block_addr: u64,
    pub(crate) array_offset_size: u8,
    pub(crate) data_block_page_elements: usize,
    pub(crate) index_block_super_blocks: usize,
    pub(crate) index_block_data_block_addrs: usize,
    pub(crate) index_block_super_block_addrs: usize,
    pub(crate) derived_super_block_count: usize,
    pub(crate) super_block_info: Vec<SuperBlockInfo>,
    pub(crate) checksum_pos: u64,
    pub(crate) super_block_count_pos: u64,
    pub(crate) super_block_size_pos: u64,
    pub(crate) data_block_count_pos: u64,
    pub(crate) data_block_size_pos: u64,
    pub(crate) max_index_set_pos: u64,
    pub(crate) realized_elements_pos: u64,
}

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
pub(crate) struct SuperBlockInfo {
    pub(super) data_blocks: usize,
    pub(super) data_block_elements: usize,
    pub(super) start_data_block: u64,
}

pub(super) fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ExtensibleArrayHeader> {
    let parsed = read_header_core(reader, addr)?;

    Ok(ExtensibleArrayHeader {
        class_id: parsed.class_id,
        raw_element_size: parsed.raw_element_size,
        index_block_elements: parsed.index_block_elements,
        max_index_set: parsed.max_index_set,
        index_block_addr: parsed.index_block_addr,
        array_offset_size: parsed.array_offset_size,
        data_block_page_elements: parsed.data_block_page_elements,
        index_block_super_blocks: parsed.index_block_super_blocks,
        index_block_data_block_addrs: parsed.index_block_data_block_addrs,
        index_block_super_block_addrs: parsed.index_block_super_block_addrs,
        super_block_info: parsed.super_block_info,
    })
}

pub(crate) fn read_header_core<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ParsedExtensibleArrayHeader> {
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

    let super_block_count_pos = reader.position()?;
    let stored_super_block_count = reader.read_length()?;
    let super_block_size_pos = reader.position()?;
    let super_block_size = reader.read_length()?;
    let data_block_count_pos = reader.position()?;
    let data_block_count = reader.read_length()?;
    let data_block_size_pos = reader.position()?;
    let data_block_size = reader.read_length()?;
    let max_index_set_pos = reader.position()?;
    let max_index_set = reader.read_length()?;
    let max_index_count = super::usize_from_u64(max_index_set, "extensible array max index")?;
    if max_index_count > MAX_EXTENSIBLE_ARRAY_ELEMENTS {
        return Err(Error::InvalidFormat(format!(
            "extensible array max index {max_index_count} exceeds supported maximum {MAX_EXTENSIBLE_ARRAY_ELEMENTS}"
        )));
    }
    let realized_elements_pos = reader.position()?;
    let realized_elements = reader.read_length()?;
    let index_block_addr = reader.read_addr()?;
    let checksum_pos = reader.position()?;
    verify_checksum(reader, addr, "extensible array header")?;

    if index_block_elements == 0
        || data_block_min_elements == 0
        || !data_block_min_elements.is_power_of_two()
        || !super_block_min_data_ptrs.is_power_of_two()
    {
        return Err(Error::InvalidFormat(
            "invalid extensible array block parameters".into(),
        ));
    }

    let array_offset_size = max_elements_bits.div_ceil(8);
    let data_block_page_elements = 1usize
        .checked_shl(max_data_block_page_elements_bits as u32)
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array page element count overflow".into())
        })?;
    let derived_super_block_count = 1usize
        + (max_elements_bits as usize)
            .checked_sub(super::log2_power2(data_block_min_elements as u64)?)
            .ok_or_else(|| {
                Error::InvalidFormat("invalid extensible array block parameters".into())
            })?;
    let index_block_super_blocks = 2 * super::log2_power2(super_block_min_data_ptrs as u64)?;
    let index_block_data_block_addrs = 2 * (super_block_min_data_ptrs as usize - 1);
    let index_block_super_block_addrs = derived_super_block_count
        .checked_sub(index_block_super_blocks)
        .ok_or_else(|| {
            Error::InvalidFormat("invalid extensible array super block layout".into())
        })?;
    let super_block_info =
        build_super_block_info(derived_super_block_count, data_block_min_elements as usize)?;

    Ok(ParsedExtensibleArrayHeader {
        class_id,
        raw_element_size,
        index_block_elements,
        data_block_min_elements: data_block_min_elements as usize,
        super_block_min_data_ptrs: super_block_min_data_ptrs as usize,
        super_block_count: stored_super_block_count,
        super_block_size,
        data_block_count,
        data_block_size,
        max_index_set,
        realized_elements,
        index_block_addr,
        array_offset_size,
        data_block_page_elements,
        index_block_super_blocks,
        index_block_data_block_addrs,
        index_block_super_block_addrs,
        derived_super_block_count,
        super_block_info,
        checksum_pos,
        super_block_count_pos,
        super_block_size_pos,
        data_block_count_pos,
        data_block_size_pos,
        max_index_set_pos,
        realized_elements_pos,
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
