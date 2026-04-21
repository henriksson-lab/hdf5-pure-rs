//! Extensible array index block — mirrors libhdf5's `H5EAiblock.c` plus
//! the iblock half of `H5EAcache.c` (`H5EA__cache_iblock_deserialize`).
//! Includes the spillover descent (the post-iblock walk into super-blocks
//! and data-blocks) that `H5EA_iterate` performs after `H5EA__iblock_protect`.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::dblock::{append_data_block_elements, read_element};
use super::fixed_array::FixedArrayElement;
use super::hdr::ExtensibleArrayHeader;
use super::sblock::read_super_block;

/// Decoded extensible-array index block: the inline elements plus the
/// data-block and super-block address tables. Mirrors
/// `H5EA__cache_iblock_deserialize` in libhdf5: parse the prefix and the
/// variable-length tables, but don't dereference the listed addresses.
pub(super) struct ExtArrayIndexBlock {
    /// Inline elements stored directly in the index block.
    pub(super) elements: Vec<FixedArrayElement>,
    /// Addresses of the data blocks owned directly by the index block.
    pub(super) data_block_addrs: Vec<u64>,
    /// Addresses of the super-blocks owned by the index block.
    pub(super) super_block_addrs: Vec<u64>,
}

/// Pure deserializer for the extensible-array index block.
pub(super) fn decode_index_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<ExtArrayIndexBlock> {
    reader.seek(header.index_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EAIB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array index block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array index block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array index block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array index block owner address does not match header".into(),
        ));
    }

    let expected_element_size = if filtered {
        reader.sizeof_addr() as usize + chunk_size_len + 4
    } else {
        reader.sizeof_addr() as usize
    };
    if header.raw_element_size != expected_element_size {
        return Err(Error::InvalidFormat(format!(
            "extensible array raw element size {} does not match expected {}",
            header.raw_element_size, expected_element_size
        )));
    }

    let max_index_count =
        super::usize_from_u64(header.max_index_set, "extensible array max index")?;
    let mut elements = Vec::with_capacity(max_index_count);
    for idx in 0..header.index_block_elements {
        let element = read_element(reader, filtered, chunk_size_len)?;
        if (idx as u64) < header.max_index_set {
            elements.push(element);
        }
    }

    let mut data_block_addrs = Vec::with_capacity(header.index_block_data_block_addrs);
    for _ in 0..header.index_block_data_block_addrs {
        data_block_addrs.push(reader.read_addr()?);
    }

    let mut super_block_addrs = Vec::with_capacity(header.index_block_super_block_addrs);
    for _ in 0..header.index_block_super_block_addrs {
        super_block_addrs.push(reader.read_addr()?);
    }

    let _checksum = reader.read_u32()?;
    Ok(ExtArrayIndexBlock {
        elements,
        data_block_addrs,
        super_block_addrs,
    })
}

/// Drive the decoded index block to materialize the full element vector
/// — descends into spillover data/super blocks. Composition mirrors
/// the C-side `H5EA_iterate` after `H5EA__iblock_protect` returns.
pub(super) fn read_index_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let iblock = decode_index_block(reader, header_addr, header, filtered, chunk_size_len)?;
    let ExtArrayIndexBlock {
        mut elements,
        data_block_addrs,
        super_block_addrs,
    } = iblock;
    read_spillover_blocks(
        reader,
        header_addr,
        header,
        filtered,
        chunk_size_len,
        &data_block_addrs,
        &super_block_addrs,
        &mut elements,
    )?;
    Ok(elements)
}

#[allow(clippy::too_many_arguments)]
fn read_spillover_blocks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    data_block_addrs: &[u64],
    super_block_addrs: &[u64],
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    for (super_block_index, info) in header.super_block_info.iter().enumerate() {
        if elements.len() as u64 >= header.max_index_set {
            break;
        }

        if super_block_index < header.index_block_super_blocks {
            for local_data_block in 0..info.data_blocks {
                if elements.len() as u64 >= header.max_index_set {
                    break;
                }

                let data_block_index = info.start_data_block as usize + local_data_block;
                let Some(&data_block_addr) = data_block_addrs.get(data_block_index) else {
                    return Err(Error::InvalidFormat(
                        "extensible array data block address index out of bounds".into(),
                    ));
                };
                let remaining = super::usize_from_u64(
                    header.max_index_set - elements.len() as u64,
                    "extensible array remaining element count",
                )?;
                let count = info.data_block_elements.min(remaining);
                append_data_block_elements(
                    reader,
                    header_addr,
                    header,
                    filtered,
                    chunk_size_len,
                    data_block_addr,
                    info.data_block_elements,
                    None,
                    count,
                    elements,
                )?;
            }
        } else {
            let super_block_addr_index = super_block_index - header.index_block_super_blocks;
            let Some(&super_block_addr) = super_block_addrs.get(super_block_addr_index) else {
                return Err(Error::InvalidFormat(
                    "extensible array super block address index out of bounds".into(),
                ));
            };
            read_super_block(
                reader,
                header_addr,
                header,
                filtered,
                chunk_size_len,
                super_block_addr,
                info,
                elements,
            )?;
        }
    }

    Ok(())
}
