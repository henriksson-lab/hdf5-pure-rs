//! Extensible array super-block — mirrors libhdf5's `H5EAsblock.c` plus
//! the super-block half of `H5EAcache.c`
//! (`H5EA__cache_sblock_deserialize`).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader};

use super::dblock::append_data_block_elements;
use super::fixed_array::FixedArrayElement;
use super::hdr::{ExtensibleArrayHeader, SuperBlockInfo};

/// Decoded extensible-array super-block contents — page-init bitmap and
/// the data-block address table. Mirrors `H5EA__cache_sblock_deserialize`:
/// magic+version+class+owner+offset are validated, all variable-length
/// arrays are pulled, but no descent into the listed data blocks happens.
pub(super) struct ExtArraySuperBlock {
    /// Concatenated page-init bytes; one slice of `page_init_size` bytes
    /// per data block, or empty when the data blocks aren't paginated.
    pub(super) page_init: Vec<u8>,
    /// Bytes of page-init data per data block (0 if unpaginated).
    pub(super) page_init_size: usize,
    /// Addresses of the data blocks owned by this super-block.
    pub(super) data_block_addrs: Vec<u64>,
}

/// Pure deserializer for an extensible-array super-block.
pub(super) fn decode_super_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    super_block_addr: u64,
    info: &SuperBlockInfo,
) -> Result<ExtArraySuperBlock> {
    reader.seek(super_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EASB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array super block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array super block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array super block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array super block owner address does not match header".into(),
        ));
    }

    let _block_offset = reader.read_uint(header.array_offset_size)?;
    let data_block_pages = super::data_block_pages(header, info.data_block_elements);
    let page_init_size = if data_block_pages > 0 {
        data_block_pages.div_ceil(8)
    } else {
        0
    };
    let page_init = if page_init_size > 0 {
        reader.read_bytes(info.data_blocks * page_init_size)?
    } else {
        Vec::new()
    };

    let mut data_block_addrs = Vec::with_capacity(info.data_blocks);
    for _ in 0..info.data_blocks {
        data_block_addrs.push(reader.read_addr()?);
    }

    let _checksum = reader.read_u32()?;
    Ok(ExtArraySuperBlock {
        page_init,
        page_init_size,
        data_block_addrs,
    })
}

/// Walk a decoded super-block: descend into each owned data block and
/// stream elements into the shared output vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_super_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    super_block_addr: u64,
    info: &SuperBlockInfo,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    if is_undef_addr(super_block_addr) {
        super::append_fill_elements(
            header,
            info.data_blocks * info.data_block_elements,
            elements,
        )?;
        return Ok(());
    }

    let sblock = decode_super_block(reader, header_addr, header, super_block_addr, info)?;

    for (data_block_index, &data_block_addr) in sblock.data_block_addrs.iter().enumerate() {
        if elements.len() as u64 >= header.max_index_set {
            break;
        }
        let remaining = super::usize_from_u64(
            header.max_index_set - elements.len() as u64,
            "extensible array remaining element count",
        )?;
        let count = info.data_block_elements.min(remaining);
        let page_init_for_block = if sblock.page_init_size > 0 {
            Some(
                &sblock.page_init[data_block_index * sblock.page_init_size
                    ..(data_block_index + 1) * sblock.page_init_size],
            )
        } else {
            None
        };
        append_data_block_elements(
            reader,
            header_addr,
            header,
            filtered,
            chunk_size_len,
            data_block_addr,
            info.data_block_elements,
            page_init_for_block,
            count,
            elements,
        )?;
    }

    Ok(())
}
