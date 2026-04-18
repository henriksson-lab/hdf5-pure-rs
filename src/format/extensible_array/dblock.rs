//! Extensible array data block — mirrors libhdf5's `H5EAdblock.c` plus
//! the data-block half of `H5EAcache.c`. The page handling that lives
//! in libhdf5's `H5EAdblkpage.c` is folded in here because the Rust port
//! doesn't model pages as a separate cache entry.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader};

use super::fixed_array::FixedArrayElement;
use super::hdr::ExtensibleArrayHeader;

/// Decoded extensible-array data-block prefix: page count + raw geometry
/// numbers, all derived from the on-disk magic+version+class+owner block
/// header. Mirrors `H5EA__cache_dblock_deserialize` — pure parse, no I/O
/// over the element pages themselves.
pub(super) struct ExtArrayDataBlockPrefix {
    /// Number of pages this data block is split into (0 = unpaginated).
    pub(super) pages: usize,
    /// Total prefix size on disk (used to compute per-page offsets).
    pub(super) prefix_size: usize,
}

/// Pure prefix decode for an extensible-array data block.
pub(super) fn decode_data_block_prefix<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    data_block_addr: u64,
    data_block_elements: usize,
) -> Result<ExtArrayDataBlockPrefix> {
    reader.seek(data_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EADB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array data block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array data block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array data block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array data block owner address does not match header".into(),
        ));
    }

    let _block_offset = reader.read_uint(header.array_offset_size)?;
    let pages = super::data_block_pages(header, data_block_elements);
    let prefix_size =
        4 + 1 + 1 + reader.sizeof_addr() as usize + header.array_offset_size as usize + 4;
    Ok(ExtArrayDataBlockPrefix { pages, prefix_size })
}

/// Drive a decoded data-block prefix to push `count` elements onto the
/// shared output vector. C-side analogue: the iteration done inside
/// `H5EA_iterate` after a page or block has been protected.
#[allow(clippy::too_many_arguments)]
pub(super) fn append_data_block_elements<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    data_block_addr: u64,
    data_block_elements: usize,
    page_init: Option<&[u8]>,
    count: usize,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    if is_undef_addr(data_block_addr) {
        super::append_fill_elements(header, count, elements)?;
        return Ok(());
    }

    let prefix =
        decode_data_block_prefix(reader, header_addr, header, data_block_addr, data_block_elements)?;
    if prefix.pages == 0 {
        for _ in 0..count {
            elements.push(read_element(reader, filtered, chunk_size_len)?);
        }
        let unread = data_block_elements.saturating_sub(count);
        if unread > 0 {
            reader.skip((unread * header.raw_element_size) as u64)?;
        }
        let _checksum = reader.read_u32()?;
    } else {
        let page_size = header.data_block_page_elements * header.raw_element_size + 4;
        let mut remaining = count;
        for page_index in 0..prefix.pages {
            if remaining == 0 {
                break;
            }
            let page_elements = header.data_block_page_elements.min(remaining);
            let page_addr =
                data_block_addr + prefix.prefix_size as u64 + (page_index * page_size) as u64;
            let page_initialized = page_init
                .map(|bits| super::bit_is_set(bits, page_index))
                .unwrap_or(true);
            if page_initialized {
                reader.seek(page_addr)?;
                for _ in 0..page_elements {
                    elements.push(read_element(reader, filtered, chunk_size_len)?);
                }
            } else {
                super::append_fill_elements(header, page_elements, elements)?;
            }
            remaining -= page_elements;
        }
    }

    Ok(())
}

pub(super) fn read_element<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<FixedArrayElement> {
    let addr = reader.read_addr()?;
    if filtered {
        let nbytes = reader.read_uint(chunk_size_len as u8)?;
        let filter_mask = reader.read_u32()?;
        Ok(FixedArrayElement {
            addr,
            nbytes: Some(nbytes),
            filter_mask,
        })
    } else {
        Ok(FixedArrayElement {
            addr,
            nbytes: None,
            filter_mask: 0,
        })
    }
}
