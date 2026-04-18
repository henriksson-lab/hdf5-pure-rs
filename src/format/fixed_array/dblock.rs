//! Fixed array data block — mirrors libhdf5's `H5FAdblock.c` plus the
//! data-block half of `H5FAcache.c` (`H5FA__cache_dblock_deserialize`).
//! Page-init + per-page iteration (libhdf5's `H5FAdblkpage.c`) is folded
//! in here because the Rust port doesn't model pages as a separate cache
//! entry.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::hdr::FixedArrayHeader;
use super::{bit_is_set, FixedArrayElement};

/// Decoded fixed-array data-block prefix — magic, version, class id,
/// owner-address validation, page-init bitmap (if paginated), and the
/// computed page-layout numbers. Mirrors `H5FA__cache_dblock_deserialize`
/// in libhdf5: pure parsing of the fixed-size header, no element walk.
pub(super) struct FixedArrayDataBlockPrefix {
    /// Whether the elements are split into pages (vs a single contiguous run).
    pub(super) paginated: bool,
    /// Number of pages (only meaningful when `paginated`).
    pub(super) pages: usize,
    /// Page-initialized bitmap (one bit per page).
    pub(super) page_init: Vec<u8>,
    /// Total size of the on-disk header (used to compute page addresses).
    pub(super) prefix_size: usize,
    /// On-disk size of one element record (filtered vs unfiltered).
    pub(super) raw_element_size: usize,
    /// Per-page element count (`1 << max_page_elements_bits`).
    pub(super) page_elements: usize,
    /// Total element count.
    pub(super) element_count: usize,
}

/// Pure prefix decode + sanity validation for a fixed-array data block.
pub(super) fn decode_data_block_prefix<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &FixedArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<FixedArrayDataBlockPrefix> {
    reader.seek(header.data_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"FADB" {
        return Err(Error::InvalidFormat(
            "invalid fixed array data block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "fixed array data block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "fixed array data block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "fixed array data block owner address does not match header".into(),
        ));
    }

    let page_elements = 1usize
        .checked_shl(header.max_page_elements_bits as u32)
        .ok_or_else(|| Error::InvalidFormat("fixed array page size overflow".into()))?;
    let expected_element_size = if filtered {
        reader.sizeof_addr() as usize + chunk_size_len + 4
    } else {
        reader.sizeof_addr() as usize
    };
    if header.raw_element_size != expected_element_size {
        return Err(Error::InvalidFormat(format!(
            "fixed array raw element size {} does not match expected {}",
            header.raw_element_size, expected_element_size
        )));
    }

    let element_count = super::usize_from_u64(header.elements, "fixed array element count")?;
    let paginated = element_count > page_elements;
    if paginated {
        let pages = element_count.div_ceil(page_elements);
        let page_init_size = pages.div_ceil(8);
        let page_init = reader.read_bytes(page_init_size)?;
        let _checksum = reader.read_u32()?;
        let prefix_size = 4 + 1 + 1 + reader.sizeof_addr() as usize + page_init_size + 4;
        Ok(FixedArrayDataBlockPrefix {
            paginated: true,
            pages,
            page_init,
            prefix_size,
            raw_element_size: header.raw_element_size,
            page_elements,
            element_count,
        })
    } else {
        Ok(FixedArrayDataBlockPrefix {
            paginated: false,
            pages: 0,
            page_init: Vec::new(),
            prefix_size: 0,
            raw_element_size: header.raw_element_size,
            page_elements,
            element_count,
        })
    }
}

/// Walk a decoded prefix to materialize the element vector.
pub(super) fn collect_data_block_elements<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &FixedArrayHeader,
    prefix: &FixedArrayDataBlockPrefix,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let mut elements = Vec::with_capacity(prefix.element_count);
    if prefix.paginated {
        let page_size = prefix.page_elements * prefix.raw_element_size + 4;
        for page_index in 0..prefix.pages {
            let page_start = page_index * prefix.page_elements;
            let page_count = prefix.page_elements.min(prefix.element_count - page_start);
            if bit_is_set(&prefix.page_init, page_index) {
                let page_addr = header.data_block_addr
                    + prefix.prefix_size as u64
                    + (page_index * page_size) as u64;
                reader.seek(page_addr)?;
                for _ in 0..page_count {
                    elements.push(read_element(reader, filtered, chunk_size_len)?);
                }
                let _page_checksum = reader.read_u32()?;
            } else {
                super::append_fill_elements(page_count, &mut elements);
            }
        }
    } else {
        for _ in 0..header.elements {
            elements.push(read_element(reader, filtered, chunk_size_len)?);
        }
        let _checksum = reader.read_u32()?;
    }
    Ok(elements)
}

/// Drive `decode_data_block_prefix` + `collect_data_block_elements` —
/// the C-side composition is `H5FA__cache_dblock_deserialize` followed by
/// the iterate path in `H5FA_iterate`.
pub(super) fn read_data_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &FixedArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let prefix = decode_data_block_prefix(reader, header_addr, header, filtered, chunk_size_len)?;
    collect_data_block_elements(reader, header, &prefix, filtered, chunk_size_len)
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
