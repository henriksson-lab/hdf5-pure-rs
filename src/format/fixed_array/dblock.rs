//! Fixed array data block — mirrors libhdf5's `H5FAdblock.c` plus the
//! data-block half of `H5FAcache.c` (`H5FA__cache_dblock_deserialize`).
//! Page-init + per-page iteration (libhdf5's `H5FAdblkpage.c`) is folded
//! in here because the Rust port doesn't model pages as a separate cache
//! entry.

#![allow(dead_code)]

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::hdr::FixedArrayHeader;
use super::{
    bit_is_set, checked_u64_add, checked_usize_add, checked_usize_mul, u64_from_usize,
    FixedArrayElement,
};

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

pub(super) fn dblock_debug(prefix: &FixedArrayDataBlockPrefix) -> String {
    format!(
        "FixedArrayDataBlockPrefix(paginated={}, pages={}, prefix_size={}, raw_element_size={}, page_elements={}, element_count={})",
        prefix.paginated,
        prefix.pages,
        prefix.prefix_size,
        prefix.raw_element_size,
        prefix.page_elements,
        prefix.element_count
    )
}

pub(super) fn cache_dblock_image_len(prefix: &FixedArrayDataBlockPrefix) -> Result<usize> {
    if prefix.paginated {
        Ok(prefix.prefix_size)
    } else {
        checked_usize_mul(
            prefix.element_count,
            prefix.raw_element_size,
            "fixed array data block image length",
        )
        .and_then(|value| checked_usize_add(value, 4, "fixed array data block image length"))
    }
}

pub(super) fn cache_dblock_serialize(prefix_and_payload: &[u8]) -> Vec<u8> {
    let mut out = prefix_and_payload.to_vec();
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

pub(super) fn cache_dblock_notify(_prefix: &FixedArrayDataBlockPrefix) {}

pub(super) fn cache_dblock_free_icr(_prefix: FixedArrayDataBlockPrefix) {}

pub(super) fn cache_dblock_fsf_size(prefix: &FixedArrayDataBlockPrefix) -> usize {
    prefix.prefix_size
}

pub(super) fn cache_dblk_page_get_initial_load_size() -> usize {
    4
}

pub(super) fn cache_dblk_page_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "fixed array data block page")
}

pub(super) fn cache_dblk_page_deserialize(payload: &[u8]) -> Result<&[u8]> {
    if payload.len() < 4 {
        return Err(Error::InvalidFormat(
            "fixed array data block page is truncated".into(),
        ));
    }
    cache_dblk_page_verify_chksum(payload)?;
    Ok(&payload[..payload.len() - 4])
}

pub(super) fn cache_dblk_page_image_len(payload_len: usize) -> Result<usize> {
    payload_len.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat("fixed array data block page image length overflow".into())
    })
}

pub(super) fn cache_dblk_page_serialize(payload: &[u8]) -> Vec<u8> {
    let mut out = payload.to_vec();
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

pub(super) fn cache_dblk_page_notify(_page_index: usize) {}

pub(super) fn cache_dblk_page_free_icr(_payload: Vec<u8>) {}

pub(super) fn dblk_page_alloc(size: usize) -> Vec<u8> {
    vec![0; size]
}

pub(super) fn dblk_page_protect(payload: &[u8]) -> &[u8] {
    payload
}

pub(super) fn dblk_page_unprotect(_payload: &[u8]) {}

pub(super) fn dblk_page_dest(_payload: Vec<u8>) {}

pub(super) fn dblock_alloc(
    paginated: bool,
    pages: usize,
    raw_element_size: usize,
    page_elements: usize,
    element_count: usize,
) -> FixedArrayDataBlockPrefix {
    FixedArrayDataBlockPrefix {
        paginated,
        pages,
        page_init: if paginated {
            vec![0; pages.div_ceil(8)]
        } else {
            Vec::new()
        },
        prefix_size: 0,
        raw_element_size,
        page_elements,
        element_count,
    }
}

pub(super) fn dblock_create(prefix: FixedArrayDataBlockPrefix) -> FixedArrayDataBlockPrefix {
    prefix
}

pub(super) fn dblock_unprotect(_prefix: FixedArrayDataBlockPrefix) {}

pub(super) fn dblock_delete(prefix: &mut FixedArrayDataBlockPrefix) {
    prefix.page_init.clear();
    prefix.element_count = 0;
}

pub(super) fn dblock_dest(_prefix: FixedArrayDataBlockPrefix) {}

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
        let prefix_size = checked_usize_add(
            4 + 1 + 1,
            reader.sizeof_addr() as usize,
            "fixed array data block prefix size",
        )
        .and_then(|value| {
            checked_usize_add(value, page_init_size, "fixed array data block prefix size")
        })
        .and_then(|value| checked_usize_add(value, 4, "fixed array data block prefix size"))?;
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

fn verify_trailing_checksum(data: &[u8], context: &str) -> Result<()> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(format!("{context} image too short")));
    }
    let split = data.len() - 4;
    let stored = u32::from_le_bytes(
        data[split..]
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} checksum is truncated")))?,
    );
    let computed = crate::format::checksum::checksum_metadata(&data[..split]);
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
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
        let page_payload = checked_usize_mul(
            prefix.page_elements,
            prefix.raw_element_size,
            "fixed array data block page size",
        )?;
        let page_size = checked_usize_add(page_payload, 4, "fixed array data block page size")?;
        for page_index in 0..prefix.pages {
            let page_start = checked_usize_mul(
                page_index,
                prefix.page_elements,
                "fixed array page start index",
            )?;
            let remaining = prefix
                .element_count
                .checked_sub(page_start)
                .ok_or_else(|| {
                    Error::InvalidFormat("fixed array page start exceeds element count".into())
                })?;
            let page_count = prefix.page_elements.min(remaining);
            if bit_is_set(&prefix.page_init, page_index) {
                let page_offset =
                    checked_usize_mul(page_index, page_size, "fixed array data block page offset")?;
                let offset = checked_usize_add(
                    prefix.prefix_size,
                    page_offset,
                    "fixed array data block page offset",
                )?;
                let page_addr = checked_u64_add(
                    header.data_block_addr,
                    u64_from_usize(offset, "fixed array data block page offset")?,
                    "fixed array data block page address",
                )?;
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
