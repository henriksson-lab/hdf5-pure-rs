//! Extensible array data block — mirrors libhdf5's `H5EAdblock.c` plus
//! the data-block half of `H5EAcache.c`. The page handling that lives
//! in libhdf5's `H5EAdblkpage.c` is folded in here because the Rust port
//! doesn't model pages as a separate cache entry.

#![allow(dead_code)]

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

pub(super) fn dblock_debug(prefix: &ExtArrayDataBlockPrefix) -> String {
    format!(
        "ExtArrayDataBlockPrefix(pages={}, prefix_size={})",
        prefix.pages, prefix.prefix_size
    )
}

pub(super) fn cache_dblock_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "extensible array data block")
}

pub(super) fn cache_dblock_image_len(
    prefix: &ExtArrayDataBlockPrefix,
    payload_len: usize,
) -> Result<usize> {
    prefix
        .prefix_size
        .checked_add(payload_len)
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array data block image length overflow".into())
        })
}

pub(super) fn cache_dblock_serialize(prefix: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(prefix.len() + payload.len() + 4);
    out.extend_from_slice(prefix);
    out.extend_from_slice(payload);
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

pub(super) fn cache_dblock_notify(_prefix: &ExtArrayDataBlockPrefix) {}

pub(super) fn cache_dblock_free_icr(_prefix: ExtArrayDataBlockPrefix) {}

pub(super) fn cache_dblock_fsf_size(prefix: &ExtArrayDataBlockPrefix) -> usize {
    prefix.prefix_size
}

pub(super) fn cache_dblk_page_get_initial_load_size() -> usize {
    4
}

pub(super) fn cache_dblk_page_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "extensible array data block page")
}

pub(super) fn cache_dblk_page_image_len(payload_len: usize) -> Result<usize> {
    payload_len.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat("extensible array data block page image length overflow".into())
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

pub(super) fn dblk_page_create(payload: Vec<u8>) -> Vec<u8> {
    payload
}

pub(super) fn dblk_page_protect(payload: &[u8]) -> &[u8] {
    payload
}

pub(super) fn dblk_page_unprotect(_payload: &[u8]) {}

pub(super) fn dblk_page_dest(_payload: Vec<u8>) {}

pub(super) fn dblock_alloc(pages: usize, prefix_size: usize) -> ExtArrayDataBlockPrefix {
    ExtArrayDataBlockPrefix { pages, prefix_size }
}

pub(super) fn dblock_sblk_idx(
    header: &ExtensibleArrayHeader,
    data_block_elements: usize,
) -> Option<usize> {
    header
        .super_block_info
        .iter()
        .position(|info| info.data_block_elements == data_block_elements)
}

pub(super) fn dblock_protect<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    data_block_addr: u64,
    data_block_elements: usize,
) -> Result<ExtArrayDataBlockPrefix> {
    decode_data_block_prefix(
        reader,
        header_addr,
        header,
        data_block_addr,
        data_block_elements,
    )
}

pub(super) fn dblock_unprotect(_prefix: ExtArrayDataBlockPrefix) {}

pub(super) fn dblock_delete(prefix: &mut ExtArrayDataBlockPrefix) {
    prefix.pages = 0;
    prefix.prefix_size = 0;
}

pub(super) fn dblock_dest(_prefix: ExtArrayDataBlockPrefix) {}

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
    let prefix_size = super::checked_usize_add(
        4 + 1 + 1,
        reader.sizeof_addr() as usize,
        "extensible array data block prefix size",
    )
    .and_then(|value| {
        super::checked_usize_add(
            value,
            header.array_offset_size as usize,
            "extensible array data block prefix size",
        )
    })
    .and_then(|value| {
        super::checked_usize_add(value, 4, "extensible array data block prefix size")
    })?;
    Ok(ExtArrayDataBlockPrefix { pages, prefix_size })
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

    let prefix = decode_data_block_prefix(
        reader,
        header_addr,
        header,
        data_block_addr,
        data_block_elements,
    )?;
    if prefix.pages == 0 {
        for _ in 0..count {
            elements.push(read_element(reader, filtered, chunk_size_len)?);
        }
        let unread = data_block_elements.checked_sub(count).ok_or_else(|| {
            Error::InvalidFormat(
                "extensible array data block read count exceeds data block elements".into(),
            )
        })?;
        if unread > 0 {
            let skip_bytes = super::checked_usize_mul(
                unread,
                header.raw_element_size,
                "extensible array unread data block span",
            )?;
            reader.skip(super::u64_from_usize(
                skip_bytes,
                "extensible array unread data block span",
            )?)?;
        }
        let _checksum = reader.read_u32()?;
    } else {
        let page_payload = super::checked_usize_mul(
            header.data_block_page_elements,
            header.raw_element_size,
            "extensible array data block page size",
        )?;
        let page_size =
            super::checked_usize_add(page_payload, 4, "extensible array data block page size")?;
        let mut remaining = count;
        for page_index in 0..prefix.pages {
            if remaining == 0 {
                break;
            }
            let page_elements = header.data_block_page_elements.min(remaining);
            let page_offset = super::checked_usize_mul(
                page_index,
                page_size,
                "extensible array data block page offset",
            )?;
            let page_addr = super::checked_u64_add(
                data_block_addr,
                super::u64_from_usize(
                    prefix.prefix_size,
                    "extensible array data block prefix size",
                )?,
                "extensible array data block page address",
            )
            .and_then(|value| {
                super::checked_u64_add(
                    value,
                    super::u64_from_usize(page_offset, "extensible array data block page offset")?,
                    "extensible array data block page address",
                )
            })?;
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
