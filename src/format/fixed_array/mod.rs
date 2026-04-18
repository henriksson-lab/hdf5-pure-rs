//! Fixed array — top-level public API. Mirrors libhdf5's `H5FA.c` (the
//! file-spanning entry points). Per-component code lives in sibling
//! modules: `hdr` (header decode + checksum), `dblock` (data-block
//! decode + element iteration; absorbs `dblkpage` because the Rust port
//! doesn't model pages as a separate cache entry).

mod dblock;
mod hdr;

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

use hdr::read_header;

#[derive(Debug, Clone)]
pub struct FixedArrayElement {
    pub addr: u64,
    pub nbytes: Option<u64>,
    pub filter_mask: u32,
}

pub fn read_fixed_array_chunks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let header = read_header(reader, addr)?;
    let expected_class = if filtered { 1 } else { 0 };
    if header.class_id != expected_class {
        return Err(Error::InvalidFormat(format!(
            "fixed array class {} does not match filtered={filtered}",
            header.class_id
        )));
    }

    if is_undef_addr(header.data_block_addr) {
        return Ok(Vec::new());
    }

    dblock::read_data_block(reader, addr, &header, filtered, chunk_size_len)
}

/// Locate the file offset of an existing fixed-array element.
///
/// The returned offset points at the element address field. For filtered chunk
/// arrays, the filtered-size and filter-mask fields follow immediately.
pub fn locate_fixed_array_element<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
    element_index: usize,
) -> Result<u64> {
    let header = read_header(reader, addr)?;
    let expected_class = if filtered { 1 } else { 0 };
    if header.class_id != expected_class {
        return Err(Error::InvalidFormat(format!(
            "fixed array class {} does not match filtered={filtered}",
            header.class_id
        )));
    }
    let element_count = usize_from_u64(header.elements, "fixed array element count")?;
    if element_index >= element_count {
        return Err(Error::InvalidFormat(format!(
            "fixed array element index {element_index} out of bounds for {} elements",
            header.elements
        )));
    }
    if is_undef_addr(header.data_block_addr) {
        return Err(Error::Unsupported(
            "cannot update fixed-array chunk entry without a data block".into(),
        ));
    }

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

    let page_elements = 1usize
        .checked_shl(header.max_page_elements_bits as u32)
        .ok_or_else(|| Error::InvalidFormat("fixed array page size overflow".into()))?;
    let data_prefix_size = 4 + 1 + 1 + reader.sizeof_addr() as usize;

    if element_count > page_elements {
        reader.seek(header.data_block_addr + data_prefix_size as u64)?;
        let pages = element_count.div_ceil(page_elements);
        let page_init_size = pages.div_ceil(8);
        let page_init = reader.read_bytes(page_init_size)?;
        let page_index = element_index / page_elements;
        if !bit_is_set(&page_init, page_index) {
            return Err(Error::Unsupported(
                "cannot update uninitialized fixed-array chunk page".into(),
            ));
        }
        let prefix_size = data_prefix_size + page_init_size + 4;
        let page_size = page_elements * header.raw_element_size + 4;
        let within_page = element_index % page_elements;
        Ok(header.data_block_addr
            + prefix_size as u64
            + (page_index * page_size) as u64
            + (within_page * header.raw_element_size) as u64)
    } else {
        Ok(header.data_block_addr
            + data_prefix_size as u64
            + (element_index * header.raw_element_size) as u64)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — shared across `hdr` and `dblock`. Mirrors libhdf5's
// `H5FAint.c` (the package-internal helper file).
// ---------------------------------------------------------------------------

pub(super) fn append_fill_elements(count: usize, elements: &mut Vec<FixedArrayElement>) {
    for _ in 0..count {
        elements.push(FixedArrayElement {
            addr: UNDEF_ADDR,
            nbytes: None,
            filter_mask: 0,
        });
    }
}

pub(super) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

pub(super) fn bit_is_set(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .map(|byte| (byte & (0x80 >> (bit % 8))) != 0)
        .unwrap_or(false)
}
