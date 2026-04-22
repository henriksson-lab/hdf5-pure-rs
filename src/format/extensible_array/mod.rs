//! Extensible array — top-level public API. Mirrors libhdf5's `H5EA.c`
//! (the file-spanning entry points). Per-component code lives in sibling
//! modules: `hdr` (header decode + checksum + super-block-info build),
//! `iblock` (index block decode + spillover descent), `sblock` (super
//! block decode + walk), `dblock` (data block decode + element walk;
//! absorbs `dblkpage` because the Rust port doesn't model pages as a
//! separate cache entry).

mod dblock;
pub(crate) mod hdr;
mod iblock;
mod sblock;

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

use super::fixed_array;
use super::fixed_array::FixedArrayElement;
use hdr::{read_header, ExtensibleArrayHeader};
use iblock::read_index_block;

pub fn read_extensible_array_chunks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let header = read_header(reader, addr)?;
    let expected_class = if filtered { 1 } else { 0 };
    if header.class_id != expected_class {
        return Err(Error::InvalidFormat(format!(
            "extensible array class {} does not match filtered={filtered}",
            header.class_id
        )));
    }

    if is_undef_addr(header.index_block_addr) {
        return Ok(Vec::new());
    }
    read_index_block(reader, addr, &header, filtered, chunk_size_len)
}

// ---------------------------------------------------------------------------
// Internal helpers shared across hdr/iblock/sblock/dblock. Mirrors libhdf5's
// `H5EAint.c` (the package-internal helper file).
// `unreachable_pub` is allowed because submodules need `pub(super)`
// visibility on these, even though the helpers operate on
// extensible-array-private types.
// ---------------------------------------------------------------------------

#[allow(private_interfaces)]
pub(super) fn append_fill_elements(
    header: &ExtensibleArrayHeader,
    count: usize,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    let remaining = usize_from_u64(header.max_index_set, "extensible array max index")?
        .saturating_sub(elements.len());
    for _ in 0..count.min(remaining) {
        elements.push(FixedArrayElement {
            addr: UNDEF_ADDR,
            nbytes: None,
            filter_mask: 0,
        });
    }
    Ok(())
}

#[allow(private_interfaces)]
pub(super) fn data_block_pages(
    header: &ExtensibleArrayHeader,
    data_block_elements: usize,
) -> usize {
    if data_block_elements > header.data_block_page_elements {
        data_block_elements / header.data_block_page_elements
    } else {
        0
    }
}

pub(super) fn bit_is_set(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .map(|byte| (byte & (0x80 >> (bit % 8))) != 0)
        .unwrap_or(false)
}

pub(super) fn log2_power2(value: u64) -> Result<usize> {
    if value == 0 || !value.is_power_of_two() {
        return Err(Error::InvalidFormat(format!(
            "extensible array value {value} is not a power of two"
        )));
    }
    Ok(value.trailing_zeros() as usize)
}

pub(super) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}
