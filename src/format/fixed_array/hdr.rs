//! Fixed array header — mirrors libhdf5's `H5FAhdr.c` + the header-half
//! of `H5FAcache.c` (`H5FA__cache_hdr_deserialize`).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

const MAX_FIXED_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
pub(super) struct FixedArrayHeader {
    pub(super) class_id: u8,
    pub(super) raw_element_size: usize,
    pub(super) max_page_elements_bits: u8,
    pub(super) elements: u64,
    pub(super) data_block_addr: u64,
}

pub(super) fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<FixedArrayHeader> {
    reader.seek(addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"FAHD" {
        return Err(Error::InvalidFormat(
            "invalid fixed array header magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "fixed array header version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    let raw_element_size = reader.read_u8()? as usize;
    let max_page_elements_bits = reader.read_u8()?;
    let elements = reader.read_length()?;
    let element_count = super::usize_from_u64(elements, "fixed array element count")?;
    if element_count > MAX_FIXED_ARRAY_ELEMENTS {
        return Err(Error::InvalidFormat(format!(
            "fixed array element count {element_count} exceeds supported maximum {MAX_FIXED_ARRAY_ELEMENTS}"
        )));
    }
    let data_block_addr = reader.read_addr()?;
    verify_checksum(reader, addr, "fixed array header")?;

    Ok(FixedArrayHeader {
        class_id,
        raw_element_size,
        max_page_elements_bits,
        elements,
        data_block_addr,
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
