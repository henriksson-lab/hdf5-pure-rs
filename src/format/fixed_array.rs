use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

#[derive(Debug, Clone)]
pub struct FixedArrayElement {
    pub addr: u64,
    pub nbytes: Option<u64>,
    pub filter_mask: u32,
}

#[derive(Debug, Clone)]
struct FixedArrayHeader {
    class_id: u8,
    raw_element_size: usize,
    max_page_elements_bits: u8,
    elements: u64,
    data_block_addr: u64,
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

    read_data_block(reader, addr, &header, filtered, chunk_size_len)
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
    if element_index >= header.elements as usize {
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

    if (header.elements as usize) > page_elements {
        reader.seek(header.data_block_addr + data_prefix_size as u64)?;
        let pages = (header.elements as usize).div_ceil(page_elements);
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

fn read_header<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<FixedArrayHeader> {
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
    let data_block_addr = reader.read_addr()?;
    let _checksum = reader.read_u32()?;

    Ok(FixedArrayHeader {
        class_id,
        raw_element_size,
        max_page_elements_bits,
        elements,
        data_block_addr,
    })
}

fn read_data_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &FixedArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
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

    let mut elements = Vec::with_capacity(header.elements as usize);
    if (header.elements as usize) > page_elements {
        let pages = (header.elements as usize).div_ceil(page_elements);
        let page_init_size = pages.div_ceil(8);
        let page_init = reader.read_bytes(page_init_size)?;
        let _checksum = reader.read_u32()?;
        let prefix_size = 4 + 1 + 1 + reader.sizeof_addr() as usize + page_init_size + 4;
        let page_size = page_elements * header.raw_element_size + 4;
        for page_index in 0..pages {
            let page_start = page_index * page_elements;
            let page_count = page_elements.min(header.elements as usize - page_start);
            if bit_is_set(&page_init, page_index) {
                let page_addr =
                    header.data_block_addr + prefix_size as u64 + (page_index * page_size) as u64;
                reader.seek(page_addr)?;
                for _ in 0..page_count {
                    elements.push(read_element(reader, filtered, chunk_size_len)?);
                }
                let _page_checksum = reader.read_u32()?;
            } else {
                append_fill_elements(page_count, &mut elements);
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

fn read_element<R: Read + Seek>(
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

fn append_fill_elements(count: usize, elements: &mut Vec<FixedArrayElement>) {
    for _ in 0..count {
        elements.push(FixedArrayElement {
            addr: UNDEF_ADDR,
            nbytes: None,
            filter_mask: 0,
        });
    }
}

fn bit_is_set(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .map(|byte| (byte & (0x80 >> (bit % 8))) != 0)
        .unwrap_or(false)
}
