use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader};

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

    let page_elements = 1u64
        .checked_shl(header.max_page_elements_bits as u32)
        .ok_or_else(|| Error::InvalidFormat("fixed array page size overflow".into()))?;
    if header.elements > page_elements {
        return Err(Error::Unsupported(
            "paged fixed array data blocks are not implemented".into(),
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

    let mut elements = Vec::with_capacity(header.elements as usize);
    for _ in 0..header.elements {
        let addr = reader.read_addr()?;
        if filtered {
            let nbytes = reader.read_uint(chunk_size_len as u8)?;
            let filter_mask = reader.read_u32()?;
            elements.push(FixedArrayElement {
                addr,
                nbytes: Some(nbytes),
                filter_mask,
            });
        } else {
            elements.push(FixedArrayElement {
                addr,
                nbytes: None,
                filter_mask: 0,
            });
        }
    }

    let _checksum = reader.read_u32()?;
    Ok(elements)
}
