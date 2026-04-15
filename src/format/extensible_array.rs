use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader};

use super::fixed_array::FixedArrayElement;

#[derive(Debug, Clone)]
struct ExtensibleArrayHeader {
    class_id: u8,
    raw_element_size: usize,
    index_block_elements: u8,
    max_index_set: u64,
    index_block_addr: u64,
}

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
    if header.max_index_set > header.index_block_elements as u64 {
        return Err(Error::Unsupported(
            "extensible array data/super blocks are not implemented".into(),
        ));
    }

    read_index_block(reader, addr, &header, filtered, chunk_size_len)
}

fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ExtensibleArrayHeader> {
    reader.seek(addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EAHD" {
        return Err(Error::InvalidFormat(
            "invalid extensible array header magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array header version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    let raw_element_size = reader.read_u8()? as usize;
    let _max_elements_bits = reader.read_u8()?;
    let index_block_elements = reader.read_u8()?;
    let _data_block_min_elements = reader.read_u8()?;
    let _super_block_min_data_ptrs = reader.read_u8()?;
    let _max_page_elements_bits = reader.read_u8()?;

    let _super_block_count = reader.read_length()?;
    let _super_block_size = reader.read_length()?;
    let _data_block_count = reader.read_length()?;
    let _data_block_size = reader.read_length()?;
    let max_index_set = reader.read_length()?;
    let _realized_elements = reader.read_length()?;
    let index_block_addr = reader.read_addr()?;
    let _checksum = reader.read_u32()?;

    Ok(ExtensibleArrayHeader {
        class_id,
        raw_element_size,
        index_block_elements,
        max_index_set,
        index_block_addr,
    })
}

fn read_index_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    reader.seek(header.index_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EAIB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array index block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array index block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array index block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array index block owner address does not match header".into(),
        ));
    }

    let expected_element_size = if filtered {
        reader.sizeof_addr() as usize + chunk_size_len + 4
    } else {
        reader.sizeof_addr() as usize
    };
    if header.raw_element_size != expected_element_size {
        return Err(Error::InvalidFormat(format!(
            "extensible array raw element size {} does not match expected {}",
            header.raw_element_size, expected_element_size
        )));
    }

    let mut elements = Vec::with_capacity(header.max_index_set as usize);
    for idx in 0..header.index_block_elements {
        let addr = reader.read_addr()?;
        if filtered {
            let nbytes = reader.read_uint(chunk_size_len as u8)?;
            let filter_mask = reader.read_u32()?;
            if (idx as u64) < header.max_index_set {
                elements.push(FixedArrayElement {
                    addr,
                    nbytes: Some(nbytes),
                    filter_mask,
                });
            }
        } else if (idx as u64) < header.max_index_set {
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
