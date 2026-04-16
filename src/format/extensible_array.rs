use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

use super::fixed_array::FixedArrayElement;

const MAX_EXTENSIBLE_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
struct ExtensibleArrayHeader {
    class_id: u8,
    raw_element_size: usize,
    index_block_elements: u8,
    max_index_set: u64,
    index_block_addr: u64,
    array_offset_size: u8,
    data_block_page_elements: usize,
    index_block_super_blocks: usize,
    index_block_data_block_addrs: usize,
    index_block_super_block_addrs: usize,
    super_block_info: Vec<SuperBlockInfo>,
}

#[derive(Debug, Clone)]
struct SuperBlockInfo {
    data_blocks: usize,
    data_block_elements: usize,
    start_data_block: u64,
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
    let max_elements_bits = reader.read_u8()?;
    let index_block_elements = reader.read_u8()?;
    let data_block_min_elements = reader.read_u8()?;
    let super_block_min_data_ptrs = reader.read_u8()?;
    let max_data_block_page_elements_bits = reader.read_u8()?;

    let _super_block_count = reader.read_length()?;
    let _super_block_size = reader.read_length()?;
    let _data_block_count = reader.read_length()?;
    let _data_block_size = reader.read_length()?;
    let max_index_set = reader.read_length()?;
    let max_index_count = usize_from_u64(max_index_set, "extensible array max index")?;
    if max_index_count > MAX_EXTENSIBLE_ARRAY_ELEMENTS {
        return Err(Error::InvalidFormat(format!(
            "extensible array max index {max_index_count} exceeds supported maximum {MAX_EXTENSIBLE_ARRAY_ELEMENTS}"
        )));
    }
    let _realized_elements = reader.read_length()?;
    let index_block_addr = reader.read_addr()?;
    verify_checksum(reader, addr, "extensible array header")?;

    let array_offset_size = max_elements_bits.div_ceil(8);
    let data_block_page_elements = 1usize
        .checked_shl(max_data_block_page_elements_bits as u32)
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array page element count overflow".into())
        })?;
    let super_block_count = 1usize
        + (max_elements_bits as usize)
            .checked_sub(log2_power2(data_block_min_elements as u64)?)
            .ok_or_else(|| {
                Error::InvalidFormat("invalid extensible array block parameters".into())
            })?;
    let index_block_super_blocks = 2 * log2_power2(super_block_min_data_ptrs as u64)?;
    let index_block_data_block_addrs = 2 * (super_block_min_data_ptrs as usize - 1);
    let index_block_super_block_addrs = super_block_count
        .checked_sub(index_block_super_blocks)
        .ok_or_else(|| {
            Error::InvalidFormat("invalid extensible array super block layout".into())
        })?;
    let super_block_info =
        build_super_block_info(super_block_count, data_block_min_elements as usize)?;

    Ok(ExtensibleArrayHeader {
        class_id,
        raw_element_size,
        index_block_elements,
        max_index_set,
        index_block_addr,
        array_offset_size,
        data_block_page_elements,
        index_block_super_blocks,
        index_block_data_block_addrs,
        index_block_super_block_addrs,
        super_block_info,
    })
}

fn verify_checksum<R: Read + Seek>(
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

    let max_index_count = usize_from_u64(header.max_index_set, "extensible array max index")?;
    let mut elements = Vec::with_capacity(max_index_count);
    for idx in 0..header.index_block_elements {
        let element = read_element(reader, filtered, chunk_size_len)?;
        if (idx as u64) < header.max_index_set {
            elements.push(element);
        }
    }

    let mut data_block_addrs = Vec::with_capacity(header.index_block_data_block_addrs);
    for _ in 0..header.index_block_data_block_addrs {
        data_block_addrs.push(reader.read_addr()?);
    }

    let mut super_block_addrs = Vec::with_capacity(header.index_block_super_block_addrs);
    for _ in 0..header.index_block_super_block_addrs {
        super_block_addrs.push(reader.read_addr()?);
    }

    let _checksum = reader.read_u32()?;
    read_spillover_blocks(
        reader,
        header_addr,
        header,
        filtered,
        chunk_size_len,
        &data_block_addrs,
        &super_block_addrs,
        &mut elements,
    )?;
    Ok(elements)
}

#[allow(clippy::too_many_arguments)]
fn read_spillover_blocks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    data_block_addrs: &[u64],
    super_block_addrs: &[u64],
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    for (super_block_index, info) in header.super_block_info.iter().enumerate() {
        if elements.len() as u64 >= header.max_index_set {
            break;
        }

        if super_block_index < header.index_block_super_blocks {
            for local_data_block in 0..info.data_blocks {
                if elements.len() as u64 >= header.max_index_set {
                    break;
                }

                let data_block_index = info.start_data_block as usize + local_data_block;
                let Some(&data_block_addr) = data_block_addrs.get(data_block_index) else {
                    return Err(Error::InvalidFormat(
                        "extensible array data block address index out of bounds".into(),
                    ));
                };
                let remaining = usize_from_u64(
                    header.max_index_set - elements.len() as u64,
                    "extensible array remaining element count",
                )?;
                let count = info.data_block_elements.min(remaining);
                append_data_block_elements(
                    reader,
                    header_addr,
                    header,
                    filtered,
                    chunk_size_len,
                    data_block_addr,
                    info.data_block_elements,
                    None,
                    count,
                    elements,
                )?;
            }
        } else {
            let super_block_addr_index = super_block_index - header.index_block_super_blocks;
            let Some(&super_block_addr) = super_block_addrs.get(super_block_addr_index) else {
                return Err(Error::InvalidFormat(
                    "extensible array super block address index out of bounds".into(),
                ));
            };
            read_super_block(
                reader,
                header_addr,
                header,
                filtered,
                chunk_size_len,
                super_block_addr,
                info,
                elements,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_super_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    super_block_addr: u64,
    info: &SuperBlockInfo,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    if is_undef_addr(super_block_addr) {
        append_fill_elements(
            header,
            info.data_blocks * info.data_block_elements,
            elements,
        )?;
        return Ok(());
    }

    reader.seek(super_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EASB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array super block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array super block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array super block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array super block owner address does not match header".into(),
        ));
    }

    let _block_offset = reader.read_uint(header.array_offset_size)?;
    let data_block_pages = data_block_pages(header, info.data_block_elements);
    let page_init_size = if data_block_pages > 0 {
        data_block_pages.div_ceil(8)
    } else {
        0
    };
    let page_init = if page_init_size > 0 {
        reader.read_bytes(info.data_blocks * page_init_size)?
    } else {
        Vec::new()
    };

    let mut data_block_addrs = Vec::with_capacity(info.data_blocks);
    for _ in 0..info.data_blocks {
        data_block_addrs.push(reader.read_addr()?);
    }

    let _checksum = reader.read_u32()?;

    for (data_block_index, &data_block_addr) in data_block_addrs.iter().enumerate() {
        if elements.len() as u64 >= header.max_index_set {
            break;
        }
        let remaining = usize_from_u64(
            header.max_index_set - elements.len() as u64,
            "extensible array remaining element count",
        )?;
        let count = info.data_block_elements.min(remaining);
        let page_init_for_block = if page_init_size > 0 {
            Some(
                &page_init
                    [data_block_index * page_init_size..(data_block_index + 1) * page_init_size],
            )
        } else {
            None
        };
        append_data_block_elements(
            reader,
            header_addr,
            header,
            filtered,
            chunk_size_len,
            data_block_addr,
            info.data_block_elements,
            page_init_for_block,
            count,
            elements,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_data_block_elements<R: Read + Seek>(
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
        append_fill_elements(header, count, elements)?;
        return Ok(());
    }

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
    let pages = data_block_pages(header, data_block_elements);
    if pages == 0 {
        for _ in 0..count {
            elements.push(read_element(reader, filtered, chunk_size_len)?);
        }
        let unread = data_block_elements.saturating_sub(count);
        if unread > 0 {
            reader.skip((unread * header.raw_element_size) as u64)?;
        }
        let _checksum = reader.read_u32()?;
    } else {
        let prefix_size =
            4 + 1 + 1 + reader.sizeof_addr() as usize + header.array_offset_size as usize + 4;
        let page_size = header.data_block_page_elements * header.raw_element_size + 4;
        let mut remaining = count;
        for page_index in 0..pages {
            if remaining == 0 {
                break;
            }
            let page_elements = header.data_block_page_elements.min(remaining);
            let page_addr = data_block_addr + prefix_size as u64 + (page_index * page_size) as u64;
            let page_initialized = page_init
                .map(|bits| bit_is_set(bits, page_index))
                .unwrap_or(true);
            if page_initialized {
                reader.seek(page_addr)?;
                for _ in 0..page_elements {
                    elements.push(read_element(reader, filtered, chunk_size_len)?);
                }
            } else {
                append_fill_elements(header, page_elements, elements)?;
            }
            remaining -= page_elements;
        }
    }

    Ok(())
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

fn append_fill_elements(
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

fn build_super_block_info(
    count: usize,
    min_data_block_elements: usize,
) -> Result<Vec<SuperBlockInfo>> {
    let mut infos = Vec::with_capacity(count);
    let mut start_index = 0u64;
    let mut start_data_block = 0u64;
    for index in 0..count {
        let data_blocks = 1usize.checked_shl((index / 2) as u32).ok_or_else(|| {
            Error::InvalidFormat("extensible array data block count overflow".into())
        })?;
        let data_block_elements = min_data_block_elements
            .checked_mul(
                1usize
                    .checked_shl(index.div_ceil(2) as u32)
                    .ok_or_else(|| {
                        Error::InvalidFormat(
                            "extensible array data block element count overflow".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block size overflow".into())
            })?;
        infos.push(SuperBlockInfo {
            data_blocks,
            data_block_elements,
            start_data_block,
        });
        start_index = start_index
            .checked_add((data_blocks as u64) * (data_block_elements as u64))
            .ok_or_else(|| Error::InvalidFormat("extensible array start index overflow".into()))?;
        start_data_block = start_data_block
            .checked_add(data_blocks as u64)
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block index overflow".into())
            })?;
    }
    Ok(infos)
}

fn data_block_pages(header: &ExtensibleArrayHeader, data_block_elements: usize) -> usize {
    if data_block_elements > header.data_block_page_elements {
        data_block_elements / header.data_block_page_elements
    } else {
        0
    }
}

fn bit_is_set(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .map(|byte| (byte & (0x80 >> (bit % 8))) != 0)
        .unwrap_or(false)
}

fn log2_power2(value: u64) -> Result<usize> {
    if value == 0 || !value.is_power_of_two() {
        return Err(Error::InvalidFormat(format!(
            "extensible array value {value} is not a power of two"
        )));
    }
    Ok(value.trailing_zeros() as usize)
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}
