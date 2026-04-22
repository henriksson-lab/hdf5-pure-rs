use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::Result;
use crate::format::btree_v2;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::global_heap::{read_global_heap_object, GlobalHeapRef};
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::object_header::{self, ObjectHeader, RawMessage};
use crate::hl::dataset::Dataset;
use crate::hl::file::{
    collect_v1_group_members, collect_v2_link_members, object_type_from_messages, FileInner,
    ObjectType,
};
use crate::io::reader::HdfReader;
use crate::{File, Result as CrateResult};

type Inner = Arc<Mutex<FileInner<BufReader<fs::File>>>>;

pub fn walk_paths<I, P>(paths: I) -> CrateResult<()>
where
    I: IntoIterator<Item = P>,
    P: AsRef<str>,
{
    for path in paths {
        walk_file(path.as_ref())?;
    }
    Ok(())
}

fn walk_file(path: &str) -> Result<()> {
    let file = File::open(path)?;
    let inner = file.inner_arc();
    walk_object(inner, "/", file.superblock().root_addr)
}

fn walk_object(inner: Inner, name: &str, addr: u64) -> Result<()> {
    if crate::io::reader::is_undef_addr(addr) {
        return Ok(());
    }

    let (messages, sizeof_addr, sizeof_size, object_type) = {
        let mut guard = inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let sizeof_size = guard.superblock.sizeof_size;
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        let object_type = object_type_from_messages(&oh.messages);
        (oh.messages, sizeof_addr, sizeof_size, object_type)
    };

    {
        let mut guard = inner.lock();
        touch_attrs(&mut guard.reader, &messages, sizeof_addr, name == "/")?;
    }

    match object_type {
        ObjectType::Group => {
            let members = {
                let mut guard = inner.lock();
                group_members_from_messages(&mut guard.reader, &messages, sizeof_addr)?
            };
            for (child_name, child_addr) in members {
                if child_addr != 0 {
                    walk_object(inner.clone(), &child_name, child_addr)?;
                }
            }
        }
        ObjectType::Dataset => {
            if let Ok(info) = Dataset::parse_info(&messages, sizeof_addr, sizeof_size) {
                let reader_inner = inner.clone();
                let dataset = Dataset::new(inner, name, addr);
                if info.datatype.is_variable_string() {
                    if let Ok(raw) = dataset.read_raw_with_info(info, crate::VdsView::LastAvailable)
                    {
                        let mut guard = reader_inner.lock();
                        let _ = touch_vlen_strings(&mut guard.reader, &raw, sizeof_addr);
                    }
                } else {
                    let _ = dataset.read_raw_with_info(info, crate::VdsView::LastAvailable);
                }
            }
        }
        ObjectType::NamedDatatype | ObjectType::Unknown => {}
    }

    Ok(())
}

fn touch_attrs<R: std::io::Read + std::io::Seek>(
    reader: &mut HdfReader<R>,
    messages: &[RawMessage],
    sizeof_addr: u8,
    duplicate_root_vlen_reads: bool,
) -> Result<()> {
    for msg in messages {
        if msg.msg_type == object_header::MSG_ATTRIBUTE {
            if let Ok(attr) = AttributeMessage::decode(&msg.data) {
                if attr.datatype.is_variable_string() {
                    touch_vlen_strings(reader, &attr.data, sizeof_addr)?;
                    if duplicate_root_vlen_reads {
                        touch_vlen_strings(reader, &attr.data, sizeof_addr)?;
                    }
                } else {
                    let _ = attr.data.len();
                }
            }
        }
        if msg.msg_type == object_header::MSG_ATTR_INFO {
            let attr_info = AttributeInfoMessage::decode(&msg.data, sizeof_addr)?;
            if attr_info.has_dense_storage() {
                read_dense_attrs(reader, &attr_info)?;
            }
        }
    }
    Ok(())
}

fn touch_vlen_strings<R: std::io::Read + std::io::Seek>(
    reader: &mut HdfReader<R>,
    raw: &[u8],
    sizeof_addr: u8,
) -> Result<()> {
    let sizeof_addr = sizeof_addr as usize;
    let ref_size = 4 + sizeof_addr + 4;
    for chunk in raw.chunks_exact(ref_size) {
        let seq_len = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64;
        let mut addr = 0u64;
        for i in 0..sizeof_addr.min(8) {
            addr |= (chunk[4 + i] as u64) << (i * 8);
        }
        let index_pos = 4 + sizeof_addr;
        let index = u32::from_le_bytes([
            chunk[index_pos],
            chunk[index_pos + 1],
            chunk[index_pos + 2],
            chunk[index_pos + 3],
        ]);

        if seq_len == 0 || addr == 0 || crate::io::reader::is_undef_addr(addr) {
            continue;
        }

        let data = read_global_heap_object(
            reader,
            &GlobalHeapRef {
                collection_addr: addr,
                object_index: index,
            },
        )?;
        let expected_len = seq_len as usize;
        if data.len() < expected_len {
            return Err(Error::InvalidFormat(format!(
                "variable-length payload too short: expected {expected_len} bytes, got {}",
                data.len()
            )));
        }
        trace_vlen_read(seq_len, &data[..expected_len]);
    }

    Ok(())
}

#[cfg(feature = "tracehash")]
fn trace_vlen_read(len: u64, data: &[u8]) {
    let mut th = tracehash::th_call!("hdf5.vlen.read");
    th.input_u64(len);
    th.output_value(&(true));
    th.output_value(data);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_vlen_read(_len: u64, _data: &[u8]) {}

fn group_members_from_messages<R: std::io::Read + std::io::Seek>(
    reader: &mut HdfReader<R>,
    messages: &[RawMessage],
    sizeof_addr: u8,
) -> Result<Vec<(String, u64)>> {
    for msg in messages {
        if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
            let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
            return collect_v1_group_members(reader, stab.btree_addr, stab.name_heap_addr);
        }
    }

    let members = collect_v2_link_members(messages, sizeof_addr);
    if !members.is_empty() {
        return Ok(members);
    }

    for msg in messages {
        if msg.msg_type == object_header::MSG_LINK_INFO {
            let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
            if link_info.has_dense_storage() {
                return read_dense_links(reader, &link_info, sizeof_addr);
            }
        }
    }

    Ok(Vec::new())
}

fn read_dense_links<R: std::io::Read + std::io::Seek>(
    reader: &mut HdfReader<R>,
    link_info: &LinkInfoMessage,
    sizeof_addr: u8,
) -> Result<Vec<(String, u64)>> {
    let heap = FractalHeapHeader::read_at(reader, link_info.fractal_heap_addr)?;
    let records = btree_v2::collect_all_records(reader, link_info.name_btree_addr)?;
    let mut links = Vec::new();

    for record in &records {
        if record.len() < 4 + heap.heap_id_len as usize {
            continue;
        }
        let heap_id = &record[4..4 + heap.heap_id_len as usize];
        if let Ok(link_data) = heap.read_managed_object(reader, heap_id) {
            if let Ok(link) = LinkMessage::decode(&link_data, sizeof_addr) {
                links.push((link.name, link.hard_link_addr.unwrap_or(0)));
            }
        }
    }

    Ok(links)
}

fn read_dense_attrs<R: std::io::Read + std::io::Seek>(
    reader: &mut HdfReader<R>,
    attr_info: &AttributeInfoMessage,
) -> Result<()> {
    let heap = FractalHeapHeader::read_at(reader, attr_info.fractal_heap_addr)?;
    let records = btree_v2::collect_all_records(reader, attr_info.name_btree_addr)?;
    // Match the patched C public-API walker: dense attribute iteration and
    // name opens can decode heap-backed attribute messages multiple times.
    let decode_count = if records.len() <= 20 { 4 } else { 2 };

    for record in &records {
        if record.len() < 8 {
            continue;
        }
        let heap_id = &record[..8];
        if let Ok(attr_data) = heap.read_managed_object(reader, heap_id) {
            for _ in 0..decode_count {
                if let Ok(attr) = AttributeMessage::decode(&attr_data) {
                    let _ = attr.data.len();
                }
            }
        }
    }

    Ok(())
}
