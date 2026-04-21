use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

/// v2 B-tree header magic: "BTHD"
const B2HD_MAGIC: [u8; 4] = [b'B', b'T', b'H', b'D'];
/// v2 B-tree leaf node magic: "BTLF"
const B2LF_MAGIC: [u8; 4] = [b'B', b'T', b'L', b'F'];
/// v2 B-tree internal node magic: "BTIN"
const B2IN_MAGIC: [u8; 4] = [b'B', b'T', b'I', b'N'];
const B2_METADATA_PREFIX_SIZE: usize = 10;

/// v2 B-tree header.
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    pub tree_type: u8,
    pub node_size: u32,
    pub record_size: u16,
    pub depth: u16,
    pub split_pct: u8,
    pub merge_pct: u8,
    pub root_addr: u64,
    pub root_nrecords: u16,
    pub total_records: u64,
}

impl BTreeV2Header {
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr).map_err(|err| {
            Error::InvalidFormat(format!("failed to seek to v2 B-tree header {addr}: {err}"))
        })?;

        let magic = reader.read_bytes(4)?;
        if magic != B2HD_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid v2 B-tree header magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "v2 B-tree header version {version}"
            )));
        }

        let tree_type = reader.read_u8()?;
        let node_size = reader.read_u32()?;
        let record_size = reader.read_u16()?;
        let depth = reader.read_u16()?;
        let split_pct = reader.read_u8()?;
        let merge_pct = reader.read_u8()?;
        if split_pct == 0 || split_pct > 100 {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree split percent {split_pct} must be in 1..=100"
            )));
        }
        if merge_pct == 0 || merge_pct > 100 {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree merge percent {merge_pct} must be in 1..=100"
            )));
        }
        let root_addr = reader.read_addr()?;
        let root_nrecords = reader.read_u16()?;
        let total_records = reader.read_length()?;

        verify_checksum(reader, addr, "v2 B-tree header")?;

        Ok(Self {
            tree_type,
            node_size,
            record_size,
            depth,
            split_pct,
            merge_pct,
            root_addr,
            root_nrecords,
            total_records,
        })
    }
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

/// Collect all records from a v2 B-tree as raw byte arrays.
pub fn collect_all_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
) -> Result<Vec<Vec<u8>>> {
    let header = BTreeV2Header::read_at(reader, header_addr)?;

    if header.total_records == 0 {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    if header.depth == 0 {
        read_leaf_records(
            reader,
            header.root_addr,
            header.root_nrecords,
            header.record_size,
            &mut records,
        )?;
    } else {
        let node_info = compute_node_info(&header, reader.sizeof_addr() as usize)?;
        read_internal_records(
            reader,
            &header,
            &node_info,
            header.root_addr,
            header.root_nrecords,
            header.depth,
            &mut records,
        )?;
    }

    Ok(records)
}

#[derive(Debug, Clone)]
struct NodeInfo {
    max_nrec: usize,
    cum_max_nrec: u64,
    cum_max_nrec_size: usize,
}

fn compute_node_info(header: &BTreeV2Header, sizeof_addr: usize) -> Result<Vec<NodeInfo>> {
    let node_size = header.node_size as usize;
    let record_size = header.record_size as usize;
    if node_size <= B2_METADATA_PREFIX_SIZE || record_size == 0 {
        return Err(Error::InvalidFormat("invalid v2 B-tree node sizing".into()));
    }

    let leaf_max = (node_size - B2_METADATA_PREFIX_SIZE) / record_size;
    if leaf_max == 0 {
        return Err(Error::InvalidFormat(
            "v2 B-tree leaf cannot hold any records".into(),
        ));
    }

    let max_nrec_size = bytes_needed(leaf_max as u64);
    let mut node_info = Vec::with_capacity(header.depth as usize + 1);
    node_info.push(NodeInfo {
        max_nrec: leaf_max,
        cum_max_nrec: leaf_max as u64,
        cum_max_nrec_size: 0,
    });

    for depth in 1..=header.depth as usize {
        let pointer_size = sizeof_addr + max_nrec_size + node_info[depth - 1].cum_max_nrec_size;
        if node_size <= B2_METADATA_PREFIX_SIZE + pointer_size {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal node cannot hold records".into(),
            ));
        }

        let max_nrec =
            (node_size - (B2_METADATA_PREFIX_SIZE + pointer_size)) / (record_size + pointer_size);
        if max_nrec == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal node cannot hold records".into(),
            ));
        }

        let prev_cum = node_info[depth - 1].cum_max_nrec;
        let cum_max_nrec = ((max_nrec as u64 + 1) * prev_cum) + max_nrec as u64;
        node_info.push(NodeInfo {
            max_nrec,
            cum_max_nrec,
            cum_max_nrec_size: bytes_needed(cum_max_nrec),
        });
    }

    Ok(node_info)
}

/// Decoded v2 B-tree internal node — magic, version, the record array,
/// and the child-pointer table. Mirrors `H5B2__cache_int_deserialize`:
/// pure parsing, no traversal of children.
#[derive(Debug, Clone)]
pub struct BTreeV2InternalNode {
    pub records: Vec<Vec<u8>>,
    /// One entry per child pointer: `(child_addr, child_nrecords)`.
    /// `records.len() + 1` entries total.
    pub children: Vec<(u64, u16)>,
}

/// Pure deserializer for a v2 B-tree internal node — mirrors libhdf5's
/// `H5B2__cache_int_deserialize`.
fn decode_internal_node<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
) -> Result<BTreeV2InternalNode> {
    reader.seek(addr).map_err(|err| {
        Error::InvalidFormat(format!(
            "failed to seek to v2 B-tree internal node {addr}: {err}"
        ))
    })?;

    let magic = reader.read_bytes(4)?;
    if magic != B2IN_MAGIC {
        return Err(Error::InvalidFormat(
            "invalid v2 B-tree internal magic".into(),
        ));
    }

    let _version = reader.read_u8()?;
    let _type = reader.read_u8()?;

    let mut records = Vec::with_capacity(nrecords as usize);
    for _ in 0..nrecords {
        records.push(reader.read_bytes(header.record_size as usize)?);
    }

    let max_nrec_size = bytes_needed(node_info[0].max_nrec as u64);
    let child_all_nrec_size = if depth > 1 {
        node_info[depth as usize - 1].cum_max_nrec_size
    } else {
        0
    };

    let mut children = Vec::with_capacity(nrecords as usize + 1);
    for _ in 0..=nrecords {
        let child_addr = reader.read_addr()?;
        let child_nrecords = read_var_uint(reader, max_nrec_size)? as u16;
        let child_all_records = if child_all_nrec_size > 0 {
            read_var_uint(reader, child_all_nrec_size)?
        } else {
            child_nrecords as u64
        };
        if matches!(header.tree_type, 10 | 11) {
            trace_internal_child(
                depth,
                children.len(),
                child_addr,
                child_nrecords,
                child_all_records,
            );
        }
        children.push((child_addr, child_nrecords));
    }

    Ok(BTreeV2InternalNode { records, children })
}

/// Drive the decoded internal-node into the depth-first record stream —
/// mirrors libhdf5's `H5B2_iterate` for an internal node.
fn read_internal_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let node = decode_internal_node(reader, header, node_info, addr, nrecords, depth)?;

    for idx in 0..node.records.len() {
        read_child_records(
            reader,
            header,
            node_info,
            node.children[idx],
            depth - 1,
            records,
        )?;
        records.push(node.records[idx].clone());
    }
    read_child_records(
        reader,
        header,
        node_info,
        node.children[node.records.len()],
        depth - 1,
        records,
    )?;

    Ok(())
}

#[cfg(feature = "tracehash")]
fn trace_internal_child(
    depth: u16,
    child_index: usize,
    child_addr: u64,
    child_nrecords: u16,
    child_all_records: u64,
) {
    let mut th = tracehash::th_call!("hdf5.chunk_index.btree2.internal_traverse");
    th.input_u64(depth as u64);
    th.input_u64(child_index as u64);
    th.input_u64(child_addr);
    th.output_value(&(true));
    th.output_u64(child_nrecords as u64);
    th.output_u64(child_all_records);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_internal_child(
    _depth: u16,
    _child_index: usize,
    _child_addr: u64,
    _child_nrecords: u16,
    _child_all_records: u64,
) {
}

fn read_child_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    child: (u64, u16),
    depth: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    if depth == 0 {
        read_leaf_records(reader, child.0, child.1, header.record_size, records)
    } else {
        read_internal_records(reader, header, node_info, child.0, child.1, depth, records)
    }
}

fn read_leaf_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    nrecords: u16,
    record_size: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    reader.seek(addr).map_err(|err| {
        Error::InvalidFormat(format!("failed to seek to v2 B-tree leaf {addr}: {err}"))
    })?;

    let magic = reader.read_bytes(4)?;
    if magic != B2LF_MAGIC {
        return Err(Error::InvalidFormat("invalid v2 B-tree leaf magic".into()));
    }

    let _version = reader.read_u8()?;
    let _type = reader.read_u8()?;

    for _ in 0..nrecords {
        let record = reader.read_bytes(record_size as usize)?;
        records.push(record);
    }

    Ok(())
}

fn read_var_uint<R: Read + Seek>(reader: &mut HdfReader<R>, size: usize) -> Result<u64> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(format!(
            "invalid v2 B-tree variable integer size {size}"
        )));
    }

    let bytes = reader.read_bytes(size)?;
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().enumerate() {
        value |= (*byte as u64) << (idx * 8);
    }
    Ok(value)
}

fn bytes_needed(mut value: u64) -> usize {
    let mut bytes = 1usize;
    while value > 0xff {
        value >>= 8;
        bytes += 1;
    }
    bytes
}
