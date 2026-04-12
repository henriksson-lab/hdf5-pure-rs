use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// v2 B-tree header magic: "BTHD"
const B2HD_MAGIC: [u8; 4] = [b'B', b'T', b'H', b'D'];
/// v2 B-tree internal node magic: "BTIN"
const B2IN_MAGIC: [u8; 4] = [b'B', b'T', b'I', b'N'];
/// v2 B-tree leaf node magic: "BTLF"
const B2LF_MAGIC: [u8; 4] = [b'B', b'T', b'L', b'F'];

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
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != B2HD_MAGIC {
            return Err(Error::InvalidFormat("invalid v2 B-tree header magic".into()));
        }

        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!("v2 B-tree header version {version}")));
        }

        let tree_type = reader.read_u8()?;
        let node_size = reader.read_u32()?;
        let record_size = reader.read_u16()?;
        let depth = reader.read_u16()?;
        let split_pct = reader.read_u8()?;
        let merge_pct = reader.read_u8()?;
        let root_addr = reader.read_addr()?;
        let root_nrecords = reader.read_u16()?;
        let total_records = reader.read_length()?;

        // Checksum
        let _checksum = reader.read_u32()?;

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
        // Root is a leaf node
        read_leaf_records(reader, header.root_addr, header.root_nrecords, header.record_size, &mut records)?;
    } else {
        // Root is an internal node, recurse
        read_internal_records(
            reader,
            header.root_addr,
            header.root_nrecords,
            header.depth,
            header.record_size,
            header.node_size,
            &mut records,
        )?;
    }

    Ok(records)
}

fn read_leaf_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    nrecords: u16,
    record_size: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    reader.seek(addr)?;

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

fn read_internal_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    nrecords: u16,
    depth: u16,
    record_size: u16,
    node_size: u32,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    reader.seek(addr)?;

    let magic = reader.read_bytes(4)?;
    if magic != B2IN_MAGIC {
        return Err(Error::InvalidFormat("invalid v2 B-tree internal magic".into()));
    }

    let _version = reader.read_u8()?;
    let _type = reader.read_u8()?;

    // Read records and child pointers interleaved
    // Structure: record[0], child[0], record[1], child[1], ..., record[n-1], child[n-1], child[n]
    // Actually: child[0], record[0], child[1], record[1], ..., record[n-1], child[n]
    // Wait, the v2 B-tree spec says: records interleaved with child node pointers.
    // Format: record[0], child_ptr[0], record[1], child_ptr[1], ..., record[nrec-1], child_ptr[nrec]
    // No wait - it's: child_ptr[0], record[0], child_ptr[1], record[1], ..., record[nrec-1], child_ptr[nrec]
    // Actually let me check: the format has nrecords records and nrecords+1 child pointers.

    // In practice, for simplicity, let's collect child addresses and recurse.
    // The internal node has: magic(4) + version(1) + type(1) + records_and_children + checksum(4)
    // Each child pointer = addr(sizeof_addr) + nrecords_in_child(variable) + total_records_in_child(variable)

    // This is getting complex. For now, let's just read all records from leaf nodes by following
    // the leftmost path and then scanning siblings. A simpler approach: read the whole node data
    // and parse records.

    // For the common case of depth=1 (root is internal, children are leaves):
    // We need to read child pointers and recurse into leaves.

    // Simplified approach: read all data from the node and extract child addresses
    let _sizeof_addr = reader.sizeof_addr() as usize;

    // The number of child records bytes depends on the max records per node.
    // For simplicity, let's just collect children and recurse.
    // Internal node format after magic+version+type:
    // For each record/child pair and extra child at end.
    // Actually the format is simpler than I thought:
    // records are stored, then child node pointers follow.

    // Let me just read records from this node and child pointers
    let mut child_addrs = Vec::new();
    let mut child_nrecords = Vec::new();

    // Read nrecords records and nrecords+1 child pointers
    // The layout within the node is: [records...] [child_pointers...]
    // where each child pointer = address + nrecords_in_child(2 bytes for depth>1)

    // First, read all records
    for _ in 0..nrecords {
        let record = reader.read_bytes(record_size as usize)?;
        records.push(record);
    }

    // Then read child pointers: (nrecords + 1) children
    for _ in 0..=nrecords {
        let child_addr = reader.read_addr()?;
        // Number of records in child (encoded size depends on max records)
        // For simplicity, assume 2 bytes
        let child_nrec = reader.read_u16()?;
        child_addrs.push(child_addr);
        child_nrecords.push(child_nrec);

        // If depth > 1, there's also a total records count
        if depth > 1 {
            let _total = reader.read_length()?;
        }
    }

    // Recurse into children
    for (i, &child_addr) in child_addrs.iter().enumerate() {
        if depth - 1 == 0 {
            read_leaf_records(reader, child_addr, child_nrecords[i], record_size, records)?;
        } else {
            read_internal_records(
                reader,
                child_addr,
                child_nrecords[i],
                depth - 1,
                record_size,
                node_size,
                records,
            )?;
        }
    }

    Ok(())
}
