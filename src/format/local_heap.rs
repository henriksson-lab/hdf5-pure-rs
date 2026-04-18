use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Local heap magic: "HEAP"
const HEAP_MAGIC: [u8; 4] = [b'H', b'E', b'A', b'P'];
const MAX_LOCAL_HEAP_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Parsed local-heap prefix (the on-disk header that precedes the data
/// segment). Mirrors the work `H5HL__cache_prefix_deserialize` does in
/// libhdf5: header bytes in, struct out, no data-segment I/O.
#[derive(Debug, Clone, Copy)]
pub struct LocalHeapPrefix {
    pub data_size: u64,
    pub free_list_offset: u64,
    pub data_addr: u64,
}

/// A local heap stores variable-length strings (link names) for v1 groups.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// The raw data content of the heap.
    pub data: Vec<u8>,
}

impl LocalHeap {
    /// Read a local heap from the given address.
    ///
    /// Composition mirrors the C side: `decode_prefix` parses the on-disk
    /// header (`H5HL__cache_prefix_deserialize`), then we follow the
    /// `data_addr` pointer to pull the actual heap bytes.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        let prefix = Self::decode_prefix(reader, addr)?;
        Self::load_data_segment(reader, &prefix)
    }

    /// Pure header decode: read & validate the local-heap prefix at `addr`,
    /// returning its parsed fields. No I/O against the data segment.
    pub fn decode_prefix<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<LocalHeapPrefix> {
        reader.seek(addr)?;

        // Magic
        let magic = reader.read_bytes(4)?;
        if magic != HEAP_MAGIC {
            return Err(Error::InvalidFormat("invalid local heap magic".into()));
        }

        // Version
        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!("local heap version {version}")));
        }

        // Reserved
        reader.skip(3)?;

        // Data segment size
        let data_size = reader.read_length()?;

        // Free list head offset
        let free_list_offset = reader.read_length()?;

        // Data segment address
        let data_addr = reader.read_addr()?;

        Ok(LocalHeapPrefix {
            data_size,
            free_list_offset,
            data_addr,
        })
    }

    /// Follow a decoded prefix to load the heap's data segment.
    pub fn load_data_segment<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        prefix: &LocalHeapPrefix,
    ) -> Result<Self> {
        reader.seek(prefix.data_addr)?;
        let data = reader.read_bytes(heap_len(prefix.data_size, "local heap data size")?)?;
        Ok(Self { data })
    }

    /// Get a null-terminated string at the given offset in the heap data.
    pub fn get_string(&self, offset: usize) -> Result<String> {
        if offset >= self.data.len() {
            return Err(Error::InvalidFormat(
                "local heap string offset is out of bounds".into(),
            ));
        }

        let end = self.data[offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| offset + p)
            .ok_or_else(|| {
                Error::InvalidFormat("local heap string is not null-terminated".into())
            })?;

        Ok(String::from_utf8_lossy(&self.data[offset..end]).to_string())
    }
}

fn heap_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_LOCAL_HEAP_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_LOCAL_HEAP_BYTES}"
        )));
    }
    Ok(len)
}
