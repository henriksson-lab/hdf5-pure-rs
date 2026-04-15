use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Local heap magic: "HEAP"
const HEAP_MAGIC: [u8; 4] = [b'H', b'E', b'A', b'P'];

/// A local heap stores variable-length strings (link names) for v1 groups.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// The raw data content of the heap.
    pub data: Vec<u8>,
}

impl LocalHeap {
    /// Read a local heap from the given address.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
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
        let _free_list_offset = reader.read_length()?;

        // Data segment address
        let data_addr = reader.read_addr()?;

        // Read the data segment
        reader.seek(data_addr)?;
        let data = reader.read_bytes(data_size as usize)?;

        Ok(Self { data })
    }

    /// Get a null-terminated string at the given offset in the heap data.
    pub fn get_string(&self, offset: usize) -> Option<String> {
        if offset >= self.data.len() {
            return None;
        }

        let end = self.data[offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| offset + p)
            .unwrap_or(self.data.len());

        Some(String::from_utf8_lossy(&self.data[offset..end]).to_string())
    }
}
