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
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn create(size_hint: usize) -> Self {
        Self {
            data: Vec::with_capacity(size_hint),
        }
    }

    pub fn protect(heap: Self) -> Self {
        heap
    }

    pub fn unprotect(_heap: Self) {}

    pub fn remove_free(&mut self) {
        while self.data.last().copied() == Some(0) {
            self.data.pop();
        }
    }

    pub fn dirty(&mut self) {}

    pub fn insert(&mut self, value: &[u8]) -> Result<usize> {
        let offset = self.data.len();
        self.data.extend_from_slice(value);
        if !value.ends_with(&[0]) {
            self.data.push(0);
        }
        Ok(offset)
    }

    pub fn remove(&mut self, offset: usize) -> Result<()> {
        let end = self.string_end(offset)?;
        for byte in &mut self.data[offset..=end] {
            *byte = 0;
        }
        Ok(())
    }

    pub fn delete(&mut self) {
        self.data.clear();
    }

    pub fn heap_get_size(&self) -> usize {
        self.data.len()
    }

    pub fn get_size(&self) -> usize {
        self.heap_get_size()
    }

    pub fn heapsize(&self) -> usize {
        self.heap_get_size()
    }

    pub fn prfx_new(data_size: u64, free_list_offset: u64, data_addr: u64) -> LocalHeapPrefix {
        LocalHeapPrefix {
            data_size,
            free_list_offset,
            data_addr,
        }
    }

    pub fn prfx_dest(_prefix: LocalHeapPrefix) {}

    pub fn hdr_deserialize<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<LocalHeapPrefix> {
        Self::decode_prefix(reader, addr)
    }

    pub fn fl_deserialize(data: &[u8]) -> Vec<(u64, u64)> {
        let mut entries = Vec::new();
        let mut pos = 0usize;
        while pos + 16 <= data.len() {
            let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            let size = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap());
            entries.push((offset, size));
            pos += 16;
        }
        entries
    }

    pub fn fl_serialize(entries: &[(u64, u64)]) -> Vec<u8> {
        let mut out = Vec::with_capacity(entries.len() * 16);
        for &(offset, size) in entries {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
        }
        out
    }

    pub fn cache_prefix_get_initial_load_size() -> usize {
        8
    }

    pub fn cache_prefix_get_final_load_size(addr_size: usize, length_size: usize) -> Result<usize> {
        Self::cache_prefix_image_len(addr_size, length_size)
    }

    pub fn cache_prefix_image_len(addr_size: usize, length_size: usize) -> Result<usize> {
        4usize
            .checked_add(4)
            .and_then(|value| value.checked_add(length_size))
            .and_then(|value| value.checked_add(length_size))
            .and_then(|value| value.checked_add(addr_size))
            .ok_or_else(|| Error::InvalidFormat("local heap prefix image length overflow".into()))
    }

    pub fn cache_prefix_serialize(
        prefix: &LocalHeapPrefix,
        addr_size: usize,
        length_size: usize,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(Self::cache_prefix_image_len(addr_size, length_size)?);
        out.extend_from_slice(&HEAP_MAGIC);
        out.push(0);
        out.extend_from_slice(&[0, 0, 0]);
        encode_var(&mut out, prefix.data_size, length_size)?;
        encode_var(&mut out, prefix.free_list_offset, length_size)?;
        encode_var(&mut out, prefix.data_addr, addr_size)?;
        Ok(out)
    }

    pub fn cache_prefix_free_icr(_prefix: LocalHeapPrefix) {}

    pub fn cache_datablock_get_initial_load_size(prefix: &LocalHeapPrefix) -> Result<usize> {
        heap_len(prefix.data_size, "local heap data size")
    }

    pub fn cache_datablock_image_len(&self) -> usize {
        self.data.len()
    }

    pub fn cache_datablock_serialize(&self) -> Vec<u8> {
        self.data.clone()
    }

    pub fn cache_datablock_notify(&self) {}

    pub fn cache_datablock_free_icr(self) {}

    pub fn inc_rc(&mut self) {}

    pub fn dec_rc(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn dest(self) {}

    pub fn dblk_new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn dblk_dest(self) {}

    pub fn dblk_realloc(&mut self, new_size: usize) {
        self.data.resize(new_size, 0);
    }

    pub fn debug(&self) -> String {
        format!("LocalHeap(size={})", self.data.len())
    }

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

    fn string_end(&self, offset: usize) -> Result<usize> {
        if offset >= self.data.len() {
            return Err(Error::InvalidFormat(
                "local heap string offset is out of bounds".into(),
            ));
        }
        self.data[offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| offset + p)
            .ok_or_else(|| Error::InvalidFormat("local heap string is not null-terminated".into()))
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

fn encode_var(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size > 8 {
        return Err(Error::InvalidFormat(
            "local heap encoded integer size exceeds u64".into(),
        ));
    }
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..size]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_and_remove_string() {
        let mut heap = LocalHeap::new();
        let offset = heap.insert(b"name").unwrap();
        assert_eq!(heap.get_string(offset).unwrap(), "name");
        heap.remove(offset).unwrap();
        assert_eq!(heap.data[offset], 0);
    }

    #[test]
    fn prefix_serializes_fixed_header_fields() {
        let prefix = LocalHeap::prfx_new(16, 0, 128);
        let encoded = LocalHeap::cache_prefix_serialize(&prefix, 8, 8).unwrap();
        assert_eq!(&encoded[..4], b"HEAP");
        assert_eq!(encoded[4], 0);
    }
}
