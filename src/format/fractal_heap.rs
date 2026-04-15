use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Fractal heap header magic: "FRHP"
const FRHP_MAGIC: [u8; 4] = [b'F', b'R', b'H', b'P'];
/// Direct block magic: "FHDB"
/// Direct block magic (not currently used since we read by offset, kept for reference).
#[allow(dead_code)]
const FHDB_MAGIC: [u8; 4] = [b'F', b'H', b'D', b'B'];
/// Indirect block magic: "FHIB"
const FHIB_MAGIC: [u8; 4] = [b'F', b'H', b'I', b'B'];

/// Fractal heap header.
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    pub heap_id_len: u16,
    pub io_filter_len: u16,
    pub flags: u8,
    pub max_managed_obj_size: u32,

    pub table_width: u16,
    pub start_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size: u16,
    pub start_root_rows: u16,
    pub root_block_addr: u64,
    pub current_root_rows: u16,
    pub num_managed_objects: u64,
    pub has_checksum: bool,
}

impl FractalHeapHeader {
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != FRHP_MAGIC {
            return Err(Error::InvalidFormat("invalid fractal heap magic".into()));
        }

        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "fractal heap version {version}"
            )));
        }

        let heap_id_len = reader.read_u16()?;
        let io_filter_len = reader.read_u16()?;
        let flags = reader.read_u8()?;

        // "Huge" object info
        let max_managed_obj_size = reader.read_u32()?;
        let _next_huge_id = reader.read_length()?; // sizeof_size
        let _huge_btree_addr = reader.read_addr()?; // sizeof_addr

        // Managed free space
        let _total_man_free = reader.read_length()?; // sizeof_size
        let _fs_addr = reader.read_addr()?; // sizeof_addr

        // Heap statistics
        let _man_size = reader.read_length()?; // sizeof_size
        let _man_alloc_size = reader.read_length()?; // sizeof_size
        let _man_iter_off = reader.read_length()?; // sizeof_size
        let num_managed_objects = reader.read_length()?; // sizeof_size
        let _huge_size = reader.read_length()?; // sizeof_size
        let _huge_nobjs = reader.read_length()?; // sizeof_size
        let _tiny_size = reader.read_length()?; // sizeof_size
        let _tiny_nobjs = reader.read_length()?; // sizeof_size

        // Doubling table info
        let table_width = reader.read_u16()?;
        let start_block_size = reader.read_length()?; // sizeof_size
        let max_direct_block_size = reader.read_length()?; // sizeof_size
        let max_heap_size = reader.read_u16()?;
        let start_root_rows = reader.read_u16()?;
        let root_block_addr = reader.read_addr()?; // sizeof_addr
        let current_root_rows = reader.read_u16()?;

        let has_checksum = flags & 0x02 != 0;

        // If I/O filters present, skip filter info
        if io_filter_len > 0 {
            let _filtered_root_size = reader.read_length()?;
            let _filter_mask = reader.read_u32()?;
            reader.skip(io_filter_len as u64)?;
        }

        // Checksum
        let _checksum = reader.read_u32()?;

        Ok(Self {
            heap_id_len,
            io_filter_len,
            flags,
            max_managed_obj_size,
            table_width,
            start_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            root_block_addr,
            current_root_rows,
            num_managed_objects,
            has_checksum,
        })
    }

    /// Read a managed object from the fractal heap by its heap ID.
    pub fn read_managed_object<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        if self.io_filter_len > 0 {
            return Err(Error::Unsupported(
                "filtered fractal heaps are not implemented".into(),
            ));
        }

        if heap_id.is_empty() {
            return Err(Error::InvalidFormat("empty heap ID".into()));
        }

        let id_type = (heap_id[0] >> 4) & 0x03;

        match id_type {
            0 => self.read_managed(reader, heap_id),
            1 => Err(Error::Unsupported("huge objects in fractal heap".into())),
            2 => self.read_tiny(heap_id),
            _ => Err(Error::InvalidFormat(format!(
                "unknown heap ID type {id_type}"
            ))),
        }
    }

    /// Read a managed (type 0) object.
    fn read_managed<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        // Managed object heap ID:
        // byte 0: version(2 bits) + type(2 bits) + reserved(4 bits)
        // then: offset (ceil(max_heap_size/8) bytes) + length (remaining bytes)

        let offset_bytes = ((self.max_heap_size as usize) + 7) / 8;

        if heap_id.len() < 1 + offset_bytes {
            return Err(Error::InvalidFormat("heap ID too short for offset".into()));
        }

        let mut offset = 0u64;
        for i in 0..offset_bytes {
            offset |= (heap_id[1 + i] as u64) << (i * 8);
        }

        let len_start = 1 + offset_bytes;
        let mut length = 0u64;
        for i in 0..(heap_id.len() - len_start).min(8) {
            length |= (heap_id[len_start + i] as u64) << (i * 8);
        }

        if self.current_root_rows == 0 {
            // Root is a direct block -- offset is relative to block start
            self.read_from_direct_block(reader, self.root_block_addr, offset, length)
        } else {
            // Root is an indirect block -- need to find which direct block contains the offset
            self.read_from_indirect_block(reader, self.root_block_addr, offset, length)
        }
    }

    fn read_from_direct_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        // The offset in the heap ID is relative to the start of the block,
        // INCLUDING the block header. So we just seek to block_addr + offset.
        reader.seek(block_addr + offset)?;
        let data = reader.read_bytes(length as usize)?;
        Ok(data)
    }

    fn read_from_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        reader.seek(block_addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != FHIB_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid fractal heap indirect block magic".into(),
            ));
        }

        let _version = reader.read_u8()?;
        let _heap_header_addr = reader.read_addr()?;

        let block_offset_bytes = ((self.max_heap_size as usize) + 7) / 8;
        reader.skip(block_offset_bytes as u64)?;

        // Navigate direct block entries to find the one containing our offset
        let nrows = self.current_root_rows as usize;
        let width = self.table_width as usize;
        let mut current_heap_offset = 0u64;

        // Calculate the block size overhead for a direct block
        let dblock_header_size = 5
            + reader.sizeof_addr() as u64
            + block_offset_bytes as u64
            + if self.has_checksum { 4 } else { 0 };

        for row in 0..nrows {
            let block_size = if row < 2 {
                self.start_block_size
            } else {
                self.start_block_size * (1u64 << (row - 1))
            };

            if block_size > self.max_direct_block_size {
                // Indirect block entries -- skip for now
                for _ in 0..width {
                    let _child = reader.read_addr()?;
                }
                continue;
            }

            let data_capacity = block_size - dblock_header_size;

            for _ in 0..width {
                let child_addr = reader.read_addr()?;

                if crate::io::reader::is_undef_addr(child_addr) {
                    current_heap_offset += data_capacity;
                    continue;
                }

                if offset >= current_heap_offset && offset < current_heap_offset + data_capacity {
                    let local_offset = offset - current_heap_offset;
                    return self.read_from_direct_block(reader, child_addr, local_offset, length);
                }

                current_heap_offset += data_capacity;
            }
        }

        Err(Error::InvalidFormat(format!(
            "fractal heap offset {offset} not found in indirect block"
        )))
    }

    fn read_tiny(&self, heap_id: &[u8]) -> Result<Vec<u8>> {
        let length = (heap_id[0] & 0x0F) as usize + 1;
        if heap_id.len() < 1 + length {
            return Err(Error::InvalidFormat("tiny heap ID too short".into()));
        }
        Ok(heap_id[1..1 + length].to_vec())
    }
}
