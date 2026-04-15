use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
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
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
    pub huge_btree_addr: u64,
    pub root_direct_filtered_size: Option<u64>,
    pub root_direct_filter_mask: u32,
    pub filter_pipeline: Option<FilterPipelineMessage>,
}

#[derive(Debug, Clone, Copy)]
struct HugeRecord {
    addr: u64,
    len: u64,
    filtered: bool,
    obj_size: Option<u64>,
    id: Option<u64>,
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
        let huge_btree_addr = reader.read_addr()?; // sizeof_addr

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
        let mut root_direct_filtered_size = None;
        let mut root_direct_filter_mask = 0;
        let mut filter_pipeline = None;
        if io_filter_len > 0 {
            root_direct_filtered_size = Some(reader.read_length()?);
            root_direct_filter_mask = reader.read_u32()?;
            let pipeline_bytes = reader.read_bytes(io_filter_len as usize)?;
            filter_pipeline = Some(FilterPipelineMessage::decode(&pipeline_bytes)?);
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
            sizeof_addr: reader.sizeof_addr(),
            sizeof_size: reader.sizeof_size(),
            huge_btree_addr,
            root_direct_filtered_size,
            root_direct_filter_mask,
            filter_pipeline,
        })
    }

    /// Read a managed object from the fractal heap by its heap ID.
    pub fn read_managed_object<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        if heap_id.is_empty() {
            return Err(Error::InvalidFormat("empty heap ID".into()));
        }

        let id_type = (heap_id[0] >> 4) & 0x03;

        match id_type {
            0 => self.read_managed(reader, heap_id),
            1 => self.read_huge(reader, heap_id),
            2 => self.read_tiny(heap_id),
            _ => Err(Error::InvalidFormat(format!(
                "unknown heap ID type {id_type}"
            ))),
        }
    }

    fn read_huge<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        let addr_size = self.sizeof_addr as usize;
        let len_size = self.sizeof_size as usize;

        if self.io_filter_len == 0 && heap_id.len() >= 1 + addr_size + len_size {
            let mut p = 1usize;
            let addr = read_le_uint(&heap_id[p..p + addr_size]);
            p += addr_size;
            let len = read_le_uint(&heap_id[p..p + len_size]);
            if crate::io::reader::is_undef_addr(addr) {
                return Err(Error::InvalidFormat(
                    "huge heap object has undefined address".into(),
                ));
            }
            reader.seek(addr)?;
            return reader.read_bytes(len as usize);
        }

        if self.io_filter_len > 0 && heap_id.len() >= 1 + addr_size + len_size + 4 + len_size {
            let mut p = 1usize;
            let addr = read_le_uint(&heap_id[p..p + addr_size]);
            p += addr_size;
            let len = read_le_uint(&heap_id[p..p + len_size]);
            p += len_size;
            let _filter_mask =
                u32::from_le_bytes([heap_id[p], heap_id[p + 1], heap_id[p + 2], heap_id[p + 3]]);
            p += 4;
            let obj_size = read_le_uint(&heap_id[p..p + len_size]);
            if crate::io::reader::is_undef_addr(addr) {
                return Err(Error::InvalidFormat(
                    "huge heap object has undefined address".into(),
                ));
            }
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("filtered huge object missing filter pipeline".into())
            })?;
            reader.seek(addr)?;
            let filtered = reader.read_bytes(len as usize)?;
            let mut data = crate::filters::apply_pipeline_reverse(&filtered, pipeline, 1)?;
            data.truncate(obj_size as usize);
            return Ok(data);
        }

        if crate::io::reader::is_undef_addr(self.huge_btree_addr) {
            return Err(Error::InvalidFormat(
                "huge heap object ID is indirect but heap has no huge-object B-tree".into(),
            ));
        }

        let id = read_le_uint(&heap_id[1..]);
        let records = crate::format::btree_v2::collect_all_records(reader, self.huge_btree_addr)?;
        for record in records {
            let huge = self.decode_huge_record(&record)?;
            if huge.id == Some(id) {
                reader.seek(huge.addr)?;
                let mut data = reader.read_bytes(huge.len as usize)?;
                if huge.filtered {
                    let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                        Error::InvalidFormat("filtered huge object missing filter pipeline".into())
                    })?;
                    data = crate::filters::apply_pipeline_reverse(&data, pipeline, 1)?;
                    data.truncate(huge.obj_size.unwrap_or(data.len() as u64) as usize);
                }
                return Ok(data);
            }
        }

        Err(Error::InvalidFormat(format!(
            "huge fractal heap object id {id} not found"
        )))
    }

    fn decode_huge_record(&self, record: &[u8]) -> Result<HugeRecord> {
        let sa = self.sizeof_addr as usize;
        let ss = self.sizeof_size as usize;
        if record.len() == sa + ss {
            return Ok(HugeRecord {
                addr: read_le_uint(&record[..sa]),
                len: read_le_uint(&record[sa..sa + ss]),
                filtered: false,
                obj_size: None,
                id: None,
            });
        }
        if record.len() == sa + ss + ss {
            return Ok(HugeRecord {
                addr: read_le_uint(&record[..sa]),
                len: read_le_uint(&record[sa..sa + ss]),
                filtered: false,
                obj_size: None,
                id: Some(read_le_uint(&record[sa + ss..sa + ss + ss])),
            });
        }
        if record.len() == sa + ss + 4 + ss {
            return Ok(HugeRecord {
                addr: read_le_uint(&record[..sa]),
                len: read_le_uint(&record[sa..sa + ss]),
                filtered: true,
                obj_size: Some(read_le_uint(&record[sa + ss + 4..sa + ss + 4 + ss])),
                id: None,
            });
        }
        if record.len() == sa + ss + 4 + ss + ss {
            return Ok(HugeRecord {
                addr: read_le_uint(&record[..sa]),
                len: read_le_uint(&record[sa..sa + ss]),
                filtered: true,
                obj_size: Some(read_le_uint(&record[sa + ss + 4..sa + ss + 4 + ss])),
                id: Some(read_le_uint(
                    &record[sa + ss + 4 + ss..sa + ss + 4 + ss + ss],
                )),
            });
        }

        Err(Error::Unsupported(format!(
            "unsupported huge fractal heap B-tree record size {}",
            record.len()
        )))
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
            self.read_from_direct_block(
                reader,
                self.root_block_addr,
                self.root_direct_filtered_size,
                self.root_direct_filter_mask,
                offset,
                length,
            )
        } else {
            // Root is an indirect block -- need to find which direct block contains the offset
            self.read_from_indirect_block(reader, self.root_block_addr, offset, length)
        }
    }

    fn read_from_direct_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        filtered_size: Option<u64>,
        _filter_mask: u32,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if let Some(filtered_size) = filtered_size {
            reader.seek(block_addr)?;
            let filtered = reader.read_bytes(filtered_size as usize)?;
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("filtered fractal heap missing filter pipeline".into())
            })?;
            let data = crate::filters::apply_pipeline_reverse(&filtered, pipeline, 1)?;
            let start = offset as usize;
            let end = start
                .checked_add(length as usize)
                .ok_or_else(|| Error::InvalidFormat("fractal heap object range overflow".into()))?;
            return data
                .get(start..end)
                .map(|slice| slice.to_vec())
                .ok_or_else(|| {
                    Error::InvalidFormat("fractal heap object exceeds filtered direct block".into())
                });
        }

        reader.seek(block_addr + offset)?;
        reader.read_bytes(length as usize)
    }

    fn read_from_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if self.io_filter_len > 0 {
            return self.read_from_filtered_indirect_block(reader, block_addr, offset, length);
        }

        self.read_from_indirect_block_rows(
            reader,
            block_addr,
            self.current_root_rows as usize,
            0,
            offset,
            length,
        )
    }

    fn read_from_indirect_block_rows<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        nrows: usize,
        block_start: u64,
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
        let _block_offset = read_le_uint(&reader.read_bytes(block_offset_bytes)?);

        let width = self.table_width as usize;
        let max_direct_rows = self.max_direct_rows();
        let mut current_heap_offset = block_start;

        for row in 0..nrows {
            if row < max_direct_rows {
                let block_span = self.row_block_size(row);
                for _ in 0..width {
                    let child_addr = reader.read_addr()?;

                    if crate::io::reader::is_undef_addr(child_addr) {
                        current_heap_offset += block_span;
                        continue;
                    }

                    if offset >= current_heap_offset && offset < current_heap_offset + block_span {
                        let local_offset = offset - current_heap_offset;
                        return self.read_from_direct_block(
                            reader,
                            child_addr,
                            None,
                            0,
                            local_offset,
                            length,
                        );
                    }

                    current_heap_offset += block_span;
                }
            } else {
                let child_rows = self.child_indirect_rows(row);
                let child_span = self.indirect_data_span(reader, child_rows)?;
                for _ in 0..width {
                    let child_addr = reader.read_addr()?;
                    if offset >= current_heap_offset && offset < current_heap_offset + child_span {
                        if crate::io::reader::is_undef_addr(child_addr) {
                            break;
                        }
                        return self.read_from_indirect_block_rows(
                            reader,
                            child_addr,
                            child_rows,
                            current_heap_offset,
                            offset,
                            length,
                        );
                    }
                    current_heap_offset += child_span;
                }
            }
        }

        Err(Error::InvalidFormat(format!(
            "fractal heap offset {offset} not found in indirect block"
        )))
    }

    fn max_direct_rows(&self) -> usize {
        let start_bits = log2_power2(self.start_block_size);
        let max_direct_bits = log2_power2(self.max_direct_block_size);
        (max_direct_bits - start_bits + 2) as usize
    }

    fn child_indirect_rows(&self, row: usize) -> usize {
        let first_row_bits =
            log2_power2(self.start_block_size) + log2_power2(self.table_width as u64);
        (log2_floor(self.row_block_size(row)) - first_row_bits + 1) as usize
    }

    fn indirect_data_span<R: Read + Seek>(
        &self,
        reader: &HdfReader<R>,
        nrows: usize,
    ) -> Result<u64> {
        let width = self.table_width as u64;
        let max_direct_rows = self.max_direct_rows();
        let mut span = 0u64;

        for row in 0..nrows {
            if row < max_direct_rows {
                span = span
                    .checked_add(self.row_block_size(row) * width)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap span overflow".into()))?;
            } else {
                let child_rows = self.child_indirect_rows(row);
                span = span
                    .checked_add(self.indirect_data_span(reader, child_rows)? * width)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap span overflow".into()))?;
            }
        }

        Ok(span)
    }

    fn row_block_size(&self, row: usize) -> u64 {
        if row == 0 {
            self.start_block_size
        } else {
            self.start_block_size * (1u64 << (row - 1))
        }
    }

    fn read_from_filtered_indirect_block<R: Read + Seek>(
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

        let nrows = self.current_root_rows as usize;
        let width = self.table_width as usize;
        let mut current_heap_offset = 0u64;
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
                for _ in 0..width {
                    let _child = reader.read_addr()?;
                }
                continue;
            }

            let data_capacity = block_size - dblock_header_size;
            for _ in 0..width {
                let child_addr = reader.read_addr()?;
                let filtered_size = reader.read_length()?;
                let filter_mask = reader.read_u32()?;
                if crate::io::reader::is_undef_addr(child_addr) {
                    current_heap_offset += data_capacity;
                    continue;
                }
                if offset >= current_heap_offset && offset < current_heap_offset + data_capacity {
                    return self.read_from_direct_block(
                        reader,
                        child_addr,
                        Some(filtered_size),
                        filter_mask,
                        offset - current_heap_offset,
                        length,
                    );
                }
                current_heap_offset += data_capacity;
            }
        }

        Err(Error::InvalidFormat(format!(
            "filtered fractal heap offset {offset} not found in indirect block"
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

fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().take(8).enumerate() {
        value |= (*byte as u64) << (idx * 8);
    }
    value
}

fn log2_power2(value: u64) -> u32 {
    debug_assert!(value.is_power_of_two());
    value.trailing_zeros()
}

fn log2_floor(value: u64) -> u32 {
    63 - value.leading_zeros()
}
