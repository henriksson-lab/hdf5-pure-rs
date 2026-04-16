use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
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
const MAX_HEAP_OBJECT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Fractal heap header.
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    pub heap_addr: u64,
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
            heap_addr: addr,
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
            let data = reader.read_bytes(heap_object_len(len, "huge heap object length")?)?;
            self.trace_huge_object(heap_id, addr, len, len, 0, false);
            return Ok(data);
        }

        if self.io_filter_len > 0 && heap_id.len() >= 1 + addr_size + len_size + 4 + len_size {
            let mut p = 1usize;
            let addr = read_le_uint(&heap_id[p..p + addr_size]);
            p += addr_size;
            let len = read_le_uint(&heap_id[p..p + len_size]);
            p += len_size;
            let filter_mask =
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
            let filtered =
                reader.read_bytes(heap_object_len(len, "filtered huge heap object length")?)?;
            let mut data = crate::filters::apply_pipeline_reverse(&filtered, pipeline, 1)?;
            data.truncate(heap_object_len(obj_size, "filtered huge heap object size")?);
            self.trace_huge_object(heap_id, addr, len, obj_size, filter_mask, true);
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
                let mut data =
                    reader.read_bytes(heap_object_len(huge.len, "huge heap object length")?)?;
                if huge.filtered {
                    let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                        Error::InvalidFormat("filtered huge object missing filter pipeline".into())
                    })?;
                    data = crate::filters::apply_pipeline_reverse(&data, pipeline, 1)?;
                    data.truncate(heap_object_len(
                        huge.obj_size.unwrap_or(data.len() as u64),
                        "filtered huge heap object size",
                    )?);
                }
                self.trace_huge_object(
                    heap_id,
                    huge.addr,
                    huge.len,
                    huge.obj_size.unwrap_or(huge.len),
                    0,
                    huge.filtered,
                );
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
                self.start_block_size,
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
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if let Some(filtered_size) = filtered_size {
            reader.seek(block_addr)?;
            let filtered = reader.read_bytes(heap_object_len(
                filtered_size,
                "filtered fractal heap block size",
            )?)?;
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("filtered fractal heap missing filter pipeline".into())
            })?;
            let data = crate::filters::apply_pipeline_reverse(&filtered, pipeline, 1)?;
            let start = heap_object_len(offset, "fractal heap object offset")?;
            let end = start
                .checked_add(heap_object_len(length, "fractal heap object length")?)
                .ok_or_else(|| Error::InvalidFormat("fractal heap object range overflow".into()))?;
            let out = data
                .get(start..end)
                .map(|slice| slice.to_vec())
                .ok_or_else(|| {
                    Error::InvalidFormat("fractal heap object exceeds filtered direct block".into())
                })?;
            self.trace_managed_object(block_addr, block_size, offset, length, filter_mask, true);
            return Ok(out);
        }

        if self.has_checksum && self.heap_addr == 0 {
            verify_direct_block_checksum(reader, block_addr, self.max_heap_size, block_size)?;
        }

        let addr = block_addr
            .checked_add(offset)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object address overflow".into()))?;
        reader.seek(addr)?;
        let data = reader.read_bytes(heap_object_len(length, "fractal heap object length")?)?;
        self.trace_managed_object(block_addr, block_size, offset, length, 0, false);
        Ok(data)
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
        if self.has_checksum {
            let checksum_span = 4usize
                .checked_add(1)
                .and_then(|n| n.checked_add(self.sizeof_addr as usize))
                .and_then(|n| n.checked_add(block_offset_bytes))
                .and_then(|n| {
                    n.checked_add(
                        nrows
                            .checked_mul(self.table_width as usize)?
                            .checked_mul(self.sizeof_addr as usize)?,
                    )
                })
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "fractal heap indirect block checksum span overflow".into(),
                    )
                })?;
            verify_metadata_checksum(
                reader,
                block_addr,
                checksum_span as u64,
                "fractal heap indirect block",
            )?;
        }

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
                            block_span,
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
                        block_size,
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
        let data = heap_id[1..1 + length].to_vec();
        self.trace_tiny_object(heap_id, length as u64);
        Ok(data)
    }

    #[cfg(feature = "tracehash")]
    fn trace_managed_object(
        &self,
        block_addr: u64,
        block_size: u64,
        block_offset: u64,
        object_len: u64,
        _filter_mask: u32,
        filtered: bool,
    ) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.managed_object");
        th.input_u64(self.heap_addr);
        th.input_u64(block_addr);
        th.input_u64(block_offset);
        th.input_u64(object_len);
        th.output_bool(true);
        th.output_u64(block_addr);
        th.output_u64(block_size);
        th.output_u64(block_offset);
        th.output_u64(object_len);
        th.output_u64(0);
        th.output_bool(filtered);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_managed_object(
        &self,
        _block_addr: u64,
        _block_size: u64,
        _block_offset: u64,
        _object_len: u64,
        _filter_mask: u32,
        _filtered: bool,
    ) {
    }

    #[cfg(feature = "tracehash")]
    fn trace_huge_object(
        &self,
        heap_id: &[u8],
        addr: u64,
        stored_len: u64,
        object_len: u64,
        filter_mask: u32,
        filtered: bool,
    ) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.huge_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_bool(true);
        th.output_u64(addr);
        th.output_u64(stored_len);
        th.output_u64(object_len);
        th.output_u64(filter_mask as u64);
        th.output_bool(filtered);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_huge_object(
        &self,
        _heap_id: &[u8],
        _addr: u64,
        _stored_len: u64,
        _object_len: u64,
        _filter_mask: u32,
        _filtered: bool,
    ) {
    }

    #[cfg(feature = "tracehash")]
    fn trace_tiny_object(&self, heap_id: &[u8], object_len: u64) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.tiny_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_bool(true);
        th.output_u64(object_len);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_tiny_object(&self, _heap_id: &[u8], _object_len: u64) {}
}

fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().take(8).enumerate() {
        value |= (*byte as u64) << (idx * 8);
    }
    value
}

fn verify_metadata_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    check_len: u64,
    context: &str,
) -> Result<()> {
    let restore = reader.position()?;
    let check_len_usize = heap_object_len(check_len, context)?;
    reader.seek(start)?;
    let bytes = reader.read_bytes(check_len_usize)?;
    let stored = reader.read_u32()?;
    let computed = checksum_metadata(&bytes);
    reader.seek(restore)?;
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

fn verify_direct_block_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    max_heap_size: u16,
    block_size: u64,
) -> Result<()> {
    let restore = reader.position()?;
    let block_offset_bytes = ((max_heap_size as usize) + 7) / 8;
    let check_len = 4usize
        .checked_add(1)
        .and_then(|n| n.checked_add(reader.sizeof_addr() as usize))
        .and_then(|n| n.checked_add(block_offset_bytes))
        .ok_or_else(|| {
            Error::InvalidFormat("fractal heap direct block checksum span overflow".into())
        })?;
    reader.seek(start)?;
    let bytes = reader.read_bytes(check_len)?;
    let stored = reader.read_u32()?;
    let payload_len = heap_object_len(block_size, "fractal heap direct block size")?;
    let payload = reader.read_bytes(payload_len)?;
    reader.seek(restore)?;

    let computed = checksum_metadata(&bytes);
    let mut with_zero_checksum = bytes.clone();
    with_zero_checksum.extend_from_slice(&0u32.to_le_bytes());
    let computed_with_zero_checksum = checksum_metadata(&with_zero_checksum);
    let mut whole_block = with_zero_checksum;
    whole_block.extend_from_slice(&payload);
    let computed_whole_block = checksum_metadata(&whole_block);
    if stored != computed && stored != computed_with_zero_checksum && stored != computed_whole_block
    {
        return Err(Error::InvalidFormat(format!(
            "fractal heap direct block checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

fn heap_object_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_HEAP_OBJECT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_HEAP_OBJECT_BYTES}"
        )));
    }
    Ok(len)
}

fn log2_power2(value: u64) -> u32 {
    debug_assert!(value.is_power_of_two());
    value.trailing_zeros()
}

fn log2_floor(value: u64) -> u32 {
    63 - value.leading_zeros()
}
