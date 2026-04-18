//! Fractal heap huge-object access — mirrors libhdf5's `H5HFhuge.c` plus
//! `H5HFbtree2.c` (the v2 B-tree record decode for huge objects).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{heap_object_len, read_le_uint, FractalHeapHeader};

#[derive(Debug, Clone, Copy)]
pub(super) struct HugeRecord {
    pub(super) addr: u64,
    pub(super) len: u64,
    pub(super) filtered: bool,
    pub(super) obj_size: Option<u64>,
    pub(super) id: Option<u64>,
}

impl FractalHeapHeader {
    pub(super) fn read_huge<R: Read + Seek>(
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

    pub(super) fn decode_huge_record(&self, record: &[u8]) -> Result<HugeRecord> {
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
}
