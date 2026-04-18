use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Global heap collection magic: "GCOL"
const GCOL_MAGIC: [u8; 4] = [b'G', b'C', b'O', b'L'];
const MAX_GLOBAL_HEAP_OBJECT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// A global heap object reference (collection address + object index).
#[derive(Debug, Clone)]
pub struct GlobalHeapRef {
    pub collection_addr: u64,
    pub object_index: u32,
}

/// Decoded global-heap header — the 16-byte (or so) prefix that precedes
/// the object table. Mirrors the work `H5HG__cache_heap_deserialize` does
/// in libhdf5: magic + version + collection size, no object iteration.
#[derive(Debug, Clone, Copy)]
pub struct GlobalHeapHeader {
    /// Address the collection lives at (anchor for object-table walking).
    pub addr: u64,
    /// Total collection size (includes the header itself).
    pub collection_size: u64,
}

/// A global heap collection containing objects.
#[derive(Debug)]
pub struct GlobalHeapCollection {
    pub objects: Vec<(u32, Vec<u8>)>, // (index, data)
}

impl GlobalHeapCollection {
    /// Read a global heap collection at the given address.
    ///
    /// Composition mirrors the C cache layer: `decode_header` parses the
    /// fixed prefix, then `walk_objects` consumes the decoded header to
    /// iterate the variable-length object table.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        let header = Self::decode_header(reader, addr)?;
        Self::walk_objects(reader, &header)
    }

    /// Pure header decode: validate magic+version, return `(addr,
    /// collection_size)`. Leaves the reader positioned at the first object
    /// entry so that callers don't have to reseek.
    pub fn decode_header<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<GlobalHeapHeader> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != GCOL_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid global heap collection magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::Unsupported(format!("global heap version {version}")));
        }

        // Reserved (3 bytes)
        reader.skip(3)?;

        // Collection size (includes header)
        let collection_size = reader.read_length()?;

        Ok(GlobalHeapHeader {
            addr,
            collection_size,
        })
    }

    /// Walk the object table from the reader's current position (which the
    /// header decoder already advanced to). Stops at the end of the
    /// declared collection or at an index-0 sentinel entry.
    pub fn walk_objects<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header: &GlobalHeapHeader,
    ) -> Result<Self> {
        let mut objects: Vec<(u32, Vec<u8>)> = Vec::new();
        let data_end = header
            .addr
            .checked_add(header.collection_size)
            .ok_or_else(|| Error::InvalidFormat("global heap collection size overflow".into()))?;

        while reader.position()? < data_end {
            let pos = reader.position()?;
            if pos + 16 > data_end {
                break;
            }

            let index = reader.read_u16()? as u32;
            if index == 0 {
                // Object index 0 marks free space/end of the collection.
                break;
            }
            let _reference_count = reader.read_u16()?;
            // Reserved (4 bytes)
            reader.skip(4)?;
            let obj_size = reader.read_length()?;
            let obj_len = heap_object_len(obj_size, "global heap object size")?;
            let padded = obj_size
                .checked_add(7)
                .map(|size| size & !7)
                .ok_or_else(|| Error::InvalidFormat("global heap object size overflow".into()))?;
            let next_pos = reader
                .position()?
                .checked_add(padded)
                .ok_or_else(|| Error::InvalidFormat("global heap object offset overflow".into()))?;
            if next_pos > data_end {
                return Err(Error::InvalidFormat(
                    "global heap object exceeds collection bounds".into(),
                ));
            }

            let data = reader.read_bytes(obj_len)?;
            objects.push((index, data));

            // Pad to 8-byte boundary
            let padding = padded - obj_size;
            if padding > 0 {
                reader.skip(padding)?;
            }
        }

        Ok(Self { objects })
    }

    /// Get an object by index from this collection.
    pub fn get_object(&self, index: u32) -> Option<&[u8]> {
        self.objects
            .iter()
            .find(|(idx, _)| *idx == index)
            .map(|(_, data)| data.as_slice())
    }
}

fn heap_object_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_GLOBAL_HEAP_OBJECT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_GLOBAL_HEAP_OBJECT_BYTES}"
        )));
    }
    Ok(len)
}

/// Read a global heap object by its reference.
pub fn read_global_heap_object<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    gh_ref: &GlobalHeapRef,
) -> Result<Vec<u8>> {
    let collection = GlobalHeapCollection::read_at(reader, gh_ref.collection_addr)?;
    let data = collection
        .get_object(gh_ref.object_index)
        .map(|d| d.to_vec())
        .ok_or_else(|| {
            Error::InvalidFormat(format!(
                "global heap object {} not found in collection at {:#x}",
                gh_ref.object_index, gh_ref.collection_addr
            ))
        })?;
    trace_global_heap_deref(gh_ref, &data);
    Ok(data)
}

#[cfg(feature = "tracehash")]
fn trace_global_heap_deref(gh_ref: &GlobalHeapRef, data: &[u8]) {
    let mut th = tracehash::th_call!("hdf5.global_heap.deref");
    th.input_u64(gh_ref.collection_addr);
    th.input_u64(gh_ref.object_index as u64);
    th.output_value(&(true));
    th.output_u64(data.len() as u64);
    th.output_value(data);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_global_heap_deref(_gh_ref: &GlobalHeapRef, _data: &[u8]) {}
