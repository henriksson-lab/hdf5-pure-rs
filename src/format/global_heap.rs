use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Global heap collection magic: "GCOL"
const GCOL_MAGIC: [u8; 4] = [b'G', b'C', b'O', b'L'];

/// A global heap object reference (collection address + object index).
#[derive(Debug, Clone)]
pub struct GlobalHeapRef {
    pub collection_addr: u64,
    pub object_index: u32,
}

/// A global heap collection containing objects.
#[derive(Debug)]
pub struct GlobalHeapCollection {
    pub objects: Vec<(u32, Vec<u8>)>, // (index, data)
}

impl GlobalHeapCollection {
    /// Read a global heap collection at the given address.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != GCOL_MAGIC {
            return Err(Error::InvalidFormat("invalid global heap collection magic".into()));
        }

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::Unsupported(format!("global heap version {version}")));
        }

        // Reserved (3 bytes)
        reader.skip(3)?;

        // Collection size (includes header)
        let collection_size = reader.read_length()?;

        let mut objects = Vec::new();
        let _header_size = 8 + reader.sizeof_size() as u64; // magic + ver + reserved + size
        let data_end = addr + collection_size;

        while reader.position()? < data_end {
            let pos = reader.position()?;
            if pos + 8 > data_end {
                break;
            }

            let index = reader.read_u16()? as u32;
            if index == 0 {
                // Object index 0 = free space marker, end of collection
                break;
            }

            let _reference_count = reader.read_u16()?;
            // Reserved (4 bytes)
            reader.skip(4)?;
            let obj_size = reader.read_length()?;

            let data = reader.read_bytes(obj_size as usize)?;
            objects.push((index, data));

            // Pad to 8-byte boundary
            let padded = (obj_size + 7) & !7;
            let padding = padded - obj_size;
            if padding > 0 {
                reader.skip(padding)?;
            }
        }

        Ok(Self { objects })
    }

    /// Get an object by index from this collection.
    pub fn get_object(&self, index: u32) -> Option<&[u8]> {
        self.objects.iter()
            .find(|(idx, _)| *idx == index)
            .map(|(_, data)| data.as_slice())
    }
}

/// Read a global heap object by its reference.
pub fn read_global_heap_object<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    gh_ref: &GlobalHeapRef,
) -> Result<Vec<u8>> {
    let collection = GlobalHeapCollection::read_at(reader, gh_ref.collection_addr)?;
    collection
        .get_object(gh_ref.object_index)
        .map(|d| d.to_vec())
        .ok_or_else(|| {
            Error::InvalidFormat(format!(
                "global heap object {} not found in collection at {:#x}",
                gh_ref.object_index, gh_ref.collection_addr
            ))
        })
}
