use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v2;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::object_header::{self, ObjectHeader};
use crate::hl::file::FileInner;

/// An HDF5 attribute, parsed from an object header.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub msg: AttributeMessage,
}

impl Attribute {
    /// Get the attribute name.
    pub fn name(&self) -> &str {
        &self.msg.name
    }

    /// Get the raw data bytes of the attribute value.
    pub fn raw_data(&self) -> &[u8] {
        &self.msg.data
    }

    /// Get the datatype size in bytes.
    pub fn element_size(&self) -> usize {
        self.msg.datatype.size as usize
    }

    /// Get the shape of the attribute.
    pub fn shape(&self) -> &[u64] {
        &self.msg.dataspace.dims
    }

    /// Try to read the attribute as a single f64 scalar.
    pub fn read_scalar_f64(&self) -> Option<f64> {
        if self.msg.data.len() >= 8 {
            Some(f64::from_le_bytes(self.msg.data[..8].try_into().ok()?))
        } else {
            None
        }
    }

    /// Try to read the attribute as a single i64 scalar.
    pub fn read_scalar_i64(&self) -> Option<i64> {
        if self.msg.data.len() >= 8 {
            Some(i64::from_le_bytes(self.msg.data[..8].try_into().ok()?))
        } else if self.msg.data.len() >= 4 {
            Some(i32::from_le_bytes(self.msg.data[..4].try_into().ok()?) as i64)
        } else {
            None
        }
    }

    /// Read the attribute value as a typed Vec.
    pub fn read<T: crate::hl::types::H5Type>(&self) -> crate::Result<Vec<T>> {
        crate::hl::types::bytes_to_vec::<T>(self.msg.data.clone())
    }

    /// Read the attribute as a typed scalar.
    pub fn read_scalar<T: crate::hl::types::H5Type>(&self) -> crate::Result<T> {
        // Use bytes_to_vec for alignment safety, then take first element
        let vec = self.read::<T>()?;
        if vec.is_empty() {
            return Err(crate::Error::InvalidFormat(
                "no data for scalar read".into(),
            ));
        }
        Ok(vec[0])
    }

    /// Read the attribute as a string (for fixed-length string attributes).
    pub fn read_string(&self) -> String {
        let data = &self.msg.data;
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        String::from_utf8_lossy(&data[..end]).trim_end().to_string()
    }
}

/// Collect all attributes from an object header at the given address.
pub(crate) fn collect_attributes(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
) -> Result<Vec<Attribute>> {
    let mut guard = inner.lock();
    let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;

    let mut attrs = Vec::new();
    for msg in &oh.messages {
        if msg.msg_type == object_header::MSG_ATTRIBUTE {
            match AttributeMessage::decode(&msg.data) {
                Ok(attr_msg) => attrs.push(Attribute { msg: attr_msg }),
                Err(e) => {
                    // Skip malformed attributes
                    eprintln!("Warning: failed to decode attribute: {e}");
                }
            }
        }
    }

    for msg in &oh.messages {
        if msg.msg_type == object_header::MSG_ATTR_INFO {
            let attr_info = AttributeInfoMessage::decode(&msg.data, guard.superblock.sizeof_addr)?;
            if attr_info.has_dense_storage() {
                let heap =
                    FractalHeapHeader::read_at(&mut guard.reader, attr_info.fractal_heap_addr)?;
                let records =
                    btree_v2::collect_all_records(&mut guard.reader, attr_info.name_btree_addr)?;
                let heap_id_len = heap.heap_id_len as usize;

                for record in &records {
                    if record.len() < heap_id_len {
                        continue;
                    }
                    let heap_id = &record[..heap_id_len];
                    if let Ok(attr_data) = heap.read_managed_object(&mut guard.reader, heap_id) {
                        match AttributeMessage::decode(&attr_data) {
                            Ok(attr_msg) => attrs.push(Attribute { msg: attr_msg }),
                            Err(e) => {
                                eprintln!("Warning: failed to decode dense attribute: {e}");
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(attrs)
}

/// Get attribute names from an object header.
pub(crate) fn attr_names(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
) -> Result<Vec<String>> {
    let attrs = collect_attributes(inner, addr)?;
    Ok(attrs.into_iter().map(|a| a.msg.name).collect())
}

/// Get a specific attribute by name.
pub(crate) fn get_attr(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
    name: &str,
) -> Result<Attribute> {
    let attrs = collect_attributes(inner, addr)?;
    attrs
        .into_iter()
        .find(|a| a.msg.name == name)
        .ok_or_else(|| Error::InvalidFormat(format!("attribute '{name}' not found")))
}
