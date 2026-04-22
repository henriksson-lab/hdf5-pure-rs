use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::filters;
// B-tree types used for chunk reading
use crate::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage, LayoutClass};
use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::fill_value::{FillValueMessage, FILL_TIME_NEVER};
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::format::object_header::{self, ObjectHeader, RawMessage};
use crate::hl::file::FileInner;
use crate::hl::value::H5Value;
use crate::io::reader::HdfReader;

const MAX_VDS_MAPPINGS: usize = 65_536;
const MAX_VDS_SELECTION_RANK: usize = 32;

/// View policy for virtual datasets with unlimited dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdsView {
    LastAvailable,
    FirstMissing,
}

/// An HDF5 dataset.
pub struct Dataset {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    name: String,
    addr: u64,
}

/// Metadata about a dataset parsed from its object header.
#[derive(Debug)]
pub struct DatasetInfo {
    pub dataspace: DataspaceMessage,
    pub datatype: DatatypeMessage,
    pub layout: DataLayoutMessage,
    pub filter_pipeline: Option<FilterPipelineMessage>,
    pub fill_value: Option<FillValueMessage>,
}

#[derive(Debug, Clone)]
struct VirtualMapping {
    file_name: String,
    dataset_name: String,
    source_select: VirtualSelection,
    virtual_select: VirtualSelection,
}

struct VirtualSourceData {
    info: DatasetInfo,
    raw: Vec<u8>,
}

struct VirtualPointMap {
    source_points: Vec<Vec<u64>>,
    virtual_points: Vec<Vec<u64>>,
}

#[derive(Debug, Clone)]
struct ChunkBTreeRecord {
    coords: Vec<u64>,
    chunk_addr: u64,
    chunk_size: u64,
    filter_mask: u32,
}

struct ChunkCopyPlan {
    out_strides: Vec<usize>,
    chunk_strides: Vec<usize>,
    chunk_suffix_products: Vec<usize>,
    total_chunk_elements: usize,
}

struct ChunkReadContext<'a> {
    idx_addr: u64,
    data_dims: &'a [u64],
    chunk_dims: &'a [u64],
    chunk_bytes: usize,
    element_size: usize,
    total_bytes: usize,
}

struct VirtualHyperslabHeader {
    flags: u8,
    enc_size: usize,
}

struct DecodedRegularHyperslabDim {
    start: u64,
    stride: u64,
    count: u64,
    block: u64,
}

struct DecodedIrregularHyperslabBlock {
    start: Vec<u64>,
    end: Vec<u64>,
}

struct DecodedVirtualSourceNames {
    file_name: String,
    dataset_name: String,
}

#[derive(Debug, Clone)]
enum VirtualSelection {
    All,
    Points(Vec<Vec<u64>>),
    Regular(RegularHyperslab),
    Irregular(Vec<IrregularHyperslabBlock>),
}

/// Output of `decode_chunk_btree_node`: either a leaf node's chunk
/// records (coords, addr, size, filter_mask) or an internal node's
/// child-pointer addresses. Mirrors the structural distinction libhdf5
/// makes via the `H5B_t` `level == 0` test.
enum ChunkBTreeNode {
    Leaf(Vec<ChunkBTreeRecord>),
    Internal(Vec<u64>),
}

#[derive(Debug, Clone)]
struct RegularHyperslab {
    start: Vec<u64>,
    stride: Vec<u64>,
    count: Vec<u64>,
    block: Vec<u64>,
}

#[derive(Debug, Clone)]
struct IrregularHyperslabBlock {
    start: Vec<u64>,
    block: Vec<u64>,
}

impl Dataset {
    pub(crate) fn new(
        inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
        name: &str,
        addr: u64,
    ) -> Self {
        Self {
            inner,
            name: name.to_string(),
            addr,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the object header address.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// List attribute names.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        crate::hl::attribute::attr_names(&self.inner, self.addr)
    }

    /// Get an attribute by name.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.addr, name)
    }

    /// Parse the dataset's metadata from its object header.
    pub fn info(&self) -> Result<DatasetInfo> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let sizeof_size = guard.superblock.sizeof_size;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)
    }

    pub(crate) fn parse_info(
        messages: &[RawMessage],
        sizeof_addr: u8,
        sizeof_size: u8,
    ) -> Result<DatasetInfo> {
        let mut dataspace = None;
        let mut datatype = None;
        let mut layout = None;
        let mut filter_pipeline = None;
        let mut fill_value = None;
        let mut old_fill_value_raw = None;
        let mut has_external_file_list = false;

        for msg in messages {
            match msg.msg_type {
                object_header::MSG_DATASPACE => {
                    dataspace = Some(DataspaceMessage::decode(&msg.data)?);
                }
                object_header::MSG_DATATYPE => {
                    datatype = Some(DatatypeMessage::decode(&msg.data)?);
                }
                object_header::MSG_LAYOUT => {
                    layout = Some(DataLayoutMessage::decode(
                        &msg.data,
                        sizeof_addr,
                        sizeof_size,
                    )?);
                }
                object_header::MSG_FILTER_PIPELINE => {
                    filter_pipeline = Some(FilterPipelineMessage::decode(&msg.data)?);
                }
                object_header::MSG_FILL_VALUE => {
                    fill_value = Some(FillValueMessage::decode(&msg.data)?);
                }
                object_header::MSG_FILL_VALUE_OLD => {
                    old_fill_value_raw = Some(msg.data.as_slice());
                }
                object_header::MSG_EXTERNAL_FILE_LIST => {
                    has_external_file_list = true;
                }
                _ => {}
            }
        }

        if has_external_file_list {
            return Err(Error::Unsupported(
                "external raw data storage is not implemented".into(),
            ));
        }

        if fill_value.is_none() {
            if let (Some(raw), Some(datatype)) = (old_fill_value_raw, datatype.as_ref()) {
                fill_value = Some(FillValueMessage::decode_old_with_datatype_size(
                    raw,
                    Some(datatype.size as usize),
                )?);
            }
        }

        Ok(DatasetInfo {
            dataspace: dataspace
                .ok_or_else(|| Error::InvalidFormat("dataset missing dataspace message".into()))?,
            datatype: datatype
                .ok_or_else(|| Error::InvalidFormat("dataset missing datatype message".into()))?,
            layout: layout
                .ok_or_else(|| Error::InvalidFormat("dataset missing layout message".into()))?,
            filter_pipeline,
            fill_value,
        })
    }

    /// Get the shape of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>> {
        self.shape_with_vds_view(VdsView::LastAvailable)
    }

    /// Get the shape of the dataset, overriding the VDS view policy.
    pub fn shape_with_vds_view(&self, view: VdsView) -> Result<Vec<u64>> {
        let info = self.info()?;
        if info.layout.layout_class == LayoutClass::Virtual {
            return self.virtual_shape_with_info(&info, view);
        }
        Ok(info.dataspace.dims)
    }

    fn virtual_shape_with_info(&self, info: &DatasetInfo, view: VdsView) -> Result<Vec<u64>> {
        let mut guard = self.inner.lock();
        let heap_addr = info.layout.virtual_heap_addr.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap address".into())
        })?;
        let heap_index = info.layout.virtual_heap_index.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap index".into())
        })?;
        let path = guard.path.clone();
        let heap_data = crate::format::global_heap::read_global_heap_object(
            &mut guard.reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: heap_addr,
                object_index: heap_index,
            },
        )?;
        let sizeof_size = guard.reader.sizeof_size() as usize;
        drop(guard);

        let mappings = Self::decode_virtual_mappings(&heap_data, sizeof_size)?;
        Self::virtual_output_dims(&mappings, path.as_deref(), info, view)
    }

    /// Get the total number of elements.
    pub fn size(&self) -> Result<u64> {
        self.size_with_vds_view(VdsView::LastAvailable)
    }

    /// Get the total number of elements, overriding the VDS view policy.
    pub fn size_with_vds_view(&self, view: VdsView) -> Result<u64> {
        let info = self.info()?;
        if info.layout.layout_class == LayoutClass::Virtual {
            let shape = self.virtual_shape_with_info(&info, view)?;
            return Self::dataspace_element_count(info.dataspace.space_type, &shape);
        }
        Self::dataspace_element_count(info.dataspace.space_type, &info.dataspace.dims)
    }

    /// Get the element size in bytes.
    pub fn element_size(&self) -> Result<usize> {
        let info = self.info()?;
        Ok(info.datatype.size as usize)
    }

    /// Get the datatype.
    pub fn dtype(&self) -> Result<crate::hl::datatype::Datatype> {
        let info = self.info()?;
        Ok(crate::hl::datatype::Datatype::from_message(info.datatype))
    }

    /// Return the parsed low-level datatype message.
    pub fn raw_datatype_message(&self) -> Result<DatatypeMessage> {
        Ok(self.info()?.datatype)
    }

    /// Get the dataspace.
    pub fn space(&self) -> Result<crate::hl::dataspace::Dataspace> {
        let info = self.info()?;
        Ok(crate::hl::dataspace::Dataspace::from_message(
            info.dataspace,
        ))
    }

    /// Return the parsed low-level dataspace message.
    pub fn raw_dataspace_message(&self) -> Result<DataspaceMessage> {
        Ok(self.info()?.dataspace)
    }

    /// Whether the dataset uses chunked storage.
    pub fn is_chunked(&self) -> Result<bool> {
        let info = self.info()?;
        Ok(info.layout.layout_class == LayoutClass::Chunked)
    }

    /// Whether this is a virtual dataset.
    pub fn is_virtual(&self) -> Result<bool> {
        let info = self.info()?;
        Ok(info.layout.layout_class == LayoutClass::Virtual)
    }

    /// Whether the dataset is resizable (has unlimited dimensions).
    pub fn is_resizable(&self) -> Result<bool> {
        Ok(self.space()?.is_resizable())
    }

    /// Get the storage layout type.
    pub fn layout(&self) -> Result<LayoutClass> {
        let info = self.info()?;
        Ok(info.layout.layout_class)
    }

    /// Get the chunk dimensions (None if not chunked).
    pub fn chunk(&self) -> Result<Option<Vec<u64>>> {
        let info = self.info()?;
        Ok(info.layout.chunk_dims.clone())
    }

    /// Get the filter pipeline (empty if no filters).
    pub fn filters(&self) -> Result<Vec<crate::format::messages::filter_pipeline::FilterDesc>> {
        let info = self.info()?;
        Ok(info.filter_pipeline.map(|p| p.filters).unwrap_or_default())
    }

    /// Get the dataset creation properties.
    pub fn create_plist(&self) -> Result<crate::hl::plist::dataset_create::DatasetCreate> {
        crate::hl::plist::dataset_create::DatasetCreate::from_dataset(self)
    }

    /// Read fixed-length strings from the dataset.
    /// Each element is `element_size` bytes, null-padded or space-padded.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        let info = self.info()?;
        let elem_size = info.datatype.size as usize;
        let raw = self.read_raw()?;

        if info.datatype.is_variable_length() {
            // Variable-length data: each element is stored as:
            // sequence_length(4) + global_heap_collection_addr(sizeof_addr) + heap_object_index(4)
            let mut guard = self.inner.lock();
            let sizeof_addr = guard.superblock.sizeof_addr as usize;
            let ref_size = 4 + sizeof_addr + 4; // seq_len + addr + index
            let mut strings = Vec::new();

            for chunk in raw.chunks_exact(ref_size) {
                let _seq_len = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let mut addr = 0u64;
                for i in 0..sizeof_addr {
                    addr |= (chunk[4 + i] as u64) << (i * 8);
                }
                let index = u32::from_le_bytes([
                    chunk[4 + sizeof_addr],
                    chunk[4 + sizeof_addr + 1],
                    chunk[4 + sizeof_addr + 2],
                    chunk[4 + sizeof_addr + 3],
                ]);

                if addr == 0 || crate::io::reader::is_undef_addr(addr) {
                    strings.push(String::new());
                } else {
                    let gh_ref = crate::format::global_heap::GlobalHeapRef {
                        collection_addr: addr,
                        object_index: index,
                    };
                    match crate::format::global_heap::read_global_heap_object(
                        &mut guard.reader,
                        &gh_ref,
                    ) {
                        Ok(data) => {
                            trace_vlen_read(data.len() as u64, &data);
                            let s = String::from_utf8_lossy(&data).to_string();
                            // Trim trailing nulls
                            strings.push(s.trim_end_matches('\0').to_string());
                        }
                        Err(_) => strings.push(String::new()),
                    }
                }
            }
            return Ok(strings);
        }

        // Fixed-length strings
        let padding = info.datatype.string_padding().unwrap_or(1);
        let mut strings = Vec::new();
        for chunk in raw.chunks_exact(elem_size) {
            strings.push(decode_fixed_string_with_padding(chunk, padding));
        }
        Ok(strings)
    }

    /// Read a single string (for scalar string datasets/attributes).
    pub fn read_string(&self) -> Result<String> {
        let strings = self.read_strings()?;
        strings
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidFormat("no string data".into()))
    }

    /// Read compound type field info. Returns field names, offsets, and sizes.
    pub fn compound_fields(&self) -> Result<Vec<crate::format::messages::datatype::CompoundField>> {
        let info = self.info()?;
        info.datatype.compound_fields()
    }

    /// Read a single field from a compound dataset as typed values.
    /// Example: `ds.read_field::<f64>("x")` reads the "x" field from all records.
    pub fn read_field<T: crate::hl::types::H5Type>(&self, field_name: &str) -> Result<Vec<T>> {
        let fields = self.compound_fields()?;
        let field = fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::InvalidFormat(format!("field '{field_name}' not found")))?;

        if field.size != T::type_size() {
            return Err(Error::InvalidFormat(format!(
                "field '{}' has size {} but requested type has size {}",
                field_name,
                field.size,
                T::type_size()
            )));
        }

        let mut raw = self.read_raw()?;
        self.maybe_byte_swap_field(&mut raw, field)?;

        let info = self.info()?;
        let record_size = info.datatype.size as usize;
        let offset = field.byte_offset;
        let elem_size = field.size;
        let n_records = raw.len() / record_size;

        let mut result = Vec::with_capacity(n_records);
        for i in 0..n_records {
            let start = i * record_size + offset;
            let bytes = &raw[start..start + elem_size];
            // Copy to aligned buffer
            let val = unsafe {
                let mut v = std::mem::MaybeUninit::<T>::uninit();
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, elem_size);
                v.assume_init()
            };
            result.push(val);
        }
        Ok(result)
    }

    /// Read a single compound field as raw per-record byte slices.
    ///
    /// This is useful for compound members whose HDF5 datatype is not directly
    /// representable as a primitive Rust `H5Type`, such as nested compound,
    /// array, variable-length, or reference members. No recursive typed
    /// conversion is performed; callers must interpret each returned byte
    /// vector using the field datatype from [`Dataset::compound_fields`].
    pub fn read_field_raw(&self, field_name: &str) -> Result<Vec<Vec<u8>>> {
        let fields = self.compound_fields()?;
        let field = fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::InvalidFormat(format!("field '{field_name}' not found")))?;

        let raw = self.read_raw()?;
        let info = self.info()?;
        let record_size = info.datatype.size as usize;
        let offset = field.byte_offset;
        let elem_size = field.size;
        let n_records = raw.len() / record_size;

        let mut result = Vec::with_capacity(n_records);
        for i in 0..n_records {
            let start = i * record_size + offset;
            let end = start + elem_size;
            if end > raw.len() {
                return Err(Error::InvalidFormat(format!(
                    "compound field '{field_name}' exceeds record bounds"
                )));
            }
            result.push(raw[start..end].to_vec());
        }

        Ok(result)
    }

    /// Read a compound field as recursively decoded high-level values.
    ///
    /// This handles nested compound, array, variable-length, and reference
    /// members. Datatype classes without a richer public representation are
    /// returned as `H5Value::Raw`. This API is intended for inspection and
    /// simple extraction, not full libhdf5 typed conversion parity.
    pub fn read_field_values(&self, field_name: &str) -> Result<Vec<H5Value>> {
        let fields = self.compound_fields()?;
        let field = fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::InvalidFormat(format!("field '{field_name}' not found")))?;

        let raw = self.read_raw()?;
        let info = self.info()?;
        let record_size = info.datatype.size as usize;
        if record_size == 0 || field.byte_offset + field.size > record_size {
            return Err(Error::InvalidFormat(format!(
                "compound field '{field_name}' exceeds record bounds"
            )));
        }

        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr as usize;
        let n_records = raw.len() / record_size;
        let mut result = Vec::with_capacity(n_records);

        for record in raw.chunks_exact(record_size) {
            let bytes = &record[field.byte_offset..field.byte_offset + field.size];
            result.push(Self::decode_value(
                &field.datatype,
                bytes,
                sizeof_addr,
                &mut guard.reader,
            )?);
        }

        Ok(result)
    }

    fn decode_value<R: Read + Seek>(
        dtype: &DatatypeMessage,
        bytes: &[u8],
        sizeof_addr: usize,
        reader: &mut HdfReader<R>,
    ) -> Result<H5Value> {
        use crate::format::messages::datatype::{ByteOrder, DatatypeClass};

        match dtype.class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField => {
                let le = matches!(dtype.byte_order(), Some(ByteOrder::LittleEndian) | None);
                if dtype.is_signed().unwrap_or(false) {
                    Ok(H5Value::Int(read_signed_int(bytes, le)))
                } else {
                    Ok(H5Value::UInt(read_unsigned_int(bytes, le)))
                }
            }
            DatatypeClass::FloatingPoint => match dtype.size {
                4 => {
                    let arr = endian_array::<4>(bytes, dtype.byte_order())?;
                    Ok(H5Value::Float(f32::from_le_bytes(arr) as f64))
                }
                8 => {
                    let arr = endian_array::<8>(bytes, dtype.byte_order())?;
                    Ok(H5Value::Float(f64::from_le_bytes(arr)))
                }
                _ => Ok(H5Value::Raw(bytes.to_vec())),
            },
            DatatypeClass::String => Ok(H5Value::String(decode_fixed_string(bytes))),
            DatatypeClass::Compound => {
                let fields = dtype.compound_fields()?;
                let mut values = Vec::with_capacity(fields.len());
                for field in fields {
                    let end = field.byte_offset.checked_add(field.size).ok_or_else(|| {
                        Error::InvalidFormat("nested compound field offset overflow".into())
                    })?;
                    if end > bytes.len() {
                        return Err(Error::InvalidFormat(format!(
                            "nested compound field '{}' exceeds record bounds",
                            field.name
                        )));
                    }
                    values.push((
                        field.name.clone(),
                        Self::decode_value(
                            &field.datatype,
                            &bytes[field.byte_offset..end],
                            sizeof_addr,
                            reader,
                        )?,
                    ));
                }
                Ok(H5Value::Compound(values))
            }
            DatatypeClass::Array => {
                let (dims, base) = dtype.array_dims_base()?;
                let count = dims.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(dim as usize)
                        .ok_or_else(|| Error::InvalidFormat("array element count overflow".into()))
                })?;
                let elem_size = base.size as usize;
                if elem_size == 0 || bytes.len() < count.saturating_mul(elem_size) {
                    return Err(Error::InvalidFormat("array field payload too short".into()));
                }
                let mut values = Vec::with_capacity(count);
                for chunk in bytes[..count * elem_size].chunks_exact(elem_size) {
                    values.push(Self::decode_value(&base, chunk, sizeof_addr, reader)?);
                }
                Ok(H5Value::Array(values))
            }
            DatatypeClass::VarLen => {
                let base = dtype.vlen_base()?;
                Self::decode_vlen_value(base.as_ref(), bytes, sizeof_addr, reader)
            }
            DatatypeClass::Reference => {
                let n = bytes.len().min(sizeof_addr).min(8);
                let mut addr = 0u64;
                for (i, byte) in bytes.iter().take(n).enumerate() {
                    addr |= (*byte as u64) << (i * 8);
                }
                Ok(H5Value::Reference(addr))
            }
            DatatypeClass::Enum | DatatypeClass::Opaque | DatatypeClass::Time => {
                Ok(H5Value::Raw(bytes.to_vec()))
            }
        }
    }

    fn decode_vlen_value<R: Read + Seek>(
        base: Option<&DatatypeMessage>,
        bytes: &[u8],
        sizeof_addr: usize,
        reader: &mut HdfReader<R>,
    ) -> Result<H5Value> {
        if bytes.len() < 4 + sizeof_addr + 4 {
            return Err(Error::InvalidFormat(
                "variable-length descriptor too short".into(),
            ));
        }

        let seq_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut addr = 0u64;
        for i in 0..sizeof_addr.min(8) {
            addr |= (bytes[4 + i] as u64) << (i * 8);
        }
        let index_pos = 4 + sizeof_addr;
        let index = u32::from_le_bytes([
            bytes[index_pos],
            bytes[index_pos + 1],
            bytes[index_pos + 2],
            bytes[index_pos + 3],
        ]);

        if seq_len == 0 || addr == 0 || crate::io::reader::is_undef_addr(addr) {
            return Ok(H5Value::VarLen(Vec::new()));
        }

        let data = crate::format::global_heap::read_global_heap_object(
            reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: addr,
                object_index: index,
            },
        )?;
        let Some(base) = base else {
            trace_vlen_read(seq_len as u64, &data[..data.len().min(seq_len)]);
            return Ok(H5Value::Raw(data[..data.len().min(seq_len)].to_vec()));
        };

        if base.class == crate::format::messages::datatype::DatatypeClass::String {
            if data.len() < seq_len {
                return Err(Error::InvalidFormat(format!(
                    "variable-length string payload too short: expected {seq_len} bytes, got {}",
                    data.len()
                )));
            }
            let data = &data[..seq_len];
            trace_vlen_read(seq_len as u64, data);
            return Ok(H5Value::String(
                String::from_utf8_lossy(&data)
                    .trim_end_matches('\0')
                    .to_string(),
            ));
        }

        let elem_size = base.size as usize;
        if elem_size == 0 {
            let data = &data[..data.len().min(seq_len)];
            trace_vlen_read(seq_len as u64, data);
            return Ok(H5Value::Raw(data.to_vec()));
        }
        let expected_len = seq_len
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("variable-length payload size overflow".into()))?;
        if data.len() < expected_len {
            return Err(Error::InvalidFormat(format!(
                "variable-length payload too short: expected {expected_len} bytes, got {}",
                data.len()
            )));
        }
        let data = &data[..expected_len];
        trace_vlen_read(expected_len as u64, data);

        let mut values = Vec::with_capacity(seq_len);
        for chunk in data.chunks_exact(elem_size) {
            values.push(Self::decode_value(base, chunk, sizeof_addr, reader)?);
        }

        Ok(H5Value::VarLen(values))
    }

    /// Byte-swap a specific compound field in the raw data buffer.
    fn maybe_byte_swap_field(
        &self,
        data: &mut [u8],
        field: &crate::format::messages::datatype::CompoundField,
    ) -> Result<()> {
        use crate::format::messages::datatype::{ByteOrder, DatatypeClass};

        if field.size <= 1 {
            return Ok(());
        }

        match field.class {
            DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField => {}
            _ => return Ok(()),
        }

        let need_swap = match field.byte_order {
            Some(ByteOrder::BigEndian) => cfg!(target_endian = "little"),
            Some(ByteOrder::LittleEndian) => cfg!(target_endian = "big"),
            None => false,
        };

        if !need_swap {
            return Ok(());
        }

        let info = self.info()?;
        let record_size = info.datatype.size as usize;
        if record_size == 0 || field.byte_offset + field.size > record_size {
            return Err(Error::InvalidFormat(format!(
                "compound field '{}' exceeds record bounds",
                field.name
            )));
        }

        for record in data.chunks_exact_mut(record_size) {
            record[field.byte_offset..field.byte_offset + field.size].reverse();
        }

        Ok(())
    }

    /// Read all raw data bytes from the dataset.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.read_raw_with_vds_view(VdsView::LastAvailable)
    }

    /// Read all raw bytes, overriding the VDS view policy.
    pub fn read_raw_with_vds_view(&self, view: VdsView) -> Result<Vec<u8>> {
        let info = {
            let mut guard = self.inner.lock();
            let sizeof_addr = guard.superblock.sizeof_addr;
            let sizeof_size = guard.superblock.sizeof_size;
            let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;
            Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)?
        };

        self.read_raw_with_info(info, view)
    }

    pub(crate) fn read_raw_with_info(&self, info: DatasetInfo, view: VdsView) -> Result<Vec<u8>> {
        let element_size = info.datatype.size as usize;
        if element_size == 0 {
            return Err(Error::InvalidFormat("zero-sized datatype".into()));
        }
        let total_elements =
            Self::dataspace_element_count(info.dataspace.space_type, &info.dataspace.dims)?;
        let total_elements_usize = usize_from_u64(total_elements, "dimension product")?;
        let total_bytes = total_elements_usize
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("total data size overflow".into()))?;

        // Sanity limit: refuse to allocate more than 4GB in a single read
        const MAX_READ_BYTES: usize = 4 * 1024 * 1024 * 1024;
        if total_bytes > MAX_READ_BYTES {
            return Err(Error::InvalidFormat(format!(
                "dataset too large for single read: {total_bytes} bytes (max {MAX_READ_BYTES})"
            )));
        }

        let mut guard = self.inner.lock();
        match info.layout.layout_class {
            LayoutClass::Compact => {
                let data = info
                    .layout
                    .compact_data
                    .ok_or_else(|| Error::InvalidFormat("compact dataset missing data".into()))?;
                if data.len() < total_bytes {
                    return Err(Error::InvalidFormat(format!(
                        "compact dataset data size {} is smaller than expected {total_bytes}",
                        data.len()
                    )));
                }
                Ok(data[..total_bytes].to_vec())
            }
            LayoutClass::Contiguous => {
                let addr = info.layout.contiguous_addr.ok_or_else(|| {
                    Error::InvalidFormat("contiguous dataset missing address".into())
                })?;
                let size = usize_from_u64(
                    info.layout.contiguous_size.unwrap_or(total_bytes as u64),
                    "contiguous dataset size",
                )?;

                if crate::io::reader::is_undef_addr(addr) {
                    return Self::filled_data(total_elements_usize, element_size, &info);
                }

                guard.reader.seek(addr)?;
                let data = guard.reader.read_bytes(size)?;
                Ok(data)
            }
            LayoutClass::Chunked => Self::read_chunked(&mut guard.reader, &info, total_bytes),
            LayoutClass::Virtual => {
                let heap_addr = info.layout.virtual_heap_addr.ok_or_else(|| {
                    Error::InvalidFormat("virtual dataset missing global heap address".into())
                })?;
                let heap_index = info.layout.virtual_heap_index.ok_or_else(|| {
                    Error::InvalidFormat("virtual dataset missing global heap index".into())
                })?;
                let path = guard.path.clone();
                let heap_data = crate::format::global_heap::read_global_heap_object(
                    &mut guard.reader,
                    &crate::format::global_heap::GlobalHeapRef {
                        collection_addr: heap_addr,
                        object_index: heap_index,
                    },
                )?;
                let sizeof_size = guard.reader.sizeof_size() as usize;
                drop(guard);
                Self::read_virtual_dataset(&heap_data, sizeof_size, path.as_deref(), &info, view)
            }
        }
    }

    fn filled_data(
        total_elements: usize,
        element_size: usize,
        info: &DatasetInfo,
    ) -> Result<Vec<u8>> {
        let total_bytes = total_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("fill buffer size overflow".into()))?;
        let Some(fill) = &info.fill_value else {
            return Ok(vec![0u8; total_bytes]);
        };
        if fill.fill_time == FILL_TIME_NEVER {
            return Ok(vec![0u8; total_bytes]);
        }
        let Some(value) = fill.value.as_deref() else {
            return Ok(vec![0u8; total_bytes]);
        };
        if value.len() != element_size {
            return Err(Error::Unsupported(format!(
                "fill value size {} does not match element size {}",
                value.len(),
                element_size
            )));
        }

        let mut out = vec![0u8; total_bytes];
        for chunk in out.chunks_exact_mut(element_size) {
            chunk.copy_from_slice(value);
        }
        Ok(out)
    }

    /// Read all data as a typed Vec.
    ///
    /// This uses the crate's supported conversion table, not the full libhdf5
    /// conversion matrix. Unsupported or lossy HDF5 datatype conversions return
    /// an error rather than attempting C-library parity.
    pub fn read<T: crate::hl::types::H5Type>(&self) -> Result<Vec<T>> {
        self.read_with_vds_view(VdsView::LastAvailable)
    }

    /// Read all data as a typed Vec, overriding the VDS view policy.
    pub fn read_with_vds_view<T: crate::hl::types::H5Type>(&self, view: VdsView) -> Result<Vec<T>> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let raw = self.read_raw_with_vds_view(view)?;
        conversion.bytes_to_vec(raw)
    }

    /// Read a scalar value.
    pub fn read_scalar<T: crate::hl::types::H5Type>(&self) -> Result<T> {
        self.read_scalar_with_vds_view(VdsView::LastAvailable)
    }

    /// Read a scalar value, overriding the VDS view policy.
    pub fn read_scalar_with_vds_view<T: crate::hl::types::H5Type>(
        &self,
        view: VdsView,
    ) -> Result<T> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let raw = self.read_raw_with_vds_view(view)?;
        conversion.bytes_to_scalar(raw)
    }

    fn dataspace_element_count(space_type: DataspaceType, dims: &[u64]) -> Result<u64> {
        if space_type == DataspaceType::Null {
            return Ok(0);
        }
        if dims.is_empty() {
            return Ok(1);
        }
        dims.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("dimension product overflow".into()))
        })
    }

    /// Read data as a 1D ndarray.
    pub fn read_1d<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::Array1<T>> {
        let vec = self.read::<T>()?;
        Ok(ndarray::Array1::from_vec(vec))
    }

    /// Read data as a 2D ndarray (row-major).
    pub fn read_2d<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::Array2<T>> {
        let shape = self.shape()?;
        if shape.len() != 2 {
            return Err(Error::InvalidFormat(format!(
                "expected 2D dataset, got {}D",
                shape.len()
            )));
        }
        let vec = self.read::<T>()?;
        ndarray::Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), vec)
            .map_err(|e| Error::Other(format!("ndarray shape error: {e}")))
    }

    /// Read data as an N-dimensional ndarray (row-major).
    pub fn read_dyn<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::ArrayD<T>> {
        let shape = self.shape()?;
        let dims: Vec<usize> = shape
            .iter()
            .map(|&dim| usize_from_u64(dim, "ndarray dimension"))
            .collect::<Result<Vec<_>>>()?;
        let vec = self.read::<T>()?;
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), vec)
            .map_err(|e| Error::Other(format!("ndarray shape error: {e}")))
    }

    /// Read a subset of the dataset using a selection.
    ///
    /// Example: `ds.read_slice::<f64>(10..20)` reads elements 10-19 from a 1D dataset.
    pub fn read_slice<T: crate::hl::types::H5Type, S: crate::hl::selection::IntoSelection>(
        &self,
        sel: S,
    ) -> Result<Vec<T>> {
        let shape = self.shape()?;
        let selection = sel.into_selection(&shape);
        let slices = selection.to_slices(&shape);
        let out_shape = selection.output_shape(&shape);
        let total_out = Self::selection_output_elements(&out_shape)?;
        let total_out_usize = usize_from_u64(total_out, "selection element count")?;
        let elem_size = T::type_size();

        if let Some(raw) = self.try_read_slice_contiguous_1d(&shape, &slices, elem_size)? {
            return crate::hl::types::bytes_to_vec(raw);
        }

        let all_data = self.read::<T>()?;

        if shape.len() == 1 && slices.len() == 1 {
            return Self::extract_1d_selection(&all_data, &slices[0]);
        }

        Self::extract_nd_selection(
            &all_data,
            &shape,
            &slices,
            &out_shape,
            total_out,
            total_out_usize,
        )
    }

    fn selection_output_elements(out_shape: &[u64]) -> Result<u64> {
        Ok(if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().try_fold(1u64, |acc, &dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| Error::InvalidFormat("selection element count overflow".into()))
            })?
        }
        .max(1))
    }

    fn try_read_slice_contiguous_1d(
        &self,
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        elem_size: usize,
    ) -> Result<Option<Vec<u8>>> {
        if !(shape.len() == 1 && slices.len() == 1 && slices[0].step == 1) {
            return Ok(None);
        }

        let info = self.info()?;
        if info.layout.layout_class != LayoutClass::Contiguous {
            return Ok(None);
        }

        let Some(addr) = info.layout.contiguous_addr else {
            return Ok(None);
        };
        if crate::io::reader::is_undef_addr(addr) {
            return Ok(None);
        }

        let start_byte = usize_from_u64(slices[0].start, "selection start")?
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("selection byte offset overflow".into()))?;
        let nbytes = usize_from_u64(slices[0].count(), "selection count")?
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("selection byte count overflow".into()))?;
        let read_addr = addr
            .checked_add(start_byte as u64)
            .ok_or_else(|| Error::InvalidFormat("selection read address overflow".into()))?;

        let mut guard = self.inner.lock();
        guard.reader.seek(read_addr)?;
        Ok(Some(guard.reader.read_bytes(nbytes)?))
    }

    fn extract_1d_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        slice: &crate::hl::selection::SliceInfo,
    ) -> Result<Vec<T>> {
        let start = usize_from_u64(slice.start, "selection start")?;
        let end = usize_from_u64(slice.end, "selection end")?;
        if start > all_data.len() {
            return Ok(Vec::new());
        }
        Ok(all_data[start..end.min(all_data.len())].to_vec())
    }

    fn extract_nd_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        out_shape: &[u64],
        total_out: u64,
        total_out_usize: usize,
    ) -> Result<Vec<T>> {
        let mut result = Vec::with_capacity(total_out_usize);
        let ndims = shape.len();

        let mut in_strides = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            in_strides[d] = in_strides[d + 1]
                .checked_mul(usize_from_u64(shape[d + 1], "selection shape")?)
                .ok_or_else(|| Error::InvalidFormat("selection stride overflow".into()))?;
        }

        let mut out_idx = vec![0u64; ndims];
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for d in 0..ndims {
                let in_d = slices[d].start + out_idx[d] * slices[d].step;
                let term = usize_from_u64(in_d, "selection input index")?
                    .checked_mul(in_strides[d])
                    .ok_or_else(|| {
                        Error::InvalidFormat("selection linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("selection linear index overflow".into())
                })?;
            }

            if in_linear < all_data.len() {
                result.push(all_data[in_linear]);
            }

            for d in (0..ndims).rev() {
                out_idx[d] += 1;
                if out_idx[d] < out_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        Ok(result)
    }

    fn read_chunked<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let element_size = info.datatype.size as usize;
        let data_dims = &info.dataspace.dims;
        let chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing chunk dims".into()))?;
        let chunk_data_dims = Self::chunk_data_dims(data_dims, chunk_dims)?;
        let chunk_bytes = Self::chunk_byte_len(chunk_dims, chunk_data_dims, element_size)?;
        let idx_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing index address".into()))?;

        if crate::io::reader::is_undef_addr(idx_addr) {
            return Self::filled_data(total_bytes / element_size, element_size, info);
        }

        let chunk_ctx = ChunkReadContext {
            idx_addr,
            data_dims,
            chunk_dims: chunk_data_dims,
            chunk_bytes,
            element_size,
            total_bytes,
        };

        Self::read_chunked_with_index(reader, info, &chunk_ctx)
    }

    fn read_chunked_with_index<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        match info.layout.chunk_index_type.clone() {
            Some(ChunkIndexType::SingleChunk) => Self::read_single_chunk(
                reader,
                chunk_ctx.idx_addr,
                info,
                chunk_ctx.chunk_bytes,
                chunk_ctx.element_size,
                chunk_ctx.total_bytes,
            ),
            Some(ChunkIndexType::BTreeV1) => Self::read_chunked_btree_v1(reader, info, chunk_ctx),
            Some(ChunkIndexType::Implicit) => Self::read_chunked_implicit(reader, info, chunk_ctx),
            Some(ChunkIndexType::FixedArray) => {
                Self::read_chunked_fixed_array(reader, info, chunk_ctx)
            }
            Some(ChunkIndexType::ExtensibleArray) => {
                Self::read_chunked_extensible_array(reader, info, chunk_ctx)
            }
            Some(ChunkIndexType::BTreeV2) => Self::read_chunked_btree_v2(reader, info, chunk_ctx),
            None if info.layout.version <= 3 => {
                Self::read_chunked_btree_v1(reader, info, chunk_ctx)
            }
            None => Err(Error::InvalidFormat(
                "chunked dataset missing chunk index type".into(),
            )),
        }
    }

    fn chunk_data_dims<'a>(data_dims: &[u64], chunk_dims: &'a [u64]) -> Result<&'a [u64]> {
        let ndims = data_dims.len();
        if chunk_dims.len() == ndims + 1 {
            Ok(&chunk_dims[..ndims])
        } else if chunk_dims.len() == ndims {
            Ok(chunk_dims)
        } else {
            Err(Error::InvalidFormat(format!(
                "chunk dims rank {} does not match dataspace rank {}",
                chunk_dims.len(),
                ndims
            )))
        }
    }

    fn chunk_byte_len(
        chunk_dims: &[u64],
        chunk_data_dims: &[u64],
        element_size: usize,
    ) -> Result<usize> {
        if chunk_dims.len() == chunk_data_dims.len() + 1 {
            let bytes = chunk_dims
                .iter()
                .copied()
                .try_fold(1u64, |a, b| a.checked_mul(b))
                .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?;
            return usize_from_u64(bytes, "chunk byte size");
        }

        let chunk_elements: u64 = chunk_data_dims
            .iter()
            .copied()
            .try_fold(1u64, |a, b| a.checked_mul(b))
            .ok_or_else(|| Error::InvalidFormat("chunk dimension product overflow".into()))?;
        usize_from_u64(chunk_elements, "chunk element count")?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))
    }

    fn read_single_chunk<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        chunk_addr: u64,
        info: &DatasetInfo,
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        reader.seek(chunk_addr)?;
        let read_size = usize_from_u64(
            info.layout
                .single_chunk_filtered_size
                .unwrap_or(chunk_bytes as u64),
            "single-chunk size",
        )?;
        let mut raw = reader.read_bytes(read_size)?;

        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                raw = filters::apply_pipeline_reverse_with_mask_expected(
                    &raw,
                    pipeline,
                    element_size,
                    info.layout.single_chunk_filter_mask.unwrap_or(0),
                    chunk_bytes,
                )?;
            }
        }

        if raw.len() < total_bytes {
            return Err(Error::InvalidFormat(
                "single-chunk data shorter than dataset size".into(),
            ));
        }
        Ok(raw[..total_bytes].to_vec())
    }

    fn read_chunked_btree_v1<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let ndims = chunk_ctx.data_dims.len();
        let mut output = Self::filled_data(
            chunk_ctx.total_bytes / chunk_ctx.element_size,
            chunk_ctx.element_size,
            info,
        )?;
        let chunk_records = Self::collect_btree_v1_chunks(reader, chunk_ctx.idx_addr, ndims)?;

        for chunk_record in &chunk_records {
            Self::process_btree_v1_chunk_record(
                reader,
                chunk_ctx.idx_addr,
                chunk_record,
                info,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.chunk_bytes,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    fn read_virtual_dataset(
        heap_data: &[u8],
        sizeof_size: usize,
        file_path: Option<&Path>,
        info: &DatasetInfo,
        view: VdsView,
    ) -> Result<Vec<u8>> {
        let mappings = Self::decode_virtual_mappings(heap_data, sizeof_size)?;
        let element_size = info.datatype.size as usize;

        let output_dims = Self::virtual_output_dims(&mappings, file_path, info, view)?;
        let total_elements = usize_from_u64(
            Self::dataspace_element_count(info.dataspace.space_type, &output_dims)?,
            "virtual dataset element count",
        )?;
        let mut output = Self::filled_data(total_elements, element_size, info)?;
        let virtual_strides = Self::row_major_strides(&output_dims)?;

        for mapping in mappings {
            let source = Self::open_virtual_source_dataset(file_path, &mapping, info)?;
            let point_map = Self::materialize_virtual_point_map(
                &mapping,
                &source.info.dataspace.dims,
                &output_dims,
            )?;
            Self::copy_virtual_mapping(
                &source.raw,
                &source.info.dataspace.dims,
                &virtual_strides,
                &point_map,
                element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    fn open_virtual_source_dataset(
        file_path: Option<&Path>,
        mapping: &VirtualMapping,
        dest_info: &DatasetInfo,
    ) -> Result<VirtualSourceData> {
        let source_file = Self::resolve_virtual_source_path(file_path, &mapping.file_name)?;
        let source = crate::hl::file::File::open(&source_file)?;
        let source_ds = source.dataset(&mapping.dataset_name)?;
        let source_info = source_ds.info()?;
        if source_info.datatype.size != dest_info.datatype.size {
            return Err(Error::Unsupported(
                "virtual dataset source datatype size does not match destination".into(),
            ));
        }
        let source_raw = source_ds.read_raw()?;
        Ok(VirtualSourceData {
            info: source_info,
            raw: source_raw,
        })
    }

    fn materialize_virtual_point_map(
        mapping: &VirtualMapping,
        source_dims: &[u64],
        output_dims: &[u64],
    ) -> Result<VirtualPointMap> {
        let source_points =
            Self::materialize_virtual_selection_points(&mapping.source_select, source_dims)?;
        let virtual_points =
            Self::materialize_virtual_selection_points(&mapping.virtual_select, output_dims)?;
        if source_points.len() != virtual_points.len() {
            return Err(Error::InvalidFormat(
                "virtual dataset source and destination selections differ in size".into(),
            ));
        }
        Ok(VirtualPointMap {
            source_points,
            virtual_points,
        })
    }

    fn copy_virtual_mapping(
        source_raw: &[u8],
        source_dims: &[u64],
        virtual_strides: &[usize],
        point_map: &VirtualPointMap,
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let source_strides = Self::row_major_strides(source_dims)?;
        for (src, dst) in point_map
            .source_points
            .iter()
            .zip(point_map.virtual_points.iter())
        {
            let src_index = Self::linear_index(src, &source_strides)?;
            let dst_index = Self::linear_index(dst, virtual_strides)?;
            let src_start = src_index.checked_mul(element_size).ok_or_else(|| {
                Error::InvalidFormat("virtual source byte offset overflow".into())
            })?;
            let dst_start = dst_index.checked_mul(element_size).ok_or_else(|| {
                Error::InvalidFormat("virtual destination byte offset overflow".into())
            })?;
            if src_start
                .checked_add(element_size)
                .is_some_and(|end| end <= source_raw.len())
                && dst_start
                    .checked_add(element_size)
                    .is_some_and(|end| end <= output.len())
            {
                output[dst_start..dst_start + element_size]
                    .copy_from_slice(&source_raw[src_start..src_start + element_size]);
            }
        }
        Ok(())
    }

    fn virtual_output_dims(
        mappings: &[VirtualMapping],
        file_path: Option<&Path>,
        info: &DatasetInfo,
        view: VdsView,
    ) -> Result<Vec<u64>> {
        let mut output_dims = info.dataspace.dims.clone();
        let mut unlimited_extents: Vec<Option<(u64, u64)>> = vec![None; output_dims.len()];
        for mapping in mappings {
            let source_file = Self::resolve_virtual_source_path(file_path, &mapping.file_name)?;
            let source = crate::hl::file::File::open(&source_file)?;
            let source_info = source.dataset(&mapping.dataset_name)?.info()?;
            for dim in 0..output_dims.len() {
                if output_dims[dim] != 0 {
                    continue;
                }
                let extent = Self::virtual_mapping_output_extent(
                    &mapping.source_select,
                    &mapping.virtual_select,
                    &source_info.dataspace.dims,
                    dim,
                )?;
                if Self::is_unlimited_vds_dim(info, dim) {
                    let entry = &mut unlimited_extents[dim];
                    match entry {
                        Some((min_extent, max_extent)) => {
                            *min_extent = (*min_extent).min(extent);
                            *max_extent = (*max_extent).max(extent);
                        }
                        None => *entry = Some((extent, extent)),
                    }
                } else {
                    output_dims[dim] = output_dims[dim].max(extent);
                }
            }
        }
        for (dim, extents) in unlimited_extents.into_iter().enumerate() {
            if output_dims[dim] != 0 {
                continue;
            }
            if let Some((min_extent, max_extent)) = extents {
                output_dims[dim] = match view {
                    VdsView::LastAvailable => max_extent,
                    VdsView::FirstMissing => min_extent,
                };
            }
        }
        Ok(output_dims)
    }

    fn is_unlimited_vds_dim(info: &DatasetInfo, dim: usize) -> bool {
        info.dataspace
            .max_dims
            .as_ref()
            .and_then(|max_dims| max_dims.get(dim))
            .copied()
            == Some(u64::MAX)
    }

    fn virtual_mapping_output_extent(
        source_select: &VirtualSelection,
        virtual_select: &VirtualSelection,
        source_dims: &[u64],
        dim: usize,
    ) -> Result<u64> {
        match virtual_select {
            VirtualSelection::All => Ok(source_dims[dim]),
            VirtualSelection::Points(points) => {
                Ok(points.iter().map(|point| point[dim] + 1).max().unwrap_or(0))
            }
            VirtualSelection::Regular(_) => Ok(Self::virtual_selection_start(virtual_select, dim)
                + Self::virtual_selection_span(source_select, source_dims, dim)),
            VirtualSelection::Irregular(blocks) => Ok(blocks
                .iter()
                .map(|block| block.start[dim] + block.block[dim])
                .max()
                .unwrap_or(0)),
        }
    }

    fn decode_virtual_mappings(
        heap_data: &[u8],
        sizeof_size: usize,
    ) -> Result<Vec<VirtualMapping>> {
        let mut pos = 0usize;
        let version = *heap_data
            .get(pos)
            .ok_or_else(|| Error::InvalidFormat("empty virtual dataset heap object".into()))?;
        pos += 1;
        if version > 1 {
            return Err(Error::Unsupported(format!(
                "virtual dataset heap encoding version {version}"
            )));
        }
        let count = usize_from_u64(
            read_le_uint_at(heap_data, &mut pos, sizeof_size)?,
            "virtual dataset mapping count",
        )?;
        if count > MAX_VDS_MAPPINGS {
            return Err(Error::InvalidFormat(format!(
                "virtual dataset mapping count {count} exceeds supported maximum {MAX_VDS_MAPPINGS}"
            )));
        }
        let mut mappings = Vec::with_capacity(count);
        let mut file_names: Vec<String> = Vec::with_capacity(count);
        let mut dataset_names: Vec<String> = Vec::with_capacity(count);

        for _ in 0..count {
            mappings.push(Self::decode_virtual_mapping(
                heap_data,
                &mut pos,
                version,
                sizeof_size,
                &mut file_names,
                &mut dataset_names,
            )?);
        }

        Ok(mappings)
    }

    fn decode_virtual_mapping(
        heap_data: &[u8],
        pos: &mut usize,
        version: u8,
        sizeof_size: usize,
        file_names: &mut Vec<String>,
        dataset_names: &mut Vec<String>,
    ) -> Result<VirtualMapping> {
        let flags = Self::decode_virtual_mapping_flags(heap_data, pos, version)?;
        let names = Self::decode_virtual_source_names(
            heap_data,
            pos,
            sizeof_size,
            flags,
            file_names,
            dataset_names,
        )?;
        let source_select = Self::decode_virtual_selection(heap_data, pos)?;
        let virtual_select = Self::decode_virtual_selection(heap_data, pos)?;
        trace_vds_source_resolve(&names.file_name, &names.dataset_name);
        file_names.push(names.file_name.clone());
        dataset_names.push(names.dataset_name.clone());
        Ok(VirtualMapping {
            file_name: names.file_name,
            dataset_name: names.dataset_name,
            source_select,
            virtual_select,
        })
    }

    fn decode_virtual_mapping_flags(data: &[u8], pos: &mut usize, version: u8) -> Result<u8> {
        if version == 0 {
            return Ok(0);
        }
        let flags = *data.get(*pos).ok_or_else(|| {
            Error::InvalidFormat("truncated virtual dataset mapping flags".into())
        })?;
        *pos += 1;
        Ok(flags)
    }

    fn decode_virtual_source_names(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        flags: u8,
        file_names: &[String],
        dataset_names: &[String],
    ) -> Result<DecodedVirtualSourceNames> {
        Ok(DecodedVirtualSourceNames {
            file_name: Self::decode_virtual_source_file_name(
                heap_data,
                pos,
                sizeof_size,
                flags,
                file_names,
            )?,
            dataset_name: Self::decode_virtual_source_dataset_name(
                heap_data,
                pos,
                sizeof_size,
                flags,
                dataset_names,
            )?,
        })
    }

    fn decode_virtual_source_file_name(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        flags: u8,
        file_names: &[String],
    ) -> Result<String> {
        if flags & 0x04 != 0 {
            return Ok(".".to_string());
        }
        if flags & 0x01 != 0 {
            return Self::decode_virtual_shared_name_ref(
                heap_data,
                pos,
                sizeof_size,
                file_names,
                "virtual dataset shared file-name index",
                "invalid shared VDS source file reference",
            );
        }
        read_c_string(heap_data, pos)
    }

    fn decode_virtual_source_dataset_name(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        flags: u8,
        dataset_names: &[String],
    ) -> Result<String> {
        if flags & 0x02 != 0 {
            return Self::decode_virtual_shared_name_ref(
                heap_data,
                pos,
                sizeof_size,
                dataset_names,
                "virtual dataset shared dataset-name index",
                "invalid shared VDS source dataset reference",
            );
        }
        read_c_string(heap_data, pos)
    }

    fn decode_virtual_shared_name_ref(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        names: &[String],
        index_context: &'static str,
        invalid_context: &'static str,
    ) -> Result<String> {
        let origin = usize_from_u64(read_le_uint_at(heap_data, pos, sizeof_size)?, index_context)?;
        names
            .get(origin)
            .cloned()
            .ok_or_else(|| Error::InvalidFormat(invalid_context.into()))
    }

    fn decode_virtual_selection(data: &[u8], pos: &mut usize) -> Result<VirtualSelection> {
        const H5S_SEL_POINTS: u32 = 1;
        const H5S_SEL_HYPERSLABS: u32 = 2;
        const H5S_SEL_ALL: u32 = 3;

        let start_pos = *pos;
        let sel_type = read_le_u32_at(data, pos)?;
        let selection = match sel_type {
            H5S_SEL_ALL => Self::decode_virtual_all_selection(data, pos)?,
            H5S_SEL_POINTS => Self::decode_virtual_point_selection(data, pos)?,
            H5S_SEL_HYPERSLABS => Self::decode_virtual_hyperslab_selection(data, pos)?,
            _ => {
                return Err(Error::Unsupported(format!(
                    "virtual dataset selection type {sel_type}"
                )))
            }
        };

        trace_selection_deserialize(&data[start_pos..*pos], sel_type);
        Ok(selection)
    }

    fn decode_virtual_all_selection(data: &[u8], pos: &mut usize) -> Result<VirtualSelection> {
        let version = read_le_u32_at(data, pos)?;
        if version != 1 {
            return Err(Error::Unsupported(format!(
                "virtual all-selection version {version}"
            )));
        }
        *pos = pos
            .checked_add(8)
            .ok_or_else(|| Error::InvalidFormat("virtual all-selection offset overflow".into()))?;
        if *pos > data.len() {
            return Err(Error::InvalidFormat(
                "truncated virtual all-selection header".into(),
            ));
        }
        Ok(VirtualSelection::All)
    }

    fn decode_virtual_point_selection(data: &[u8], pos: &mut usize) -> Result<VirtualSelection> {
        let version = read_le_u32_at(data, pos)?;
        let enc_size = Self::decode_virtual_point_enc_size(data, pos, version)?;
        let rank = Self::decode_virtual_selection_rank(data, pos)?;
        let point_count = usize_from_u64(
            read_le_uint_at(data, pos, enc_size)?,
            "virtual point selection count",
        )?;
        let coordinate_count = point_count.checked_mul(rank).ok_or_else(|| {
            Error::InvalidFormat("virtual point selection coordinate count overflow".into())
        })?;
        let mut points = Vec::with_capacity(point_count);
        for _ in 0..point_count {
            let mut point = Vec::with_capacity(rank);
            for _ in 0..rank {
                point.push(read_le_uint_at(data, pos, enc_size)?);
            }
            points.push(point);
        }
        debug_assert_eq!(coordinate_count, points.iter().map(Vec::len).sum::<usize>());
        Ok(VirtualSelection::Points(points))
    }

    fn decode_virtual_point_enc_size(data: &[u8], pos: &mut usize, version: u32) -> Result<usize> {
        let enc_size = if version >= 2 {
            read_u8_at(data, pos)? as usize
        } else if version == 1 {
            *pos = pos.checked_add(8).ok_or_else(|| {
                Error::InvalidFormat("virtual point-selection offset overflow".into())
            })?;
            4
        } else {
            return Err(Error::Unsupported(format!(
                "virtual point selection version {version}"
            )));
        };
        validate_vds_selection_enc_size(enc_size, "point")?;
        Ok(enc_size)
    }

    fn decode_virtual_hyperslab_selection(
        data: &[u8],
        pos: &mut usize,
    ) -> Result<VirtualSelection> {
        const H5S_HYPER_REGULAR: u8 = 0x01;

        let version = read_le_u32_at(data, pos)?;
        let header = Self::decode_virtual_hyperslab_header(data, pos, version)?;
        let rank = Self::decode_virtual_selection_rank(data, pos)?;

        if header.flags & H5S_HYPER_REGULAR != 0 {
            return Self::decode_virtual_regular_hyperslab_selection(
                data,
                pos,
                rank,
                header.enc_size,
            );
        }
        Self::decode_virtual_irregular_hyperslab_selection(data, pos, rank, header.enc_size)
    }

    fn decode_virtual_hyperslab_header(
        data: &[u8],
        pos: &mut usize,
        version: u32,
    ) -> Result<VirtualHyperslabHeader> {
        const H5S_SELECT_FLAG_BITS: u8 = 0x01;

        let (flags, enc_size) = if version >= 3 {
            let flags = read_u8_at(data, pos)?;
            let enc_size = read_u8_at(data, pos)? as usize;
            (flags, enc_size)
        } else if version == 2 {
            let flags = read_u8_at(data, pos)?;
            *pos = pos
                .checked_add(4)
                .ok_or_else(|| Error::InvalidFormat("virtual hyperslab offset overflow".into()))?;
            (flags, 8)
        } else if version == 1 {
            *pos = pos
                .checked_add(8)
                .ok_or_else(|| Error::InvalidFormat("virtual hyperslab offset overflow".into()))?;
            (0, 4)
        } else {
            return Err(Error::Unsupported(format!(
                "virtual hyperslab selection version {version}"
            )));
        };

        if flags & !H5S_SELECT_FLAG_BITS != 0 {
            return Err(Error::InvalidFormat(format!(
                "virtual hyperslab selection has unknown flags 0x{flags:02x}"
            )));
        }
        validate_vds_selection_enc_size(enc_size, "hyperslab")?;
        Ok(VirtualHyperslabHeader { flags, enc_size })
    }

    fn decode_virtual_selection_rank(data: &[u8], pos: &mut usize) -> Result<usize> {
        let rank = usize_from_u64(read_le_u32_at(data, pos)? as u64, "virtual selection rank")?;
        if rank == 0 || rank > MAX_VDS_SELECTION_RANK {
            return Err(Error::InvalidFormat(format!(
                "virtual selection rank {rank} exceeds supported maximum {MAX_VDS_SELECTION_RANK}"
            )));
        }
        Ok(rank)
    }

    fn decode_virtual_regular_hyperslab_selection(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
    ) -> Result<VirtualSelection> {
        let mut start = Vec::with_capacity(rank);
        let mut stride = Vec::with_capacity(rank);
        let mut count = Vec::with_capacity(rank);
        let mut block = Vec::with_capacity(rank);

        for _ in 0..rank {
            let dim = Self::decode_virtual_regular_hyperslab_dim(data, pos, enc_size)?;
            start.push(dim.start);
            stride.push(dim.stride);
            count.push(dim.count);
            block.push(dim.block);
        }

        Ok(VirtualSelection::Regular(RegularHyperslab {
            start,
            stride,
            count,
            block,
        }))
    }

    fn decode_virtual_irregular_hyperslab_selection(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
    ) -> Result<VirtualSelection> {
        let block_count = usize_from_u64(
            read_le_uint_at(data, pos, enc_size)?,
            "virtual hyperslab block count",
        )?;
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let block = Self::decode_virtual_irregular_hyperslab_block(data, pos, rank, enc_size)?;
            blocks.push(Self::materialize_virtual_irregular_hyperslab_block(block)?);
        }
        Ok(VirtualSelection::Irregular(blocks))
    }

    fn decode_virtual_hyperslab_vector(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
        decode_extent: bool,
    ) -> Result<Vec<u64>> {
        let mut values = Vec::with_capacity(rank);
        for _ in 0..rank {
            let value = read_le_uint_at(data, pos, enc_size)?;
            values.push(if decode_extent {
                decode_hyperslab_extent(value, enc_size)
            } else {
                value
            });
        }
        Ok(values)
    }

    fn decode_virtual_regular_hyperslab_dim(
        data: &[u8],
        pos: &mut usize,
        enc_size: usize,
    ) -> Result<DecodedRegularHyperslabDim> {
        Ok(DecodedRegularHyperslabDim {
            start: read_le_uint_at(data, pos, enc_size)?,
            stride: read_le_uint_at(data, pos, enc_size)?,
            count: decode_hyperslab_extent(read_le_uint_at(data, pos, enc_size)?, enc_size),
            block: decode_hyperslab_extent(read_le_uint_at(data, pos, enc_size)?, enc_size),
        })
    }

    fn decode_virtual_irregular_hyperslab_block(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
    ) -> Result<DecodedIrregularHyperslabBlock> {
        Ok(DecodedIrregularHyperslabBlock {
            start: Self::decode_virtual_hyperslab_vector(data, pos, rank, enc_size, false)?,
            end: Self::decode_virtual_hyperslab_vector(data, pos, rank, enc_size, false)?,
        })
    }

    fn materialize_virtual_irregular_hyperslab_block(
        block: DecodedIrregularHyperslabBlock,
    ) -> Result<IrregularHyperslabBlock> {
        let mut extents = Vec::with_capacity(block.start.len());
        for (start_coord, end_coord) in block.start.iter().zip(&block.end) {
            if end_coord < start_coord {
                return Err(Error::InvalidFormat(
                    "virtual irregular hyperslab end precedes start".into(),
                ));
            }
            extents.push(end_coord - start_coord + 1);
        }

        Ok(IrregularHyperslabBlock {
            start: block.start,
            block: extents,
        })
    }

    fn resolve_virtual_source_path(vds_path: Option<&Path>, source: &str) -> Result<PathBuf> {
        if source == "." {
            return vds_path.map(Path::to_path_buf).ok_or_else(|| {
                Error::Unsupported("same-file virtual dataset source has no file path".into())
            });
        }
        let source_path = Path::new(source);
        if source_path.is_absolute() && source_path.exists() {
            return Ok(source_path.to_path_buf());
        }

        if let Some(prefixed) = Self::resolve_virtual_source_with_env_prefix(vds_path, source_path)?
        {
            return Ok(prefixed);
        }

        if source_path.is_absolute() {
            return Ok(source_path.to_path_buf());
        }
        let base = vds_path.and_then(Path::parent).ok_or_else(|| {
            Error::Unsupported("relative virtual dataset source has no base file path".into())
        })?;
        Ok(base.join(source_path))
    }

    fn resolve_virtual_source_with_env_prefix(
        vds_path: Option<&Path>,
        source_path: &Path,
    ) -> Result<Option<PathBuf>> {
        let Ok(prefixes) = std::env::var("HDF5_VDS_PREFIX") else {
            return Ok(None);
        };

        let file_name = source_path.file_name().ok_or_else(|| {
            Error::InvalidFormat("virtual dataset source has no file name".into())
        })?;

        for raw_prefix in prefixes.split(':') {
            if raw_prefix.is_empty() || raw_prefix == "." {
                continue;
            }
            let prefix = Self::expand_virtual_prefix_origin(vds_path, raw_prefix)?;
            let candidate = prefix.join(file_name);
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }

        Ok(None)
    }

    fn expand_virtual_prefix_origin(vds_path: Option<&Path>, prefix: &str) -> Result<PathBuf> {
        const ORIGIN: &str = "${ORIGIN}";

        if let Some(rest) = prefix.strip_prefix(ORIGIN) {
            let origin_dir = vds_path
                .and_then(Path::parent)
                .map(Path::to_path_buf)
                .ok_or_else(|| {
                    Error::Unsupported("VDS ${ORIGIN} prefix has no base file path".into())
                })?;

            let trimmed = rest.strip_prefix(['/', '\\']).unwrap_or(rest);
            if trimmed.is_empty() {
                return Ok(origin_dir);
            }
            return Ok(origin_dir.join(trimmed));
        }

        Ok(PathBuf::from(prefix))
    }

    fn materialize_virtual_selection_points(
        selection: &VirtualSelection,
        dims: &[u64],
    ) -> Result<Vec<Vec<u64>>> {
        match selection {
            VirtualSelection::All => {
                let all = RegularHyperslab {
                    start: vec![0; dims.len()],
                    stride: vec![1; dims.len()],
                    count: vec![1; dims.len()],
                    block: dims.to_vec(),
                };
                Self::materialize_regular_hyperslab_points(&all, dims)
            }
            VirtualSelection::Points(points) => {
                Self::validate_virtual_point_coords(points, dims)?;
                Ok(points.clone())
            }
            VirtualSelection::Regular(selection) => {
                Self::materialize_regular_hyperslab_points(selection, dims)
            }
            VirtualSelection::Irregular(blocks) => {
                Self::materialize_irregular_hyperslab_points(blocks, dims)
            }
        }
    }

    fn materialize_regular_hyperslab_points(
        selection: &RegularHyperslab,
        dims: &[u64],
    ) -> Result<Vec<Vec<u64>>> {
        if selection.start.len() != dims.len() {
            return Err(Error::InvalidFormat(
                "virtual hyperslab rank does not match dataspace".into(),
            ));
        }
        let mut points = Vec::new();
        let mut current = vec![0u64; dims.len()];
        Self::push_hyperslab_points(selection, dims, 0, &mut current, &mut points)?;
        Ok(points)
    }

    fn virtual_selection_start(selection: &VirtualSelection, dim: usize) -> u64 {
        match selection {
            VirtualSelection::All => 0,
            VirtualSelection::Points(points) => {
                points.iter().map(|point| point[dim]).min().unwrap_or(0)
            }
            VirtualSelection::Regular(selection) => selection.start[dim],
            VirtualSelection::Irregular(blocks) => blocks
                .iter()
                .map(|block| block.start[dim])
                .min()
                .unwrap_or(0),
        }
    }

    fn virtual_selection_span(selection: &VirtualSelection, dims: &[u64], dim: usize) -> u64 {
        match selection {
            VirtualSelection::All => dims[dim],
            VirtualSelection::Points(points) => points
                .iter()
                .map(|point| point[dim])
                .max()
                .map(|end| end + 1 - Self::virtual_selection_start(selection, dim))
                .unwrap_or(0),
            VirtualSelection::Regular(selection) => {
                Self::regular_hyperslab_selected_span(selection, dims, dim)
            }
            VirtualSelection::Irregular(blocks) => blocks
                .iter()
                .map(|block| block.start[dim] + block.block[dim])
                .max()
                .map(|end| end - Self::virtual_selection_start(selection, dim))
                .unwrap_or(0),
        }
    }

    fn regular_hyperslab_selected_span(
        selection: &RegularHyperslab,
        dims: &[u64],
        dim: usize,
    ) -> u64 {
        let start = selection.start[dim];
        let stride = selection.stride[dim].max(1);
        let count = if selection.count[dim] == u64::MAX {
            ((dims[dim].saturating_sub(start)) + stride - 1) / stride
        } else {
            selection.count[dim]
        };
        let block = if selection.block[dim] == u64::MAX {
            dims[dim].saturating_sub(start)
        } else {
            selection.block[dim]
        };
        if count == 0 {
            0
        } else {
            (count - 1) * stride + block
        }
    }

    fn push_hyperslab_points(
        selection: &RegularHyperslab,
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        points: &mut Vec<Vec<u64>>,
    ) -> Result<()> {
        if dim == dims.len() {
            points.push(current.to_vec());
            return Ok(());
        }
        let start = selection.start[dim];
        let stride = selection.stride[dim].max(1);
        let count = if selection.count[dim] == u64::MAX {
            ((dims[dim].saturating_sub(start)) + stride - 1) / stride
        } else {
            selection.count[dim]
        };
        let block = if selection.block[dim] == u64::MAX {
            dims[dim].saturating_sub(start)
        } else {
            selection.block[dim]
        };

        for count_idx in 0..count {
            let base = start + count_idx * stride;
            for block_idx in 0..block {
                let coord = base + block_idx;
                if coord < dims[dim] {
                    current[dim] = coord;
                    Self::push_hyperslab_points(selection, dims, dim + 1, current, points)?;
                }
            }
        }
        Ok(())
    }

    fn materialize_irregular_hyperslab_points(
        blocks: &[IrregularHyperslabBlock],
        dims: &[u64],
    ) -> Result<Vec<Vec<u64>>> {
        let mut points = Vec::new();
        for block in blocks {
            if block.start.len() != dims.len() || block.block.len() != dims.len() {
                return Err(Error::InvalidFormat(
                    "virtual hyperslab rank does not match dataspace".into(),
                ));
            }
            let mut current = vec![0u64; dims.len()];
            Self::push_irregular_block_points(block, dims, 0, &mut current, &mut points)?;
        }
        Ok(points)
    }

    fn push_irregular_block_points(
        block: &IrregularHyperslabBlock,
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        points: &mut Vec<Vec<u64>>,
    ) -> Result<()> {
        if dim == dims.len() {
            points.push(current.to_vec());
            return Ok(());
        }
        for offset in 0..block.block[dim] {
            let coord = block.start[dim] + offset;
            if coord < dims[dim] {
                current[dim] = coord;
                Self::push_irregular_block_points(block, dims, dim + 1, current, points)?;
            }
        }
        Ok(())
    }

    fn validate_virtual_point_coords(points: &[Vec<u64>], dims: &[u64]) -> Result<()> {
        for point in points {
            if point.len() != dims.len() {
                return Err(Error::InvalidFormat(
                    "virtual point-selection rank does not match dataspace".into(),
                ));
            }
            for (&coord, &dim_extent) in point.iter().zip(dims) {
                if coord >= dim_extent {
                    return Err(Error::InvalidFormat(
                        "virtual point-selection coordinate exceeds dataspace extent".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn row_major_strides(dims: &[u64]) -> Result<Vec<usize>> {
        let mut strides = vec![1usize; dims.len()];
        for dim in (0..dims.len().saturating_sub(1)).rev() {
            strides[dim] = strides[dim + 1]
                .checked_mul(usize_from_u64(dims[dim + 1], "dataspace dimension")?)
                .ok_or_else(|| Error::InvalidFormat("dataspace stride overflow".into()))?;
        }
        Ok(strides)
    }

    fn linear_index(coords: &[u64], strides: &[usize]) -> Result<usize> {
        coords
            .iter()
            .zip(strides)
            .try_fold(0usize, |acc, (&coord, &stride)| {
                acc.checked_add(
                    usize_from_u64(coord, "dataspace coordinate")
                        .ok()?
                        .checked_mul(stride)?,
                )
            })
            .ok_or_else(|| Error::InvalidFormat("linear index overflow".into()))
    }

    fn read_chunked_implicit<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                return Err(Error::Unsupported(
                    "v4 implicit chunk index with filters is not implemented".into(),
                ));
            }
        }

        let ndims = chunk_ctx.data_dims.len();
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        let total_chunks: usize = chunks_per_dim
            .iter()
            .try_fold(1usize, |acc, &count| acc.checked_mul(count))
            .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?;

        let mut output = Self::filled_data(
            chunk_ctx.total_bytes / chunk_ctx.element_size,
            chunk_ctx.element_size,
            info,
        )?;
        for chunk_index in 0..total_chunks {
            let coords =
                Self::implicit_chunk_coords(chunk_index, chunk_ctx.chunk_dims, &chunks_per_dim);
            let offset = (chunk_index as u64)
                .checked_mul(chunk_ctx.chunk_bytes as u64)
                .and_then(|off| chunk_ctx.idx_addr.checked_add(off))
                .ok_or_else(|| Error::InvalidFormat("implicit chunk address overflow".into()))?;
            reader.seek(offset)?;
            let raw = reader.read_bytes(chunk_ctx.chunk_bytes)?;
            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        if ndims == 0 {
            output.truncate(chunk_ctx.total_bytes.min(chunk_ctx.chunk_bytes));
        }
        Ok(output)
    }

    fn read_chunked_fixed_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info,
                chunk_ctx.chunk_bytes,
                reader.sizeof_size() as usize,
            )?
        } else {
            0
        };

        let elements = crate::format::fixed_array::read_fixed_array_chunks(
            reader,
            chunk_ctx.idx_addr,
            filtered,
            chunk_size_len,
        )?;
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        let mut output = Self::filled_data(
            chunk_ctx.total_bytes / chunk_ctx.element_size,
            chunk_ctx.element_size,
            info,
        )?;

        for (chunk_index, element) in elements.iter().enumerate() {
            Self::trace_linear_chunk_lookup(
                "hdf5.chunk_index.fixed_array.lookup",
                chunk_ctx.idx_addr,
                chunk_index as u64,
                element.addr,
                element.nbytes.unwrap_or(chunk_ctx.chunk_bytes as u64),
                element.filter_mask,
            );

            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }

            let coords =
                Self::implicit_chunk_coords(chunk_index, chunk_ctx.chunk_dims, &chunks_per_dim);
            reader.seek(element.addr)?;
            let read_size = usize_from_u64(
                element.nbytes.unwrap_or(chunk_ctx.chunk_bytes as u64),
                "fixed-array chunk size",
            )?;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read fixed-array chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw,
                        pipeline,
                        chunk_ctx.element_size,
                        element.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    fn read_chunked_extensible_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info,
                chunk_ctx.chunk_bytes,
                reader.sizeof_size() as usize,
            )?
        } else {
            0
        };

        let elements = crate::format::extensible_array::read_extensible_array_chunks(
            reader,
            chunk_ctx.idx_addr,
            filtered,
            chunk_size_len,
        )?;
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        let mut output = Self::filled_data(
            chunk_ctx.total_bytes / chunk_ctx.element_size,
            chunk_ctx.element_size,
            info,
        )?;

        for (chunk_index, element) in elements.iter().enumerate() {
            Self::trace_linear_chunk_lookup(
                "hdf5.chunk_index.extensible_array.lookup",
                chunk_ctx.idx_addr,
                chunk_index as u64,
                element.addr,
                element.nbytes.unwrap_or(chunk_ctx.chunk_bytes as u64),
                element.filter_mask,
            );

            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }

            let coords =
                Self::implicit_chunk_coords(chunk_index, chunk_ctx.chunk_dims, &chunks_per_dim);
            reader.seek(element.addr)?;
            let read_size = usize_from_u64(
                element.nbytes.unwrap_or(chunk_ctx.chunk_bytes as u64),
                "extensible-array chunk size",
            )?;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read extensible-array chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw,
                        pipeline,
                        chunk_ctx.element_size,
                        element.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    fn read_chunked_btree_v2<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info,
                chunk_ctx.chunk_bytes,
                reader.sizeof_size() as usize,
            )?
        } else {
            0
        };
        let records = crate::format::btree_v2::collect_all_records(reader, chunk_ctx.idx_addr)?;
        let mut output = Self::filled_data(
            chunk_ctx.total_bytes / chunk_ctx.element_size,
            chunk_ctx.element_size,
            info,
        )?;

        for record in records {
            let (addr, nbytes, filter_mask, scaled) = Self::decode_btree_v2_chunk_record(
                &record,
                filtered,
                chunk_size_len,
                reader.sizeof_addr() as usize,
                chunk_ctx.data_dims.len(),
                chunk_ctx.chunk_bytes,
            )?;
            Self::trace_btree2_chunk_lookup(chunk_ctx.idx_addr, &scaled, addr, nbytes, filter_mask);
            if crate::io::reader::is_undef_addr(addr) {
                continue;
            }

            let coords: Vec<u64> = scaled
                .iter()
                .zip(chunk_ctx.chunk_dims)
                .map(|(&coord, &chunk)| {
                    coord.checked_mul(chunk).ok_or_else(|| {
                        Error::InvalidFormat("v2-B-tree chunk coordinate overflow".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            reader.seek(addr).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to seek to v2-B-tree chunk address {addr}: {err}"
                ))
            })?;
            let read_size = usize_from_u64(nbytes, "v2-B-tree chunk size")?;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read v2-B-tree chunk at address {addr} with size {nbytes}: {err}"
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw,
                        pipeline,
                        chunk_ctx.element_size,
                        filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    #[cfg(feature = "tracehash")]
    fn trace_linear_chunk_lookup(
        function: &'static str,
        index_addr: u64,
        chunk_index: u64,
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
    ) {
        let mut th = tracehash::Call::new(function, file!(), line!());
        th.input_u64(index_addr);
        th.input_u64(chunk_index);
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(if crate::io::reader::is_undef_addr(addr) {
            0
        } else {
            nbytes
        });
        th.output_u64(filter_mask as u64);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_linear_chunk_lookup(
        _function: &'static str,
        _index_addr: u64,
        _chunk_index: u64,
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
    ) {
    }

    #[cfg(feature = "tracehash")]
    fn trace_btree1_chunk_lookup(
        index_addr: u64,
        scaled: &[u64],
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
    ) {
        let mut th = tracehash::th_call!("hdf5.chunk_index.btree1.lookup");
        th.input_u64(index_addr);
        for coord in scaled {
            th.input_u64(*coord);
        }
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(if crate::io::reader::is_undef_addr(addr) {
            0
        } else {
            nbytes
        });
        th.output_u64(filter_mask as u64);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_btree1_chunk_lookup(
        _index_addr: u64,
        _scaled: &[u64],
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
    ) {
    }

    #[cfg(feature = "tracehash")]
    fn trace_btree2_chunk_lookup(
        index_addr: u64,
        scaled: &[u64],
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
    ) {
        let mut th = tracehash::th_call!("hdf5.chunk_index.btree2.lookup");
        th.input_u64(index_addr);
        for coord in scaled {
            th.input_u64(*coord);
        }
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(if crate::io::reader::is_undef_addr(addr) {
            0
        } else {
            nbytes
        });
        th.output_u64(filter_mask as u64);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_btree2_chunk_lookup(
        _index_addr: u64,
        _scaled: &[u64],
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
    ) {
    }

    #[cfg(feature = "tracehash")]
    fn trace_btree2_record_decode(
        record: &[u8],
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
        scaled: &[u64],
    ) {
        let mut th = tracehash::th_call!("hdf5.chunk_index.btree2.record_decode");
        th.input_bytes(record);
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(nbytes);
        th.output_u64(filter_mask as u64);
        th.output_u64(scaled.len() as u64);
        for coord in scaled {
            th.output_u64(*coord);
        }
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_btree2_record_decode(
        _record: &[u8],
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
        _scaled: &[u64],
    ) {
    }

    fn scaled_chunk_coords(coords: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>> {
        if coords.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match chunk dimensions".into(),
            ));
        }
        coords
            .iter()
            .zip(chunk_dims)
            .map(|(&coord, &dim)| {
                if dim == 0 {
                    return Err(Error::InvalidFormat("chunk dimension is zero".into()));
                }
                if coord % dim != 0 {
                    return Err(Error::InvalidFormat(
                        "chunk coordinate is not aligned to chunk dimension".into(),
                    ));
                }
                Ok(coord / dim)
            })
            .collect()
    }

    fn decode_btree_v2_chunk_record(
        record: &[u8],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
        ndims: usize,
        chunk_bytes: usize,
    ) -> Result<(u64, u64, u32, Vec<u64>)> {
        let mut pos = 0;
        let addr = read_le_uint(&record[pos..], sizeof_addr)?;
        pos += sizeof_addr;

        let (nbytes, filter_mask) = if filtered {
            let nbytes = read_le_uint(&record[pos..], chunk_size_len)?;
            pos += chunk_size_len;
            if pos + 4 > record.len() {
                return Err(Error::InvalidFormat(
                    "truncated v2-B-tree filter mask".into(),
                ));
            }
            let filter_mask = read_le_u32(record, pos)?;
            pos += 4;
            (nbytes, filter_mask)
        } else {
            (chunk_bytes as u64, 0)
        };

        let mut scaled = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let coord = read_le_uint(&record[pos..], 8)?;
            pos += 8;
            scaled.push(coord);
        }

        Self::trace_btree2_record_decode(record, addr, nbytes, filter_mask, &scaled);

        Ok((addr, nbytes, filter_mask, scaled))
    }

    fn filtered_chunk_size_len(
        info: &DatasetInfo,
        chunk_bytes: usize,
        sizeof_size: usize,
    ) -> Result<usize> {
        if info.layout.version > 4 {
            return Ok(sizeof_size);
        }

        let bits = if chunk_bytes == 0 {
            0
        } else {
            usize::BITS as usize - chunk_bytes.leading_zeros() as usize
        };
        Ok((1 + ((bits + 8) / 8)).min(8))
    }

    fn chunks_per_dim(data_dims: &[u64], chunk_dims: &[u64]) -> Result<Vec<usize>> {
        data_dims
            .iter()
            .zip(chunk_dims)
            .map(|(&dim, &chunk)| {
                if chunk == 0 {
                    return Err(Error::InvalidFormat("zero chunk dimension".into()));
                }
                let count = dim
                    .checked_add(chunk - 1)
                    .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?
                    / chunk;
                usize_from_u64(count, "chunks per dimension")
            })
            .collect()
    }

    fn implicit_chunk_coords(
        chunk_index: usize,
        chunk_dims: &[u64],
        chunks_per_dim: &[usize],
    ) -> Vec<u64> {
        let ndims = chunk_dims.len();
        let mut remaining = chunk_index;
        let mut coords = vec![0u64; ndims];

        for dim in (0..ndims).rev() {
            let chunk_coord = remaining % chunks_per_dim[dim];
            remaining /= chunks_per_dim[dim];
            coords[dim] = (chunk_coord as u64) * chunk_dims[dim];
        }

        coords
    }

    /// Collect all chunk entries from a v1 B-tree.
    /// Returns Vec<(coordinates, chunk_addr, chunk_size, filter_mask)>.
    /// Pure deserializer for one v1 chunk-index B-tree node — returns
    /// either the leaf chunk records or the list of child addresses,
    /// depending on the node level. Mirrors libhdf5's
    /// `H5B__cache_deserialize` for the chunk-index node type. No I/O
    /// after the read; no recursion.
    fn decode_chunk_btree_node<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        ndims: usize,
    ) -> Result<ChunkBTreeNode> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }

        let node_type = reader.read_u8()?;
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw data B-tree (type 1), got type {node_type}"
            )));
        }

        let level = reader.read_u8()?;
        let entries_used = reader.read_u16()? as usize;
        let _left_sibling = reader.read_addr()?;
        let _right_sibling = reader.read_addr()?;

        if level == 0 {
            let mut records = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                records.push(Self::decode_chunk_btree_leaf_record(reader, ndims)?);
            }
            Self::skip_chunk_btree_final_key(reader, ndims)?;
            Ok(ChunkBTreeNode::Leaf(records))
        } else {
            let mut child_addrs = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                child_addrs.push(Self::decode_chunk_btree_child_addr(reader, ndims)?);
            }
            Self::skip_chunk_btree_final_key(reader, ndims)?;
            Ok(ChunkBTreeNode::Internal(child_addrs))
        }
    }

    /// Walk a v1 chunk-index B-tree, depth-first. Mirrors libhdf5's
    /// `H5D__btree_idx_iterate` / `H5B__iterate_helper`: the actual
    /// node decoding lives in `decode_chunk_btree_node`.
    fn collect_btree_v1_chunks<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        ndims: usize,
    ) -> Result<Vec<ChunkBTreeRecord>> {
        let node = Self::decode_chunk_btree_node(reader, addr, ndims)?;
        match node {
            ChunkBTreeNode::Leaf(records) => Ok(records),
            ChunkBTreeNode::Internal(child_addrs) => {
                let mut all_chunks = Vec::new();
                for child_addr in child_addrs {
                    let mut child_chunks =
                        Self::collect_btree_v1_chunks(reader, child_addr, ndims)?;
                    all_chunks.append(&mut child_chunks);
                }
                Ok(all_chunks)
            }
        }
    }

    fn decode_chunk_btree_leaf_record<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
    ) -> Result<ChunkBTreeRecord> {
        let chunk_size = reader.read_u32()? as u64;
        let filter_mask = reader.read_u32()?;
        let mut coords = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            coords.push(reader.read_u64()?);
        }
        let _extra = reader.read_u64()?;
        let chunk_addr = reader.read_addr()?;
        Ok(ChunkBTreeRecord {
            coords,
            chunk_addr,
            chunk_size,
            filter_mask,
        })
    }

    fn decode_chunk_btree_child_addr<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
    ) -> Result<u64> {
        let _chunk_size = reader.read_u32()?;
        let _filter_mask = reader.read_u32()?;
        for _ in 0..=ndims {
            let _ = reader.read_u64()?;
        }
        reader.read_addr()
    }

    fn skip_chunk_btree_final_key<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
    ) -> Result<()> {
        let _final_chunk_size = reader.read_u32()?;
        let _final_filter_mask = reader.read_u32()?;
        for _ in 0..=ndims {
            let _ = reader.read_u64()?;
        }
        Ok(())
    }

    fn process_btree_v1_chunk_record<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        btree_addr: u64,
        chunk_record: &ChunkBTreeRecord,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        chunk_bytes: usize,
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let scaled = Self::scaled_chunk_coords(&chunk_record.coords, chunk_dims)?;
        Self::trace_btree1_chunk_lookup(
            btree_addr,
            &scaled,
            chunk_record.chunk_addr,
            chunk_record.chunk_size,
            chunk_record.filter_mask,
        );

        if crate::io::reader::is_undef_addr(chunk_record.chunk_addr) {
            return Ok(());
        }

        let raw = Self::read_btree_v1_chunk_payload(
            reader,
            chunk_record,
            info,
            chunk_bytes,
            element_size,
        )?;
        Self::copy_chunk_to_output(
            &raw,
            &chunk_record.coords,
            data_dims,
            chunk_dims,
            element_size,
            output,
        )
    }

    fn read_btree_v1_chunk_payload<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        chunk_record: &ChunkBTreeRecord,
        info: &DatasetInfo,
        chunk_bytes: usize,
        element_size: usize,
    ) -> Result<Vec<u8>> {
        reader.seek(chunk_record.chunk_addr)?;
        let read_size = usize_from_u64(chunk_record.chunk_size, "v1 B-tree chunk size")?;
        let mut raw = reader.read_bytes(read_size)?;

        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                raw = filters::apply_pipeline_reverse_with_mask_expected(
                    &raw,
                    pipeline,
                    element_size,
                    chunk_record.filter_mask,
                    chunk_bytes,
                )?;
            }
        }

        Ok(raw)
    }

    /// Copy chunk data into the output buffer at the correct position.
    fn copy_chunk_to_output(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let ndims = data_dims.len();

        if ndims == 1 {
            return Self::copy_chunk_1d(
                chunk_data,
                coords,
                data_dims,
                chunk_dims,
                element_size,
                output,
            );
        }

        let copy_plan = Self::build_chunk_copy_plan(data_dims, chunk_dims, element_size)?;
        Self::copy_chunk_nd(
            chunk_data,
            coords,
            data_dims,
            element_size,
            output,
            &copy_plan,
        )
    }

    fn copy_chunk_1d(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let start = usize_from_u64(coords[0], "chunk coordinate")?;
        let chunk_size = usize_from_u64(chunk_dims[0], "chunk dimension")?;
        let data_size = usize_from_u64(data_dims[0], "dataset dimension")?;
        if start >= data_size {
            return Ok(());
        }

        let n_copy = chunk_size.min(data_size - start);
        let src_bytes = n_copy
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy size overflow".into()))?;
        let dst_offset = start
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy offset overflow".into()))?;

        if dst_offset
            .checked_add(src_bytes)
            .is_some_and(|end| end <= output.len())
            && src_bytes <= chunk_data.len()
        {
            output[dst_offset..dst_offset + src_bytes].copy_from_slice(&chunk_data[..src_bytes]);
        }
        Ok(())
    }

    fn build_chunk_copy_plan(
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
    ) -> Result<ChunkCopyPlan> {
        let ndims = data_dims.len();

        let mut out_strides = vec![0usize; ndims];
        out_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            out_strides[i] = out_strides[i + 1]
                .checked_mul(usize_from_u64(data_dims[i + 1], "dataset dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk output stride overflow".into()))?;
        }

        let mut chunk_strides = vec![0usize; ndims];
        chunk_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            chunk_strides[i] = chunk_strides[i + 1]
                .checked_mul(usize_from_u64(chunk_dims[i + 1], "chunk dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk stride overflow".into()))?;
        }

        let mut chunk_suffix_products = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            chunk_suffix_products[d] = chunk_suffix_products[d + 1]
                .checked_mul(usize_from_u64(chunk_dims[d + 1], "chunk dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk suffix product overflow".into()))?;
        }

        let total_chunk_elements = chunk_dims.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(usize_from_u64(dim, "chunk dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))
        })?;

        Ok(ChunkCopyPlan {
            out_strides,
            chunk_strides,
            chunk_suffix_products,
            total_chunk_elements,
        })
    }

    fn copy_chunk_nd(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
        copy_plan: &ChunkCopyPlan,
    ) -> Result<()> {
        let ndims = data_dims.len();
        let mut idx = vec![0usize; ndims];

        for elem_idx in 0..copy_plan.total_chunk_elements {
            let mut remaining = elem_idx;
            for d in 0..ndims {
                idx[d] = remaining / copy_plan.chunk_suffix_products[d];
                remaining %= copy_plan.chunk_suffix_products[d];
            }

            let mut in_bounds = true;
            let mut out_offset = 0usize;
            let mut chunk_offset = 0usize;
            for d in 0..ndims {
                let global = usize_from_u64(coords[d], "chunk coordinate")?
                    .checked_add(idx[d])
                    .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
                if global >= usize_from_u64(data_dims[d], "dataset dimension")? {
                    in_bounds = false;
                    break;
                }
                out_offset =
                    out_offset
                        .checked_add(global.checked_mul(copy_plan.out_strides[d]).ok_or_else(
                            || Error::InvalidFormat("chunk output offset overflow".into()),
                        )?)
                        .ok_or_else(|| {
                            Error::InvalidFormat("chunk output offset overflow".into())
                        })?;
                chunk_offset = chunk_offset
                    .checked_add(idx[d].checked_mul(copy_plan.chunk_strides[d]).ok_or_else(
                        || Error::InvalidFormat("chunk input offset overflow".into()),
                    )?)
                    .ok_or_else(|| Error::InvalidFormat("chunk input offset overflow".into()))?;
            }

            if in_bounds
                && out_offset
                    .checked_add(element_size)
                    .is_some_and(|end| end <= output.len())
                && chunk_offset
                    .checked_add(element_size)
                    .is_some_and(|end| end <= chunk_data.len())
            {
                output[out_offset..out_offset + element_size]
                    .copy_from_slice(&chunk_data[chunk_offset..chunk_offset + element_size]);
            }
        }
        Ok(())
    }
}

fn read_le_uint(bytes: &[u8], size: usize) -> Result<u64> {
    if size == 0 || size > 8 || bytes.len() < size {
        return Err(Error::InvalidFormat(format!(
            "invalid little-endian integer size {size}"
        )));
    }

    let mut value = 0u64;
    for (idx, byte) in bytes[..size].iter().enumerate() {
        value |= (*byte as u64) << (idx * 8);
    }
    Ok(value)
}

#[cfg(feature = "tracehash")]
fn trace_selection_deserialize(data: &[u8], sel_type: u32) {
    let mut th = tracehash::th_call!("hdf5.selection.deserialize");
    th.input_bytes(data);
    th.output_value(&(true));
    th.output_u64(sel_type as u64);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_selection_deserialize(_data: &[u8], _sel_type: u32) {}

#[cfg(feature = "tracehash")]
fn trace_vlen_read(len: u64, data: &[u8]) {
    let mut th = tracehash::th_call!("hdf5.vlen.read");
    th.input_u64(len);
    th.output_value(&(true));
    th.output_value(data);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_vlen_read(_len: u64, _data: &[u8]) {}

#[cfg(feature = "tracehash")]
fn trace_vds_source_resolve(file_name: &str, dataset_name: &str) {
    let mut th = tracehash::th_call!("hdf5.vds.source.resolve");
    th.input_bytes(file_name.as_bytes());
    th.input_bytes(dataset_name.as_bytes());
    th.output_value(&(file_name == "."));
    th.output_value(file_name.as_bytes());
    th.output_value(dataset_name.as_bytes());
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_vds_source_resolve(_file_name: &str, _dataset_name: &str) {}

fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

fn read_u8_at(bytes: &[u8], pos: &mut usize) -> Result<u8> {
    let value = *bytes
        .get(*pos)
        .ok_or_else(|| Error::InvalidFormat("truncated byte field".into()))?;
    *pos += 1;
    Ok(value)
}

fn read_le_u32_at(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    if *pos + 4 > bytes.len() {
        return Err(Error::InvalidFormat("truncated u32 field".into()));
    }
    let value = read_le_u32(bytes, *pos)?;
    *pos += 4;
    Ok(value)
}

fn read_le_u32(bytes: &[u8], pos: usize) -> Result<u32> {
    let window = bytes
        .get(pos..pos + 4)
        .ok_or_else(|| Error::InvalidFormat("truncated u32 field".into()))?;
    Ok(u32::from_le_bytes([
        window[0], window[1], window[2], window[3],
    ]))
}

fn read_le_uint_at(bytes: &[u8], pos: &mut usize, size: usize) -> Result<u64> {
    if *pos + size > bytes.len() {
        return Err(Error::InvalidFormat("truncated integer field".into()));
    }
    let value = read_le_uint(&bytes[*pos..*pos + size], size)?;
    *pos += size;
    Ok(value)
}

fn read_c_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    let rel_end = bytes[*pos..]
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| Error::InvalidFormat("unterminated string field".into()))?;
    let end = *pos + rel_end;
    let value = std::str::from_utf8(&bytes[*pos..end])
        .map_err(|err| Error::InvalidFormat(format!("invalid UTF-8 string field: {err}")))?
        .to_string();
    *pos = end + 1;
    Ok(value)
}

fn decode_hyperslab_extent(value: u64, enc_size: usize) -> u64 {
    match enc_size {
        2 if value == u16::MAX as u64 => u64::MAX,
        4 if value == u32::MAX as u64 => u64::MAX,
        8 if value == u64::MAX => u64::MAX,
        _ => value,
    }
}

fn validate_vds_selection_enc_size(enc_size: usize, kind: &str) -> Result<()> {
    match enc_size {
        2 | 4 | 8 => Ok(()),
        _ => Err(Error::InvalidFormat(format!(
            "virtual {kind} selection uses unsupported encoded integer size {enc_size}"
        ))),
    }
}

fn read_unsigned_int(bytes: &[u8], little_endian: bool) -> u128 {
    let mut value = 0u128;
    let n = bytes.len().min(16);
    if little_endian {
        for (idx, byte) in bytes.iter().take(n).enumerate() {
            value |= (*byte as u128) << (idx * 8);
        }
    } else {
        for byte in bytes.iter().take(n) {
            value = (value << 8) | (*byte as u128);
        }
    }
    value
}

fn read_signed_int(bytes: &[u8], little_endian: bool) -> i128 {
    let n = bytes.len().min(16);
    let unsigned = read_unsigned_int(bytes, little_endian);
    if n == 0 {
        return 0;
    }

    let bits = n * 8;
    if bits == 128 {
        unsigned as i128
    } else {
        let sign_bit = 1u128 << (bits - 1);
        if unsigned & sign_bit == 0 {
            unsigned as i128
        } else {
            (unsigned as i128) - (1i128 << bits)
        }
    }
}

fn endian_array<const N: usize>(
    bytes: &[u8],
    byte_order: Option<crate::format::messages::datatype::ByteOrder>,
) -> Result<[u8; N]> {
    if bytes.len() < N {
        return Err(Error::InvalidFormat(
            "floating-point payload too short".into(),
        ));
    }

    let mut arr = [0u8; N];
    arr.copy_from_slice(&bytes[..N]);
    let need_swap = match byte_order {
        Some(crate::format::messages::datatype::ByteOrder::BigEndian) => {
            cfg!(target_endian = "little")
        }
        Some(crate::format::messages::datatype::ByteOrder::LittleEndian) | None => {
            cfg!(target_endian = "big")
        }
    };
    if need_swap {
        arr.reverse();
    }
    Ok(arr)
}

fn decode_fixed_string(bytes: &[u8]) -> String {
    decode_fixed_string_with_padding(bytes, 1)
}

fn decode_fixed_string_with_padding(bytes: &[u8], padding: u8) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let bytes = &bytes[..end];
    let s = String::from_utf8_lossy(bytes);
    if padding == 2 {
        s.trim_end().to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::tempdir;

    fn le_u32(value: u32, out: &mut Vec<u8>) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn le_u64(value: u64, out: &mut Vec<u8>) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn build_global_heap_collection(
        collection_addr: usize,
        object_index: u16,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut bytes = vec![0; collection_addr];
        bytes.extend_from_slice(b"GCOL");
        bytes.push(1);
        bytes.extend_from_slice(&[0; 3]);

        let aligned = (payload.len() as u64 + 7) & !7;
        let total_size = 16u64 + 16u64 + aligned + 16u64;
        bytes.extend_from_slice(&total_size.to_le_bytes());

        bytes.extend_from_slice(&object_index.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes.resize(collection_addr + 16 + 16 + aligned as usize, 0);

        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes
    }

    #[test]
    fn virtual_point_selection_decodes_and_materializes() {
        let mut encoded = Vec::new();
        le_u32(1, &mut encoded); // H5S_SEL_POINTS
        le_u32(2, &mut encoded); // point selection version
        encoded.push(8); // encoded integer size
        le_u32(2, &mut encoded); // rank
        le_u64(3, &mut encoded); // number of points
        le_u64(0, &mut encoded);
        le_u64(1, &mut encoded);
        le_u64(2, &mut encoded);
        le_u64(3, &mut encoded);
        le_u64(1, &mut encoded);
        le_u64(4, &mut encoded);
        let mut pos = 0;

        let selection = Dataset::decode_virtual_selection(&encoded, &mut pos)
            .expect("point VDS selections should decode");

        let points = Dataset::materialize_virtual_selection_points(&selection, &[3, 5])
            .expect("point VDS selections should materialize");
        assert_eq!(points, vec![vec![0, 1], vec![2, 3], vec![1, 4]]);
    }

    #[test]
    fn virtual_irregular_hyperslab_selection_decodes_and_materializes() {
        let mut encoded = Vec::new();
        le_u32(2, &mut encoded); // H5S_SEL_HYPERSLABS
        le_u32(3, &mut encoded); // hyperslab selection version
        encoded.push(0); // flags without H5S_HYPER_REGULAR
        encoded.push(8); // encoded integer size
        le_u32(2, &mut encoded); // rank
        le_u64(2, &mut encoded); // block count
        le_u64(0, &mut encoded);
        le_u64(1, &mut encoded);
        le_u64(0, &mut encoded);
        le_u64(2, &mut encoded);
        le_u64(2, &mut encoded);
        le_u64(0, &mut encoded);
        le_u64(2, &mut encoded);
        le_u64(1, &mut encoded);

        let mut pos = 0;
        let selection = Dataset::decode_virtual_selection(&encoded, &mut pos)
            .expect("irregular VDS hyperslabs should decode");

        let points = Dataset::materialize_virtual_selection_points(&selection, &[4, 4])
            .expect("irregular VDS hyperslabs should materialize");
        assert_eq!(
            points,
            vec![vec![0, 1], vec![0, 2], vec![2, 0], vec![2, 1],]
        );
    }

    #[test]
    fn virtual_hyperslab_selection_rejects_unknown_flags() {
        let mut encoded = Vec::new();
        le_u32(2, &mut encoded); // H5S_SEL_HYPERSLABS
        le_u32(3, &mut encoded); // hyperslab selection version
        encoded.push(0x80); // unknown flag
        encoded.push(8); // encoded integer size
        le_u32(1, &mut encoded); // rank
        le_u64(1, &mut encoded); // block count
        le_u64(0, &mut encoded); // start
        le_u64(0, &mut encoded); // end

        let mut pos = 0;
        let err = Dataset::decode_virtual_selection(&encoded, &mut pos)
            .expect_err("unknown hyperslab flags should be rejected");

        assert!(matches!(err, Error::InvalidFormat(_)));
        assert!(err.to_string().contains("unknown flags"));
    }

    #[test]
    fn virtual_output_extent_uses_point_and_irregular_destination_bounds() {
        let point_extent = Dataset::virtual_mapping_output_extent(
            &VirtualSelection::All,
            &VirtualSelection::Points(vec![vec![1, 4], vec![3, 2]]),
            &[10, 10],
            1,
        )
        .expect("point VDS extent should derive from destination bounds");
        assert_eq!(point_extent, 5);

        let irregular_extent = Dataset::virtual_mapping_output_extent(
            &VirtualSelection::All,
            &VirtualSelection::Irregular(vec![
                IrregularHyperslabBlock {
                    start: vec![0, 1],
                    block: vec![1, 2],
                },
                IrregularHyperslabBlock {
                    start: vec![4, 0],
                    block: vec![2, 1],
                },
            ]),
            &[10, 10],
            0,
        )
        .expect("irregular VDS extent should derive from destination block bounds");
        assert_eq!(irregular_extent, 6);
    }

    #[test]
    fn virtual_source_resolution_requires_base_path_for_relative_and_same_file_sources() {
        for source in [".", "relative-source.h5"] {
            let err = Dataset::resolve_virtual_source_path(None, source)
                .expect_err("VDS source resolution without a base file should fail");
            assert!(matches!(err, Error::Unsupported(_)));
        }
    }

    #[test]
    fn btree_v1_chunk_records_preserve_8_byte_chunk_addresses() {
        use std::io::Cursor;

        let mut node = Vec::new();
        node.extend_from_slice(b"TREE");
        node.push(1); // raw data B-tree
        node.push(0); // leaf
        node.extend_from_slice(&1u16.to_le_bytes()); // entries used
        node.extend_from_slice(&u64::MAX.to_le_bytes()); // left sibling
        node.extend_from_slice(&u64::MAX.to_le_bytes()); // right sibling

        node.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        node.extend_from_slice(&0u32.to_le_bytes()); // filter mask
        node.extend_from_slice(&4u64.to_le_bytes()); // dim 0 chunk offset
        node.extend_from_slice(&8u64.to_le_bytes()); // dim 1 chunk offset
        node.extend_from_slice(&0u64.to_le_bytes()); // extra element-size dimension
        let large_chunk_addr = 0x1_0000_0040u64;
        node.extend_from_slice(&large_chunk_addr.to_le_bytes());

        node.extend_from_slice(&0u32.to_le_bytes()); // final key chunk size
        node.extend_from_slice(&0u32.to_le_bytes()); // final key filter mask
        node.extend_from_slice(&0u64.to_le_bytes()); // final dim 0 key
        node.extend_from_slice(&0u64.to_le_bytes()); // final dim 1 key
        node.extend_from_slice(&0u64.to_le_bytes()); // final extra key

        let mut reader = HdfReader::new(Cursor::new(node));
        reader.set_sizeof_addr(8);
        let chunks = Dataset::collect_btree_v1_chunks(&mut reader, 0, 2).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, vec![4, 8]);
        assert_eq!(chunks[0].1, large_chunk_addr);
        assert_eq!(chunks[0].2, 16);
        assert_eq!(chunks[0].3, 0);
    }

    #[test]
    fn filtered_implicit_chunk_index_is_rejected() {
        let info = DatasetInfo {
            dataspace: DataspaceMessage {
                version: 2,
                space_type: DataspaceType::Simple,
                ndims: 1,
                dims: vec![4],
                max_dims: None,
            },
            datatype: DatatypeMessage {
                version: 1,
                class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
                class_bits: [0, 0, 0],
                size: 4,
                properties: Vec::new(),
            },
            layout: DataLayoutMessage {
                version: 4,
                layout_class: LayoutClass::Chunked,
                compact_data: None,
                contiguous_addr: None,
                contiguous_size: None,
                chunk_dims: Some(vec![2]),
                chunk_index_addr: Some(0),
                chunk_index_type: Some(ChunkIndexType::Implicit),
                chunk_element_size: None,
                chunk_flags: None,
                chunk_encoded_dims: None,
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
                data_addr: Some(0),
                virtual_heap_addr: None,
                virtual_heap_index: None,
            },
            filter_pipeline: Some(FilterPipelineMessage {
                version: 2,
                filters: vec![crate::format::messages::filter_pipeline::FilterDesc {
                    id: crate::format::messages::filter_pipeline::FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![4],
                }],
            }),
            fill_value: None,
        };
        let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
        let err = Dataset::read_chunked_implicit(&mut reader, 0, &info, &[4], &[2], 8, 4, 16)
            .expect_err("filtered implicit chunk indexes should be rejected");

        assert!(matches!(err, Error::Unsupported(_)));
        assert!(
            err.to_string()
                .contains("v4 implicit chunk index with filters"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn old_fill_value_must_match_datatype_size_during_dataset_parse() {
        let messages = vec![
            RawMessage {
                msg_type: object_header::MSG_DATASPACE,
                flags: 0,
                creation_index: None,
                chunk_index: 0,
                data: vec![2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            },
            RawMessage {
                msg_type: object_header::MSG_DATATYPE,
                flags: 0,
                creation_index: None,
                chunk_index: 0,
                data: vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0],
            },
            RawMessage {
                msg_type: object_header::MSG_LAYOUT,
                flags: 0,
                creation_index: None,
                chunk_index: 0,
                data: vec![3, 0, 0, 0],
            },
            RawMessage {
                msg_type: object_header::MSG_FILL_VALUE_OLD,
                flags: 0,
                creation_index: None,
                chunk_index: 0,
                data: vec![2, 0, 0, 0, 1, 2],
            },
        ];

        let err = Dataset::parse_info(&messages, 8, 8)
            .expect_err("old fill value with mismatched datatype size should fail");
        assert!(
            err.to_string().contains("does not match datatype size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn vlen_sequence_reads_only_requested_bytes() {
        let heap_addr = 32u64;
        let heap = build_global_heap_collection(heap_addr as usize, 1, &[1, 2, 3, 4, 99, 100]);
        let mut reader = HdfReader::new(Cursor::new(heap));
        reader.set_sizeof_size(8);

        let base = DatatypeMessage {
            version: 1,
            class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
            class_bits: [0, 0, 0],
            size: 2,
            properties: vec![0, 0, 16, 0],
        };

        let mut descriptor = Vec::new();
        descriptor.extend_from_slice(&2u32.to_le_bytes());
        descriptor.extend_from_slice(&heap_addr.to_le_bytes());
        descriptor.extend_from_slice(&1u32.to_le_bytes());

        let value = Dataset::decode_vlen_value(Some(&base), &descriptor, 8, &mut reader)
            .expect("exact-length vlen read should succeed");
        match value {
            H5Value::VarLen(values) => {
                assert_eq!(values.len(), 2);
                assert!(matches!(values[0], H5Value::UInt(513)));
                assert!(matches!(values[1], H5Value::UInt(1027)));
            }
            other => panic!("expected VarLen, got {other:?}"),
        }
    }

    #[test]
    fn vlen_string_rejects_short_heap_payload() {
        let heap_addr = 32u64;
        let heap = build_global_heap_collection(heap_addr as usize, 1, b"abc");
        let mut reader = HdfReader::new(Cursor::new(heap));
        reader.set_sizeof_size(8);

        let base = DatatypeMessage {
            version: 1,
            class: crate::format::messages::datatype::DatatypeClass::String,
            class_bits: [0, 0, 0],
            size: 1,
            properties: Vec::new(),
        };

        let mut descriptor = Vec::new();
        descriptor.extend_from_slice(&4u32.to_le_bytes());
        descriptor.extend_from_slice(&heap_addr.to_le_bytes());
        descriptor.extend_from_slice(&1u32.to_le_bytes());

        let err = Dataset::decode_vlen_value(Some(&base), &descriptor, 8, &mut reader)
            .expect_err("short vlen string payload should fail");
        assert!(
            err.to_string().contains("payload too short"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn virtual_output_dims_respects_vds_view_for_unlimited_dimensions() {
        let dir = tempdir().unwrap();
        let short_path = dir.path().join("short.h5");
        let long_path = dir.path().join("long.h5");

        {
            let mut wf = crate::hl::writable_file::WritableFile::create(&short_path).unwrap();
            wf.new_dataset_builder("data")
                .write::<i32>(&[1, 2, 3])
                .unwrap();
            wf.close().unwrap();
        }

        {
            let mut wf = crate::hl::writable_file::WritableFile::create(&long_path).unwrap();
            wf.new_dataset_builder("data")
                .write::<i32>(&[1, 2, 3, 4, 5])
                .unwrap();
            wf.close().unwrap();
        }

        let mappings = vec![
            VirtualMapping {
                file_name: short_path.to_string_lossy().into_owned(),
                dataset_name: "data".to_string(),
                source_select: VirtualSelection::All,
                virtual_select: VirtualSelection::All,
            },
            VirtualMapping {
                file_name: long_path.to_string_lossy().into_owned(),
                dataset_name: "data".to_string(),
                source_select: VirtualSelection::All,
                virtual_select: VirtualSelection::All,
            },
        ];
        let info = DatasetInfo {
            dataspace: DataspaceMessage {
                version: 2,
                space_type: DataspaceType::Simple,
                ndims: 1,
                dims: vec![0],
                max_dims: Some(vec![u64::MAX]),
            },
            datatype: DatatypeMessage {
                version: 1,
                class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
                class_bits: [0, 0, 0],
                size: 4,
                properties: vec![0, 0, 32, 0],
            },
            layout: DataLayoutMessage {
                version: 4,
                layout_class: LayoutClass::Virtual,
                compact_data: None,
                contiguous_addr: None,
                contiguous_size: None,
                chunk_dims: None,
                chunk_index_addr: None,
                chunk_index_type: None,
                chunk_element_size: None,
                chunk_flags: None,
                chunk_encoded_dims: None,
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
                data_addr: None,
                virtual_heap_addr: None,
                virtual_heap_index: None,
            },
            filter_pipeline: None,
            fill_value: None,
        };

        let last_available =
            Dataset::virtual_output_dims(&mappings, None, &info, VdsView::LastAvailable).unwrap();
        let first_missing =
            Dataset::virtual_output_dims(&mappings, None, &info, VdsView::FirstMissing).unwrap();

        assert_eq!(last_available, vec![5]);
        assert_eq!(first_missing, vec![3]);
    }
}
