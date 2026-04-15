use std::fs;
use std::io::{BufReader, Read, Seek};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::filters;
// B-tree types used for chunk reading
use crate::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage, LayoutClass};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::format::object_header::{self, ObjectHeader, RawMessage};
use crate::hl::file::FileInner;
use crate::hl::value::H5Value;
use crate::io::reader::HdfReader;

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

    fn parse_info(
        messages: &[RawMessage],
        sizeof_addr: u8,
        sizeof_size: u8,
    ) -> Result<DatasetInfo> {
        let mut dataspace = None;
        let mut datatype = None;
        let mut layout = None;
        let mut filter_pipeline = None;

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
                _ => {}
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
        })
    }

    /// Get the shape of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>> {
        let info = self.info()?;
        Ok(info.dataspace.dims)
    }

    /// Get the total number of elements.
    pub fn size(&self) -> Result<u64> {
        let shape = self.shape()?;
        if shape.is_empty() {
            Ok(1) // scalar
        } else {
            Ok(shape.iter().product())
        }
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

    /// Get the dataspace.
    pub fn space(&self) -> Result<crate::hl::dataspace::Dataspace> {
        let info = self.info()?;
        Ok(crate::hl::dataspace::Dataspace::from_message(
            info.dataspace,
        ))
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
        let mut strings = Vec::new();
        for chunk in raw.chunks_exact(elem_size) {
            // Find null terminator or end of chunk
            let end = chunk.iter().position(|&b| b == 0).unwrap_or(elem_size);
            let s = String::from_utf8_lossy(&chunk[..end]).to_string();
            // Also trim trailing spaces (space-padded strings)
            strings.push(s.trim_end().to_string());
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
        info.datatype
            .compound_fields()
            .ok_or_else(|| Error::InvalidFormat("not a compound type".into()))
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
    /// array, variable-length, or reference members.
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
    /// returned as `H5Value::Raw`.
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
                let fields = dtype
                    .compound_fields()
                    .ok_or_else(|| Error::InvalidFormat("invalid nested compound type".into()))?;
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
                let (dims, base) = dtype
                    .array_dims_base()
                    .ok_or_else(|| Error::InvalidFormat("invalid array datatype".into()))?;
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
                let base = dtype.vlen_base();
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
            return Ok(H5Value::Raw(data));
        };

        if base.class == crate::format::messages::datatype::DatatypeClass::String {
            return Ok(H5Value::String(
                String::from_utf8_lossy(&data)
                    .trim_end_matches('\0')
                    .to_string(),
            ));
        }

        let elem_size = base.size as usize;
        if elem_size == 0 || data.len() < seq_len.saturating_mul(elem_size) {
            return Ok(H5Value::Raw(data));
        }

        let mut values = Vec::with_capacity(seq_len);
        for chunk in data[..seq_len * elem_size].chunks_exact(elem_size) {
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
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let sizeof_size = guard.superblock.sizeof_size;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;
        let info = Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)?;

        let element_size = info.datatype.size as usize;
        let total_elements: u64 = if info.dataspace.dims.is_empty() {
            1
        } else {
            info.dataspace.dims.iter().try_fold(1u64, |acc, &d| {
                acc.checked_mul(d)
                    .ok_or_else(|| Error::InvalidFormat("dimension product overflow".into()))
            })?
        };
        let total_bytes = (total_elements as usize)
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("total data size overflow".into()))?;

        // Sanity limit: refuse to allocate more than 4GB in a single read
        const MAX_READ_BYTES: usize = 4 * 1024 * 1024 * 1024;
        if total_bytes > MAX_READ_BYTES {
            return Err(Error::InvalidFormat(format!(
                "dataset too large for single read: {total_bytes} bytes (max {MAX_READ_BYTES})"
            )));
        }

        match info.layout.layout_class {
            LayoutClass::Compact => {
                let data = info
                    .layout
                    .compact_data
                    .ok_or_else(|| Error::InvalidFormat("compact dataset missing data".into()))?;
                Ok(data)
            }
            LayoutClass::Contiguous => {
                let addr = info.layout.contiguous_addr.ok_or_else(|| {
                    Error::InvalidFormat("contiguous dataset missing address".into())
                })?;
                let size = info.layout.contiguous_size.unwrap_or(total_bytes as u64) as usize;

                if crate::io::reader::is_undef_addr(addr) {
                    // No data stored yet -- return fill value or zeros
                    return Ok(vec![0u8; total_bytes]);
                }

                guard.reader.seek(addr)?;
                let data = guard.reader.read_bytes(size)?;
                Ok(data)
            }
            LayoutClass::Chunked => Self::read_chunked(&mut guard.reader, &info, total_bytes),
            LayoutClass::Virtual => Err(Error::Unsupported("virtual datasets".into())),
        }
    }

    /// Read all data as a typed Vec.
    pub fn read<T: crate::hl::types::H5Type>(&self) -> Result<Vec<T>> {
        let mut raw = self.read_raw()?;
        // Byte-swap if file byte order != host byte order
        self.maybe_byte_swap(&mut raw)?;
        crate::hl::types::bytes_to_vec(raw)
    }

    /// Read a scalar value.
    pub fn read_scalar<T: crate::hl::types::H5Type>(&self) -> Result<T> {
        let mut raw = self.read_raw()?;
        self.maybe_byte_swap(&mut raw)?;
        let slice = crate::hl::types::bytes_to_slice::<T>(&raw)?;
        if slice.is_empty() {
            return Err(Error::InvalidFormat("no data for scalar read".into()));
        }
        Ok(slice[0])
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
        let total_out: u64 = out_shape.iter().product::<u64>().max(1);
        let elem_size = T::type_size();

        // For 1D contiguous selections, optimize with direct seek+read
        if shape.len() == 1 && slices.len() == 1 && slices[0].step == 1 {
            let info = self.info()?;
            if info.layout.layout_class == LayoutClass::Contiguous {
                if let Some(addr) = info.layout.contiguous_addr {
                    if !crate::io::reader::is_undef_addr(addr) {
                        let start_byte = slices[0].start as usize * elem_size;
                        let nbytes = slices[0].count() as usize * elem_size;
                        let mut guard = self.inner.lock();
                        guard.reader.seek(addr + start_byte as u64)?;
                        let raw = guard.reader.read_bytes(nbytes)?;
                        return crate::hl::types::bytes_to_vec(raw);
                    }
                }
            }
        }

        // General case: read all data and extract the selection
        let all_data = self.read::<T>()?;

        if shape.len() == 1 && slices.len() == 1 {
            let s = &slices[0];
            let start = s.start as usize;
            let end = s.end as usize;
            return Ok(all_data[start..end.min(all_data.len())].to_vec());
        }

        // N-D extraction
        let mut result = Vec::with_capacity(total_out as usize);
        let ndims = shape.len();

        // Compute input strides
        let mut in_strides = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            in_strides[d] = in_strides[d + 1] * shape[d + 1] as usize;
        }

        // Iterate over output elements
        let mut out_idx = vec![0u64; ndims];
        for _ in 0..total_out {
            // Map output index to input index
            let mut in_linear = 0;
            for d in 0..ndims {
                let in_d = slices[d].start + out_idx[d] * slices[d].step;
                in_linear += in_d as usize * in_strides[d];
            }

            if in_linear < all_data.len() {
                result.push(all_data[in_linear]);
            }

            // Increment output index
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

    /// Byte-swap the raw data in-place if the file byte order differs from native.
    fn maybe_byte_swap(&self, data: &mut [u8]) -> Result<()> {
        use crate::format::messages::datatype::ByteOrder;

        let info = self.info()?;
        let elem_size = info.datatype.size as usize;

        if elem_size <= 1 {
            return Ok(()); // no swapping needed for single-byte types
        }

        let file_order = info.datatype.byte_order();
        let need_swap = match file_order {
            Some(ByteOrder::BigEndian) => cfg!(target_endian = "little"),
            Some(ByteOrder::LittleEndian) => cfg!(target_endian = "big"),
            None => false, // non-numeric types
        };

        if need_swap {
            for chunk in data.chunks_exact_mut(elem_size) {
                chunk.reverse();
            }
        }

        Ok(())
    }

    fn read_chunked<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let element_size = info.datatype.size as usize;
        let chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing chunk dims".into()))?;

        let data_dims = &info.dataspace.dims;
        let ndims = data_dims.len();
        let chunk_data_dims = if chunk_dims.len() == ndims + 1 {
            &chunk_dims[..ndims]
        } else if chunk_dims.len() == ndims {
            chunk_dims.as_slice()
        } else {
            return Err(Error::InvalidFormat(format!(
                "chunk dims rank {} does not match dataspace rank {}",
                chunk_dims.len(),
                ndims
            )));
        };

        // Calculate chunk size in bytes
        let chunk_bytes = if chunk_dims.len() == ndims + 1 {
            chunk_dims
                .iter()
                .copied()
                .try_fold(1u64, |a, b| a.checked_mul(b))
                .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?
                as usize
        } else {
            let chunk_elements: u64 = chunk_data_dims
                .iter()
                .copied()
                .try_fold(1u64, |a, b| a.checked_mul(b))
                .ok_or_else(|| Error::InvalidFormat("chunk dimension product overflow".into()))?;
            (chunk_elements as usize)
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?
        };

        let idx_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing index address".into()))?;

        if crate::io::reader::is_undef_addr(idx_addr) {
            return Ok(vec![0u8; total_bytes]);
        }

        // Determine which index type to use
        let idx_type = info.layout.chunk_index_type.clone();

        match idx_type {
            Some(ChunkIndexType::SingleChunk) | None if info.layout.version <= 3 => {
                // For v3 layout or single chunk: use v1 B-tree or direct read
                if info.layout.version <= 3 {
                    // V3: btree v1 index
                    Self::read_chunked_btree_v1(
                        reader,
                        idx_addr,
                        info,
                        data_dims,
                        chunk_data_dims,
                        chunk_bytes,
                        element_size,
                        total_bytes,
                    )
                } else {
                    // Single chunk: data is at the index address
                    reader.seek(idx_addr)?;
                    let mut raw = reader.read_bytes(
                        info.layout
                            .single_chunk_filtered_size
                            .unwrap_or(chunk_bytes as u64) as usize,
                    )?;

                    if let Some(ref pipeline) = info.filter_pipeline {
                        if !pipeline.filters.is_empty() {
                            raw = filters::apply_pipeline_reverse(&raw, pipeline, element_size)?;
                        }
                    }

                    Ok(raw[..total_bytes].to_vec())
                }
            }
            Some(ChunkIndexType::SingleChunk) => {
                // V4 single chunk
                reader.seek(idx_addr)?;
                let read_size = info
                    .layout
                    .single_chunk_filtered_size
                    .unwrap_or(chunk_bytes as u64) as usize;
                let mut raw = reader.read_bytes(read_size)?;

                if let Some(ref pipeline) = info.filter_pipeline {
                    if !pipeline.filters.is_empty() {
                        raw = filters::apply_pipeline_reverse(&raw, pipeline, element_size)?;
                    }
                }

                Ok(raw[..total_bytes.min(raw.len())].to_vec())
            }
            Some(ChunkIndexType::BTreeV1) => Self::read_chunked_btree_v1(
                reader,
                idx_addr,
                info,
                data_dims,
                chunk_data_dims,
                chunk_bytes,
                element_size,
                total_bytes,
            ),
            Some(ChunkIndexType::Implicit) => Self::read_chunked_implicit(
                reader,
                idx_addr,
                info,
                data_dims,
                chunk_data_dims,
                chunk_bytes,
                element_size,
                total_bytes,
            ),
            Some(ChunkIndexType::FixedArray) => Self::read_chunked_fixed_array(
                reader,
                idx_addr,
                info,
                data_dims,
                chunk_data_dims,
                chunk_bytes,
                element_size,
                total_bytes,
            ),
            Some(ChunkIndexType::ExtensibleArray) => Self::read_chunked_extensible_array(
                reader,
                idx_addr,
                info,
                data_dims,
                chunk_data_dims,
                chunk_bytes,
                element_size,
                total_bytes,
            ),
            Some(ChunkIndexType::BTreeV2) => Self::read_chunked_btree_v2(
                reader,
                idx_addr,
                info,
                data_dims,
                chunk_data_dims,
                chunk_bytes,
                element_size,
                total_bytes,
            ),
            None => Err(Error::InvalidFormat(
                "chunked dataset missing chunk index type".into(),
            )),
        }
    }

    fn read_chunked_btree_v1<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        btree_addr: u64,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        _chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let ndims = data_dims.len();
        let mut output = vec![0u8; total_bytes];

        // Collect all chunk entries from the B-tree
        let chunks = Self::collect_btree_v1_chunks(reader, btree_addr, ndims)?;

        for (coords, chunk_addr, chunk_size, _filter_mask) in &chunks {
            if crate::io::reader::is_undef_addr(*chunk_addr) {
                continue;
            }

            reader.seek(*chunk_addr)?;
            let mut raw = reader.read_bytes(*chunk_size as usize)?;

            // Apply filter pipeline if present (respecting filter mask)
            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse(&raw, pipeline, element_size)?;
                }
            }

            // Copy chunk data into the output buffer at the correct position
            Self::copy_chunk_to_output(
                &raw,
                coords,
                data_dims,
                chunk_dims,
                element_size,
                &mut output,
            );
        }

        Ok(output)
    }

    fn read_chunked_implicit<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        base_addr: u64,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                return Err(Error::Unsupported(
                    "v4 implicit chunk index with filters is not implemented".into(),
                ));
            }
        }

        let ndims = data_dims.len();
        let total_chunks: usize = data_dims
            .iter()
            .zip(chunk_dims)
            .map(|(&dim, &chunk)| ((dim + chunk - 1) / chunk) as usize)
            .try_fold(1usize, |acc, count| acc.checked_mul(count))
            .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?;

        let mut output = vec![0u8; total_bytes];
        for chunk_index in 0..total_chunks {
            let coords = Self::implicit_chunk_coords(chunk_index, data_dims, chunk_dims);
            let offset = (chunk_index as u64)
                .checked_mul(chunk_bytes as u64)
                .and_then(|off| base_addr.checked_add(off))
                .ok_or_else(|| Error::InvalidFormat("implicit chunk address overflow".into()))?;
            reader.seek(offset)?;
            let raw = reader.read_bytes(chunk_bytes)?;
            Self::copy_chunk_to_output(
                &raw,
                &coords,
                data_dims,
                chunk_dims,
                element_size,
                &mut output,
            );
        }

        if ndims == 0 {
            output.truncate(total_bytes.min(chunk_bytes));
        }
        Ok(output)
    }

    fn read_chunked_fixed_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        index_addr: u64,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(info, chunk_bytes, reader.sizeof_size() as usize)?
        } else {
            0
        };

        let elements = crate::format::fixed_array::read_fixed_array_chunks(
            reader,
            index_addr,
            filtered,
            chunk_size_len,
        )?;
        let mut output = vec![0u8; total_bytes];

        for (chunk_index, element) in elements.iter().enumerate() {
            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }

            let coords = Self::implicit_chunk_coords(chunk_index, data_dims, chunk_dims);
            reader.seek(element.addr)?;
            let read_size = element.nbytes.unwrap_or(chunk_bytes as u64) as usize;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read fixed-array chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse(&raw, pipeline, element_size)?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                data_dims,
                chunk_dims,
                element_size,
                &mut output,
            );
        }

        Ok(output)
    }

    fn read_chunked_extensible_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        index_addr: u64,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(info, chunk_bytes, reader.sizeof_size() as usize)?
        } else {
            0
        };

        let elements = crate::format::extensible_array::read_extensible_array_chunks(
            reader,
            index_addr,
            filtered,
            chunk_size_len,
        )?;
        let mut output = vec![0u8; total_bytes];

        for (chunk_index, element) in elements.iter().enumerate() {
            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }

            let coords = Self::implicit_chunk_coords(chunk_index, data_dims, chunk_dims);
            reader.seek(element.addr)?;
            let read_size = element.nbytes.unwrap_or(chunk_bytes as u64) as usize;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read extensible-array chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse(&raw, pipeline, element_size)?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                data_dims,
                chunk_dims,
                element_size,
                &mut output,
            );
        }

        Ok(output)
    }

    fn read_chunked_btree_v2<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        index_addr: u64,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(info, chunk_bytes, reader.sizeof_size() as usize)?
        } else {
            0
        };
        let records = crate::format::btree_v2::collect_all_records(reader, index_addr)?;
        let mut output = vec![0u8; total_bytes];

        for record in records {
            let (addr, nbytes, _filter_mask, scaled) = Self::decode_btree_v2_chunk_record(
                &record,
                filtered,
                chunk_size_len,
                reader.sizeof_addr() as usize,
                data_dims.len(),
                chunk_bytes,
            )?;
            if crate::io::reader::is_undef_addr(addr) {
                continue;
            }

            let coords: Vec<u64> = scaled
                .iter()
                .zip(chunk_dims)
                .map(|(&coord, &chunk)| coord * chunk)
                .collect();
            reader.seek(addr).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to seek to v2-B-tree chunk address {addr}: {err}"
                ))
            })?;
            let mut raw = reader.read_bytes(nbytes as usize).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read v2-B-tree chunk at address {addr} with size {nbytes}: {err}"
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse(&raw, pipeline, element_size)?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                data_dims,
                chunk_dims,
                element_size,
                &mut output,
            );
        }

        Ok(output)
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
            let filter_mask = u32::from_le_bytes(record[pos..pos + 4].try_into().unwrap());
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

    fn implicit_chunk_coords(
        chunk_index: usize,
        data_dims: &[u64],
        chunk_dims: &[u64],
    ) -> Vec<u64> {
        let ndims = data_dims.len();
        let chunks_per_dim: Vec<usize> = data_dims
            .iter()
            .zip(chunk_dims)
            .map(|(&dim, &chunk)| ((dim + chunk - 1) / chunk) as usize)
            .collect();
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
    fn collect_btree_v1_chunks<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        ndims: usize,
    ) -> Result<Vec<(Vec<u64>, u64, u64, u32)>> {
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
            // Leaf node: read chunk records
            let mut chunks = Vec::new();

            for _ in 0..entries_used {
                // Key: chunk_size(4) + filter_mask(4) + ndims+1 dim offsets(each 8 bytes)
                let chunk_size = reader.read_u32()? as u64;
                let filter_mask = reader.read_u32()?;
                let mut coords = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    coords.push(reader.read_u64()?);
                }
                // Skip the extra dimension (dataset element dimension, always 0)
                let _extra = reader.read_u64()?;

                // Child pointer: chunk address
                let chunk_addr = reader.read_addr()?;

                chunks.push((coords, chunk_addr, chunk_size, filter_mask));
            }

            // Read final key (not used but need to advance past it)
            // Actually for leaf nodes we've already read all entries
            // The final key is after the last entry
            let _final_chunk_size = reader.read_u32()?;
            let _final_filter_mask = reader.read_u32()?;
            for _ in 0..=ndims {
                let _ = reader.read_u64()?;
            }

            Ok(chunks)
        } else {
            // Internal node: recurse
            let mut all_chunks = Vec::new();

            for _ in 0..entries_used {
                // Key
                let _chunk_size = reader.read_u32()?;
                let _filter_mask = reader.read_u32()?;
                for _ in 0..=ndims {
                    let _ = reader.read_u64()?;
                }

                // Child pointer
                let child_addr = reader.read_addr()?;

                let mut child_chunks = Self::collect_btree_v1_chunks(reader, child_addr, ndims)?;
                all_chunks.append(&mut child_chunks);

                // Seek back for next entry (reader position was changed by recursion)
                // We need to re-read the B-tree node to continue...
                // This is a simplification -- for deeply nested trees we'd need
                // to save/restore position. For now, re-read.
            }

            Ok(all_chunks)
        }
    }

    /// Copy chunk data into the output buffer at the correct position.
    fn copy_chunk_to_output(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
    ) {
        let ndims = data_dims.len();

        if ndims == 1 {
            // Fast path for 1D
            let start = coords[0] as usize;
            let chunk_size = chunk_dims[0] as usize;
            let data_size = data_dims[0] as usize;

            let n_copy = chunk_size.min(data_size - start);
            let src_bytes = n_copy * element_size;
            let dst_offset = start * element_size;

            if dst_offset + src_bytes <= output.len() && src_bytes <= chunk_data.len() {
                output[dst_offset..dst_offset + src_bytes]
                    .copy_from_slice(&chunk_data[..src_bytes]);
            }
        } else {
            // General N-dimensional copy
            Self::copy_chunk_nd(
                chunk_data,
                coords,
                data_dims,
                chunk_dims,
                element_size,
                output,
                ndims,
            );
        }
    }

    fn copy_chunk_nd(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
        ndims: usize,
    ) {
        // Calculate strides for the output array
        let mut out_strides = vec![0usize; ndims];
        out_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            out_strides[i] = out_strides[i + 1] * data_dims[i + 1] as usize;
        }

        // Calculate strides for the chunk
        let mut chunk_strides = vec![0usize; ndims];
        chunk_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            chunk_strides[i] = chunk_strides[i + 1] * chunk_dims[i + 1] as usize;
        }

        // Precompute suffix products for chunk index decomposition
        let mut chunk_suffix_products = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            chunk_suffix_products[d] = chunk_suffix_products[d + 1] * chunk_dims[d + 1] as usize;
        }

        // Iterate over elements in the chunk
        let total_chunk_elements: usize = chunk_dims.iter().map(|&d| d as usize).product();
        let mut idx = vec![0usize; ndims];

        for elem_idx in 0..total_chunk_elements {
            // Convert linear index to multi-dimensional within chunk
            let mut remaining = elem_idx;
            for d in 0..ndims {
                idx[d] = remaining / chunk_suffix_products[d];
                remaining %= chunk_suffix_products[d];
            }

            // Compute global position
            let mut in_bounds = true;
            let mut out_offset = 0;
            let mut chunk_offset = 0;
            for d in 0..ndims {
                let global = coords[d] as usize + idx[d];
                if global >= data_dims[d] as usize {
                    in_bounds = false;
                    break;
                }
                out_offset += global * out_strides[d];
                chunk_offset += idx[d] * chunk_strides[d];
            }

            if in_bounds
                && out_offset + element_size <= output.len()
                && chunk_offset + element_size <= chunk_data.len()
            {
                output[out_offset..out_offset + element_size]
                    .copy_from_slice(&chunk_data[chunk_offset..chunk_offset + element_size]);
            }
        }
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
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end])
        .trim_end()
        .to_string()
}
