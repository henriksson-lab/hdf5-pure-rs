use std::fs;

use crate::engine::writer::{
    CompoundFieldSpec, DatasetSpec, DtypeSpec, FillValueSpec, HdfFileWriter,
};
use crate::error::{Error, Result};
use crate::hl::types::{H5Type, TypeClass};

/// Builder for creating datasets with a fluent API.
pub struct DatasetBuilder<'a> {
    writer: &'a mut HdfFileWriter<fs::File>,
    parent: String,
    name: String,
    shape: Option<Vec<u64>>,
    max_shape: Option<Vec<u64>>,
    chunk_dims: Option<Vec<u64>>,
    deflate_level: Option<u32>,
    shuffle: bool,
    compact: bool,
    fill_value: Option<Vec<u8>>,
    alloc_time: u8,
    fill_time: u8,
}

impl<'a> DatasetBuilder<'a> {
    pub(crate) fn new(writer: &'a mut HdfFileWriter<fs::File>, parent: &str, name: &str) -> Self {
        Self {
            writer,
            parent: parent.to_string(),
            name: name.to_string(),
            shape: None,
            max_shape: None,
            chunk_dims: None,
            deflate_level: None,
            shuffle: false,
            compact: false,
            fill_value: None,
            alloc_time: 1,
            fill_time: 2,
        }
    }

    /// Set the dataset shape.
    pub fn shape(mut self, dims: &[u64]) -> Self {
        self.shape = Some(dims.to_vec());
        self
    }

    /// Set chunk dimensions (enables chunked storage).
    pub fn chunk(mut self, dims: &[u64]) -> Self {
        self.chunk_dims = Some(dims.to_vec());
        self
    }

    /// Enable deflate (gzip) compression at the given level (0-9).
    pub fn deflate(mut self, level: u32) -> Self {
        self.deflate_level = Some(level);
        self
    }

    /// Enable byte shuffle filter (should be used with compression).
    pub fn shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    /// Make the dataset resizable (unlimited max dimensions).
    /// Requires chunked storage.
    pub fn resizable(mut self) -> Self {
        self.max_shape = Some(vec![u64::MAX]); // sentinel for "unlimited"
        self
    }

    /// Set explicit maximum dimensions.
    pub fn max_shape(mut self, dims: &[u64]) -> Self {
        self.max_shape = Some(dims.to_vec());
        self
    }

    /// Use compact storage (data embedded in object header, for small datasets).
    pub fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Set raw HDF5 allocation-time and fill-write-time properties.
    pub fn fill_properties(mut self, alloc_time: u8, fill_time: u8) -> Self {
        self.alloc_time = alloc_time;
        self.fill_time = fill_time;
        self
    }

    /// Set a scalar fill value for missing or newly allocated dataset storage.
    pub fn fill_value<T: H5Type>(mut self, value: T) -> Self {
        let byte_ptr = &value as *const T as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(byte_ptr, T::type_size()) };
        self.fill_value = Some(bytes.to_vec());
        self
    }

    /// Write data and create the dataset. Infers shape from data length if not set.
    pub fn write<T: H5Type>(self, data: &[T]) -> Result<()> {
        let dtype = dtype_for_type::<T>()?;
        let fill_value = self.fill_value.clone();
        let fill = Self::fill_spec(
            fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = self.shape.unwrap_or_else(|| vec![data.len() as u64]);

        // Convert data to bytes
        let byte_ptr = data.as_ptr() as *const u8;
        let byte_len = data.len() * T::type_size();
        let data_bytes = unsafe { std::slice::from_raw_parts(byte_ptr, byte_len) };

        let spec = DatasetSpec {
            name: &self.name,
            shape: &shape,
            dtype,
            data: data_bytes,
        };

        if self.compact {
            self.writer
                .create_compact_dataset_with_fill(&self.parent, &spec, fill)?;
        } else if self.chunk_dims.is_some() || self.deflate_level.is_some() || self.shuffle {
            let chunk_dims = self.chunk_dims.unwrap_or_else(|| shape.clone());
            self.writer.create_chunked_dataset_with_fill(
                &self.parent,
                &spec,
                &chunk_dims,
                self.deflate_level,
                self.shuffle,
                fill,
            )?;
        } else {
            self.writer
                .create_dataset_with_fill(&self.parent, &spec, fill)?;
        }

        Ok(())
    }

    /// Write a scalar value.
    pub fn write_scalar<T: H5Type>(self, value: T) -> Result<()> {
        let dtype = dtype_for_type::<T>()?;
        let fill_value = self.fill_value.clone();
        let fill = Self::fill_spec(
            fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let byte_ptr = &value as *const T as *const u8;
        let data_bytes = unsafe { std::slice::from_raw_parts(byte_ptr, T::type_size()) };

        let spec = DatasetSpec {
            name: &self.name,
            shape: &[],
            dtype,
            data: data_bytes,
        };

        self.writer
            .create_dataset_with_fill(&self.parent, &spec, fill)?;
        Ok(())
    }

    /// Write fixed-length ASCII strings. Strings longer than `len` bytes are rejected.
    pub fn write_fixed_ascii_strings(self, data: &[&str], len: usize) -> Result<()> {
        self.write_fixed_strings(data, len, false)
    }

    /// Write fixed-length UTF-8 strings. Strings longer than `len` bytes are rejected.
    pub fn write_fixed_utf8_strings(self, data: &[&str], len: usize) -> Result<()> {
        self.write_fixed_strings(data, len, true)
    }

    /// Write variable-length UTF-8 strings using HDF5 global heap storage.
    pub fn write_vlen_utf8_strings(self, data: &[&str]) -> Result<()> {
        if self.compact || self.chunk_dims.is_some() || self.deflate_level.is_some() || self.shuffle
        {
            return Err(Error::Unsupported(
                "variable-length string writer currently supports contiguous storage only".into(),
            ));
        }
        if self.fill_value.is_some() {
            return Err(Error::Unsupported(
                "variable-length string fill values are not supported yet".into(),
            ));
        }
        let shape = self
            .shape
            .clone()
            .unwrap_or_else(|| vec![data.len() as u64]);
        self.writer
            .create_vlen_utf8_string_dataset(&self.parent, &self.name, &shape, data)?;
        Ok(())
    }

    fn write_fixed_strings(self, data: &[&str], len: usize, utf8: bool) -> Result<()> {
        if len > u32::MAX as usize {
            return Err(Error::InvalidFormat(
                "fixed string length exceeds u32".into(),
            ));
        }
        let dtype = if utf8 {
            DtypeSpec::FixedUtf8String {
                len: len as u32,
                padding: 1,
            }
        } else {
            DtypeSpec::FixedAsciiString {
                len: len as u32,
                padding: 1,
            }
        };
        let fill_value = self.fill_value.clone();
        let fill = Self::fill_spec(
            fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = self
            .shape
            .clone()
            .unwrap_or_else(|| vec![data.len() as u64]);
        let expected_count: u64 = shape.iter().product();
        if expected_count != data.len() as u64 {
            return Err(Error::InvalidFormat(format!(
                "fixed string data length {} does not match dataset shape element count {expected_count}",
                data.len()
            )));
        }

        let mut data_bytes = Vec::with_capacity(data.len() * len);
        for value in data {
            let bytes = value.as_bytes();
            if bytes.len() > len {
                return Err(Error::InvalidFormat(format!(
                    "fixed string value has {} bytes, maximum is {len}",
                    bytes.len()
                )));
            }
            data_bytes.extend_from_slice(bytes);
            data_bytes.resize(data_bytes.len() + (len - bytes.len()), 0);
        }

        let spec = DatasetSpec {
            name: &self.name,
            shape: &shape,
            dtype,
            data: &data_bytes,
        };

        if self.compact {
            self.writer
                .create_compact_dataset_with_fill(&self.parent, &spec, fill)?;
        } else {
            self.writer
                .create_dataset_with_fill(&self.parent, &spec, fill)?;
        }
        Ok(())
    }

    fn fill_spec(
        value: Option<&[u8]>,
        dtype_size: usize,
        alloc_time: u8,
        fill_time: u8,
    ) -> Result<Option<FillValueSpec<'_>>> {
        if let Some(value) = value {
            if value.len() != dtype_size {
                return Err(Error::InvalidFormat(format!(
                    "fill value has {} bytes, expected {} for dataset datatype",
                    value.len(),
                    dtype_size
                )));
            }
            Ok(Some(FillValueSpec::with_value(
                alloc_time, fill_time, value,
            )))
        } else if alloc_time != 1 || fill_time != 2 {
            Ok(Some(FillValueSpec::undefined(alloc_time, fill_time)))
        } else {
            Ok(None)
        }
    }
}

/// Map a Rust type to DtypeSpec.
pub(crate) fn dtype_for_type<T: H5Type>() -> Result<DtypeSpec> {
    let size = T::type_size();
    // Use TypeId to determine the exact type
    use std::any::TypeId;
    let id = TypeId::of::<T>();

    if id == TypeId::of::<f64>() {
        Ok(DtypeSpec::F64)
    } else if id == TypeId::of::<f32>() {
        Ok(DtypeSpec::F32)
    } else if id == TypeId::of::<i64>() {
        Ok(DtypeSpec::I64)
    } else if id == TypeId::of::<i32>() {
        Ok(DtypeSpec::I32)
    } else if id == TypeId::of::<i16>() {
        Ok(DtypeSpec::I16)
    } else if id == TypeId::of::<i8>() {
        Ok(DtypeSpec::I8)
    } else if id == TypeId::of::<u64>() {
        Ok(DtypeSpec::U64)
    } else if id == TypeId::of::<u32>() {
        Ok(DtypeSpec::U32)
    } else if id == TypeId::of::<u16>() {
        Ok(DtypeSpec::U16)
    } else if id == TypeId::of::<u8>() {
        Ok(DtypeSpec::U8)
    } else if let Some(fields) = T::compound_fields() {
        let mut out = Vec::with_capacity(fields.len());
        for field in fields {
            let dtype = match field.type_class {
                TypeClass::Integer { signed: true } => match field.size {
                    1 => DtypeSpec::I8,
                    2 => DtypeSpec::I16,
                    4 => DtypeSpec::I32,
                    8 => DtypeSpec::I64,
                    other => {
                        return Err(Error::Unsupported(format!(
                            "unsupported signed compound field size {other}"
                        )))
                    }
                },
                TypeClass::Integer { signed: false } => match field.size {
                    1 => DtypeSpec::U8,
                    2 => DtypeSpec::U16,
                    4 => DtypeSpec::U32,
                    8 => DtypeSpec::U64,
                    other => {
                        return Err(Error::Unsupported(format!(
                            "unsupported unsigned compound field size {other}"
                        )))
                    }
                },
                TypeClass::Float => match field.size {
                    4 => DtypeSpec::F32,
                    8 => DtypeSpec::F64,
                    other => {
                        return Err(Error::Unsupported(format!(
                            "unsupported floating compound field size {other}"
                        )))
                    }
                },
                TypeClass::Compound => {
                    return Err(Error::Unsupported(
                        "nested compound writer type descriptors are not supported".into(),
                    ))
                }
            };
            out.push(CompoundFieldSpec {
                name: field.name,
                offset: field.offset as u32,
                dtype,
            });
        }
        Ok(DtypeSpec::Compound {
            size: T::type_size() as u32,
            fields: out,
        })
    } else {
        Err(Error::Unsupported(format!(
            "unsupported type with size {size}"
        )))
    }
}
