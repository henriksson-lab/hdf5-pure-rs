use std::fs;

use crate::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use crate::error::{Error, Result};
use crate::hl::types::H5Type;

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
}

impl<'a> DatasetBuilder<'a> {
    pub(crate) fn new(
        writer: &'a mut HdfFileWriter<fs::File>,
        parent: &str,
        name: &str,
    ) -> Self {
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

    /// Write data and create the dataset. Infers shape from data length if not set.
    pub fn write<T: H5Type>(self, data: &[T]) -> Result<()> {
        let dtype = dtype_for_type::<T>()?;
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
            self.writer.create_compact_dataset(&self.parent, &spec)?;
        } else if self.chunk_dims.is_some() || self.deflate_level.is_some() || self.shuffle {
            let chunk_dims = self.chunk_dims.unwrap_or_else(|| shape.clone());
            self.writer.create_chunked_dataset(
                &self.parent,
                &spec,
                &chunk_dims,
                self.deflate_level,
                self.shuffle,
            )?;
        } else {
            self.writer.create_dataset(&self.parent, &spec)?;
        }

        Ok(())
    }

    /// Write a scalar value.
    pub fn write_scalar<T: H5Type>(self, value: T) -> Result<()> {
        let dtype = dtype_for_type::<T>()?;
        let byte_ptr = &value as *const T as *const u8;
        let data_bytes = unsafe { std::slice::from_raw_parts(byte_ptr, T::type_size()) };

        let spec = DatasetSpec {
            name: &self.name,
            shape: &[],
            dtype,
            data: data_bytes,
        };

        self.writer.create_dataset(&self.parent, &spec)?;
        Ok(())
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
    } else {
        Err(Error::Unsupported(format!(
            "unsupported type with size {size}"
        )))
    }
}
