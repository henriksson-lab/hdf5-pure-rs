use std::fs;
use std::path::{Path, PathBuf};

use crate::engine::writer::{AttrSpec, HdfFileWriter};
use crate::error::{Error, Result};
use crate::hl::dataset_builder::DatasetBuilder;
use crate::hl::types::H5Type;

/// A writable HDF5 file under construction.
///
/// Use the builder methods to create groups, datasets, and attributes,
/// then call `flush()` or `close()` to finalize and write to disk.
pub struct WritableFile {
    writer: HdfFileWriter<fs::File>,
    path: PathBuf,
    #[allow(dead_code)]
    current_group: String,
}

impl WritableFile {
    /// Create a new HDF5 file (truncating if it exists).
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = fs::File::create(&path)?;
        let mut writer = HdfFileWriter::new(f);
        writer.begin()?;
        writer.create_root_group()?;

        Ok(Self {
            writer,
            path,
            current_group: "/".to_string(),
        })
    }

    /// Create a new HDF5 file, failing if it already exists.
    pub fn create_excl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("file already exists: {}", path.display()),
            )));
        }
        Self::create(path)
    }

    /// Create a subgroup in the root group.
    pub fn create_group(&mut self, name: &str) -> Result<WritableGroup<'_>> {
        self.writer.create_group("/", name)?;
        let full_path = format!("/{name}");
        Ok(WritableGroup {
            writer: &mut self.writer,
            path: full_path,
        })
    }

    /// Get a builder for creating a dataset in the root group.
    pub fn new_dataset_builder(&mut self, name: &str) -> DatasetBuilder<'_> {
        DatasetBuilder::new(&mut self.writer, "/", name)
    }

    /// Add an attribute to the root group.
    pub fn add_attr<T: H5Type>(&mut self, name: &str, value: T) -> Result<()> {
        let dtype = crate::hl::dataset_builder::dtype_for_type::<T>()?;
        let byte_ptr = &value as *const T as *const u8;
        let data = unsafe { std::slice::from_raw_parts(byte_ptr, T::type_size()) };
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &[],
            dtype,
            data,
        });
        Ok(())
    }

    /// Create a soft link in the root group.
    pub fn link_soft(&mut self, name: &str, target_path: &str) {
        self.writer.create_soft_link("/", name, target_path);
    }

    /// Create an external link in the root group.
    pub fn link_external(&mut self, name: &str, filename: &str, obj_path: &str) {
        self.writer
            .create_external_link("/", name, filename, obj_path);
    }

    /// Finalize and close the file. Returns a read-only File handle.
    pub fn close(mut self) -> Result<crate::hl::file::File> {
        self.writer.finalize()?;
        crate::hl::file::File::open(&self.path)
    }

    /// Finalize the file (writes superblock and all metadata).
    pub fn flush(&mut self) -> Result<()> {
        self.writer.finalize()
    }
}

/// A writable group within a WritableFile.
pub struct WritableGroup<'a> {
    writer: &'a mut HdfFileWriter<fs::File>,
    path: String,
}

impl<'a> WritableGroup<'a> {
    /// Create a subgroup.
    pub fn create_group(&mut self, name: &str) -> Result<WritableGroup<'_>> {
        self.writer.create_group(&self.path, name)?;
        let full_path = format!("{}/{name}", self.path);
        Ok(WritableGroup {
            writer: self.writer,
            path: full_path,
        })
    }

    /// Get a builder for creating a dataset in this group.
    pub fn new_dataset_builder(&mut self, name: &str) -> DatasetBuilder<'_> {
        DatasetBuilder::new(self.writer, &self.path, name)
    }

    /// Create a soft link in this group.
    pub fn link_soft(&mut self, name: &str, target_path: &str) {
        self.writer.create_soft_link(&self.path, name, target_path);
    }

    /// Create an external link in this group.
    pub fn link_external(&mut self, name: &str, filename: &str, obj_path: &str) {
        self.writer
            .create_external_link(&self.path, name, filename, obj_path);
    }
}
