pub mod error;
pub mod io;
pub mod format;
pub mod engine;
pub mod filters;
pub mod hl;

pub use error::{Error, Result};
pub use hl::file::File;
pub use hl::group::Group;
pub use hl::dataset::Dataset;
pub use hl::attribute::Attribute;
pub use hl::types::H5Type;
pub use hl::datatype::Datatype;
pub use hl::dataspace::Dataspace;
pub use hl::writable_file::WritableFile;
pub use hl::dataset_builder::DatasetBuilder;

/// Re-export the derive macro when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use hdf5_pure_rust_derive::H5Type as DeriveH5Type;
