pub mod engine;
pub mod error;
pub mod filters;
pub mod format;
pub mod hl;
pub mod io;

pub use error::{Error, Result};
pub use hl::attribute::Attribute;
pub use hl::dataset::{Dataset, VdsView};
pub use hl::dataset_builder::DatasetBuilder;
pub use hl::dataspace::Dataspace;
pub use hl::datatype::Datatype;
pub use hl::file::File;
pub use hl::group::Group;
pub use hl::mutable_file::MutableFile;
pub use hl::types::H5Type;
pub use hl::value::H5Value;
pub use hl::writable_file::WritableFile;

/// Re-export the derive macro when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use hdf5_pure_rust_derive::H5Type as DeriveH5Type;
