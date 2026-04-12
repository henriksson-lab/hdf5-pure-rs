use std::fmt;
use std::io;

/// The main error type for hdf5-pure-rust.
#[derive(Debug)]
pub enum Error {
    /// An I/O error occurred.
    Io(io::Error),
    /// Invalid HDF5 file format.
    InvalidFormat(String),
    /// Unsupported HDF5 feature or version.
    Unsupported(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::InvalidFormat(msg) => write!(f, "Invalid HDF5 format: {msg}"),
            Error::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// Result type alias for hdf5-pure-rust.
pub type Result<T> = std::result::Result<T, Error>;
