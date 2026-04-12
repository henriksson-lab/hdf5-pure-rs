use crate::error::{Error, Result};

/// Decompress Blosc-compressed data.
///
/// The HDF5 Blosc filter (ID 32001) stores each chunk as a complete Blosc frame.
/// This delegates to the `blosc2` crate for decompression.
#[cfg(feature = "blosc")]
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    blosc2_pure_rs::compress::decompress(data)
        .map_err(|e| Error::InvalidFormat(format!("blosc decompression failed: {e}")))
}

/// Blosc decompression stub when the `blosc` feature is not enabled.
#[cfg(not(feature = "blosc"))]
pub fn decompress(_data: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unsupported(
        "Blosc decompression requires the 'blosc' feature. \
         Enable it with: hdf5-pure-rust = { features = [\"blosc\"] }"
            .into(),
    ))
}
