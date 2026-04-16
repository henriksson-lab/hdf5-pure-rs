use crate::error::{Error, Result};

/// Decompress SZip-compressed data.
///
/// SZip (also known as "EC" or "NN" mode) is a patented compression algorithm
/// originally developed for scientific data. A full implementation requires
/// significant complexity (entropy coding + nearest-neighbor preprocessing).
///
/// This stub returns an error directing users to use the C HDF5 library for
/// SZip-compressed datasets, or to re-save without SZip.
pub fn decompress(_data: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unsupported(
        "SZip decompression not available in pure-Rust mode. \
         Re-save the dataset with deflate compression, or use the C HDF5 library."
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn szip_error_surface_is_stable() {
        let err = decompress(b"not-szip").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported: SZip decompression not available in pure-Rust mode. Re-save the dataset with deflate compression, or use the C HDF5 library."
        );
    }
}
