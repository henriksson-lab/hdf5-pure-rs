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

#[cfg(all(test, feature = "blosc"))]
mod tests {
    use super::*;

    #[test]
    fn blosc_feature_decompresses_blosc2_frame() {
        let data = (0..128u32)
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let params = blosc2_pure_rs::compress::CParams {
            typesize: 4,
            ..Default::default()
        };
        let compressed = blosc2_pure_rs::compress::compress(&data, &params).unwrap();
        let decoded = decompress(&compressed).unwrap();
        assert_eq!(decoded, data);
    }
}
