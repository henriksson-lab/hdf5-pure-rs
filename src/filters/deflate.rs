use std::io::Read;

use flate2::read::ZlibDecoder;

use crate::error::{Error, Result};

/// Decompress deflate (zlib) compressed data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::InvalidFormat(format!("deflate decompression failed: {e}")))?;
    Ok(out)
}

/// Compress data with deflate at the given level (0-9).
pub fn compress(data: &[u8], level: u32) -> Result<Vec<u8>> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
    encoder
        .write_all(data)
        .map_err(|e| Error::InvalidFormat(format!("deflate compression failed: {e}")))?;
    encoder
        .finish()
        .map_err(|e| Error::InvalidFormat(format!("deflate compression finish failed: {e}")))
}
