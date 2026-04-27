use std::io::Read;

use flate2::read::ZlibDecoder;

use crate::error::{Error, Result};

/// Decompress deflate (zlib) compressed data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    decompress_with_hint(data, None)
}

/// Decompress deflate (zlib) compressed data using an optional capacity hint
/// for the output buffer.
pub fn decompress_with_hint(data: &[u8], expected_len: Option<usize>) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::with_capacity(expected_len.unwrap_or(0));
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::InvalidFormat(format!("deflate decompression failed: {e}")))?;
    Ok(out)
}

/// Decompress deflate (zlib) compressed data into an exactly-sized buffer.
pub fn decompress_exact(data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; expected_len];
    decompress_exact_into(data, &mut out)?;
    Ok(out)
}

/// Decompress deflate (zlib) compressed data into the provided output buffer
/// and require the decoded size to match exactly.
pub fn decompress_exact_into(data: &[u8], out: &mut [u8]) -> Result<()> {
    let mut decoder = ZlibDecoder::new(data);
    decoder
        .read_exact(out)
        .map_err(|e| Error::InvalidFormat(format!("deflate decompression failed: {e}")))?;
    let mut tail = [0u8; 1];
    match decoder.read(&mut tail) {
        Ok(0) => Ok(()),
        Ok(_) => Err(Error::InvalidFormat(
            "deflate decompression produced more bytes than expected".into(),
        )),
        Err(e) => Err(Error::InvalidFormat(format!(
            "deflate decompression tail check failed: {e}"
        ))),
    }
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
