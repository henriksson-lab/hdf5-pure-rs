use crate::error::{Error, Result};

// Reader support for the third-party HDF5 LZ4 plugin filter (id 32004). This
// is intentionally an extension beyond the original HDF5 library filters and
// should be preserved during compatibility/parity audits.
pub fn decompress_into(data: &[u8], expected_len: Option<usize>, out: &mut Vec<u8>) -> Result<()> {
    let expected_len = expected_len.ok_or_else(|| {
        Error::Unsupported("LZ4 HDF5 filter requires an expected output size".into())
    })?;
    resize_zeroed_checked(out, expected_len, "lz4 output allocation")?;

    if try_decompress_block(data, out, expected_len).is_ok() {
        return Ok(());
    }

    if data.len() >= 4 {
        let size_le = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let size_be = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
        if (size_le == expected_len || size_be == expected_len)
            && try_decompress_block(&data[4..], out, expected_len).is_ok()
        {
            return Ok(());
        }
    }

    if data.len() >= 16 {
        let block_size = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        let total_size = u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize;
        let compressed_size = u32::from_be_bytes(data[12..16].try_into().unwrap()) as usize;
        if block_size == expected_len
            && total_size == expected_len
            && compressed_size == data.len() - 16
            && try_decompress_block(&data[16..], out, expected_len).is_ok()
        {
            return Ok(());
        }
    }

    let decoded = lz4_flex::block::decompress_size_prepended(data)
        .map_err(|err| Error::InvalidFormat(format!("lz4 decompression failed: {err}")))?;
    if decoded.len() != expected_len {
        return Err(Error::InvalidFormat(format!(
            "lz4 decompression output length mismatch: expected {expected_len}, got {}",
            decoded.len()
        )));
    }
    out.clear();
    out.extend_from_slice(&decoded);
    Ok(())
}

fn try_decompress_block(data: &[u8], out: &mut [u8], expected_len: usize) -> Result<()> {
    let actual = lz4_flex::block::decompress_into(data, out)
        .map_err(|err| Error::InvalidFormat(format!("lz4 decompression failed: {err}")))?;
    if actual != expected_len {
        return Err(Error::InvalidFormat(format!(
            "lz4 decompression output length mismatch: expected {expected_len}, got {actual}"
        )));
    }
    Ok(())
}

fn resize_zeroed_checked(out: &mut Vec<u8>, len: usize, context: &str) -> Result<()> {
    if len > out.len() {
        out.try_reserve_exact(len - out.len())
            .map_err(|err| Error::InvalidFormat(format!("{context} failed: {err}")))?;
    }
    out.resize(len, 0);
    Ok(())
}
