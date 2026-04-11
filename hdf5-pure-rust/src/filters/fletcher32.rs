use crate::error::{Error, Result};

/// Verify Fletcher32 checksum and strip it from the data.
/// The last 4 bytes of the data are the checksum (stored little-endian).
pub fn verify_and_strip(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(
            "data too short for fletcher32 checksum".into(),
        ));
    }

    let payload = &data[..data.len() - 4];
    // Stored checksum is little-endian (UINT32ENCODE in HDF5 C library)
    let stored = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);

    let computed = fletcher32(payload);

    // HDF5 also checks a byte-swapped version for compatibility with pre-1.6.3
    let reversed = fletcher32_reversed(computed);

    if stored != computed && stored != reversed {
        return Err(Error::InvalidFormat(format!(
            "fletcher32 checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }

    Ok(payload.to_vec())
}

/// Compute Fletcher32 checksum matching the HDF5 C library implementation.
/// Data is processed as big-endian 16-bit words.
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let len_words = data.len() / 2;
    let mut pos = 0;
    let mut remaining = len_words;

    while remaining > 0 {
        // Process in batches of 360 to avoid overflow
        let tlen = remaining.min(360);
        remaining -= tlen;

        for _ in 0..tlen {
            // Big-endian 16-bit word (matching HDF5 C library)
            let val = ((data[pos] as u32) << 8) | (data[pos + 1] as u32);
            sum1 += val;
            sum2 += sum1;
            pos += 2;
        }

        // Ones-complement reduction
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Handle odd byte
    if data.len() % 2 != 0 {
        sum1 += (data[pos] as u32) << 8;
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Final reduction
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

/// Compute the reversed (byte-swapped) checksum for pre-1.6.3 compatibility.
fn fletcher32_reversed(checksum: u32) -> u32 {
    let bytes = checksum.to_ne_bytes();
    u32::from_ne_bytes([bytes[1], bytes[0], bytes[3], bytes[2]])
}
