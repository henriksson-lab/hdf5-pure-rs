use crate::error::{Error, Result};

/// Decompress NBit-filtered data.
///
/// NBit stores only the significant bits of each element.
/// The client data parameters encode the datatype information needed
/// for unpacking. For simple integer types, the precision is stored
/// in the datatype message.
///
/// This is a simplified implementation that handles the common case
/// of integer types where bit_precision < 8 * element_size.
pub fn decompress(
    data: &[u8],
    element_size: usize,
    bit_precision: usize,
    num_elements: usize,
) -> Result<Vec<u8>> {
    if bit_precision == 0 || bit_precision > element_size * 8 {
        // No actual compression or invalid precision -- return as-is
        return Ok(data.to_vec());
    }

    if bit_precision == element_size * 8 {
        // Full precision stored -- no decompression needed
        return Ok(data.to_vec());
    }

    let total_bits = num_elements * bit_precision;
    let expected_bytes = (total_bits + 7) / 8;

    if data.len() < expected_bytes {
        return Err(Error::InvalidFormat(format!(
            "nbit data too short: {} bytes for {} elements at {} bits",
            data.len(), num_elements, bit_precision
        )));
    }

    let mut output = vec![0u8; num_elements * element_size];
    let mut bit_pos = 0;

    for elem in 0..num_elements {
        let mut value: u64 = 0;
        for bit in 0..bit_precision {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if byte_idx < data.len() {
                let bit_val = (data[byte_idx] >> bit_idx) & 1;
                value |= (bit_val as u64) << bit;
            }
            bit_pos += 1;
        }

        // Write value to output in little-endian
        let start = elem * element_size;
        for i in 0..element_size.min(8) {
            output[start + i] = (value >> (i * 8)) as u8;
        }
    }

    Ok(output)
}
