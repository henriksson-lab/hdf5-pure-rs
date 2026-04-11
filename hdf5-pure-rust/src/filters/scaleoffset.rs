use crate::error::{Error, Result};

/// Decompress ScaleOffset-filtered data.
///
/// ScaleOffset stores integer data as: (value - min_value) with reduced bit width.
/// For integer types, the minimum value and the number of bits needed are stored
/// as parameters in the filter pipeline message.
///
/// Client data parameters (from filter pipeline):
/// [0] = scale_type (0=integer, 1=float dscale)
/// [1] = scale_factor (for float) or 0
/// [2] = number of elements
/// [3..] = minimum value bytes (for integer mode)
///
/// This is a simplified implementation for the integer case.
pub fn decompress(
    data: &[u8],
    element_size: usize,
    num_elements: usize,
    client_data: &[u32],
) -> Result<Vec<u8>> {
    if client_data.is_empty() {
        return Err(Error::InvalidFormat(
            "scaleoffset filter missing parameters".into(),
        ));
    }

    let scale_type = client_data[0];

    if scale_type != 0 {
        return Err(Error::Unsupported(format!(
            "scaleoffset scale_type {scale_type} (only integer mode supported)"
        )));
    }

    // For integer scaleoffset:
    // The compressed data starts with: min_bits(4 bytes) + min_value(element_size bytes) + packed_data
    if data.len() < 4 + element_size {
        return Err(Error::InvalidFormat("scaleoffset data too short".into()));
    }

    let min_bits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let pos = 4;

    // Read minimum value
    let mut min_value: u64 = 0;
    for i in 0..element_size.min(8) {
        min_value |= (data[pos + i] as u64) << (i * 8);
    }
    let pos = pos + element_size;

    if min_bits == 0 {
        // All values are the same (= min_value)
        let mut output = vec![0u8; num_elements * element_size];
        for elem in 0..num_elements {
            let start = elem * element_size;
            for i in 0..element_size.min(8) {
                output[start + i] = (min_value >> (i * 8)) as u8;
            }
        }
        return Ok(output);
    }

    // Unpack bit-packed offsets and add min_value
    let packed = &data[pos..];
    let mut output = vec![0u8; num_elements * element_size];
    let mut bit_pos = 0;

    for elem in 0..num_elements {
        let mut offset: u64 = 0;
        for bit in 0..min_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if byte_idx < packed.len() {
                let bit_val = (packed[byte_idx] >> bit_idx) & 1;
                offset |= (bit_val as u64) << bit;
            }
            bit_pos += 1;
        }

        let value = min_value.wrapping_add(offset);
        let start = elem * element_size;
        for i in 0..element_size.min(8) {
            output[start + i] = (value >> (i * 8)) as u8;
        }
    }

    Ok(output)
}
