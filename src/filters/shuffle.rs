use crate::error::Result;

/// Unshuffle bytes (reverse the shuffle filter).
///
/// Shuffle rearranges bytes so that byte 0 of all elements comes first,
/// then byte 1 of all elements, etc. Unshuffle reverses this.
pub fn unshuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>> {
    if element_size <= 1 || data.is_empty() {
        return Ok(data.to_vec());
    }

    let n_elements = data.len() / element_size;
    let mut out = vec![0u8; data.len()];

    for i in 0..n_elements {
        for j in 0..element_size {
            out[i * element_size + j] = data[j * n_elements + i];
        }
    }

    Ok(out)
}

/// Shuffle bytes for compression.
pub fn shuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>> {
    if element_size <= 1 || data.is_empty() {
        return Ok(data.to_vec());
    }

    let n_elements = data.len() / element_size;
    let mut out = vec![0u8; data.len()];

    for i in 0..n_elements {
        for j in 0..element_size {
            out[j * n_elements + i] = data[i * element_size + j];
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_roundtrip() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 elements of 4 bytes each
        let shuffled = shuffle(&data, 4).unwrap();
        let unshuffled = unshuffle(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, data);
    }
}
