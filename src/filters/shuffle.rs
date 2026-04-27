use crate::error::Result;

/// Unshuffle bytes (reverse the shuffle filter).
///
/// Shuffle rearranges bytes so that byte 0 of all elements comes first,
/// then byte 1 of all elements, etc. Unshuffle reverses this.
pub fn unshuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; data.len()];
    unshuffle_into(data, element_size, &mut out)?;
    Ok(out)
}

/// Unshuffle bytes into a provided output buffer.
pub fn unshuffle_into(data: &[u8], element_size: usize, out: &mut [u8]) -> Result<()> {
    if out.len() != data.len() {
        return Err(crate::error::Error::InvalidFormat(
            "shuffle output length mismatch".into(),
        ));
    }
    if element_size <= 1 || data.is_empty() {
        out.copy_from_slice(data);
        return Ok(());
    }

    let n_elements = data.len() / element_size;
    let grouped = n_elements * element_size;
    match element_size {
        8 => {
            let p0 = &data[0..n_elements];
            let p1 = &data[n_elements..2 * n_elements];
            let p2 = &data[2 * n_elements..3 * n_elements];
            let p3 = &data[3 * n_elements..4 * n_elements];
            let p4 = &data[4 * n_elements..5 * n_elements];
            let p5 = &data[5 * n_elements..6 * n_elements];
            let p6 = &data[6 * n_elements..7 * n_elements];
            let p7 = &data[7 * n_elements..8 * n_elements];
            for (i, elem) in out[..grouped].chunks_exact_mut(8).enumerate() {
                elem[0] = p0[i];
                elem[1] = p1[i];
                elem[2] = p2[i];
                elem[3] = p3[i];
                elem[4] = p4[i];
                elem[5] = p5[i];
                elem[6] = p6[i];
                elem[7] = p7[i];
            }
        }
        4 => {
            let p0 = &data[0..n_elements];
            let p1 = &data[n_elements..2 * n_elements];
            let p2 = &data[2 * n_elements..3 * n_elements];
            let p3 = &data[3 * n_elements..4 * n_elements];
            for (i, elem) in out[..grouped].chunks_exact_mut(4).enumerate() {
                elem[0] = p0[i];
                elem[1] = p1[i];
                elem[2] = p2[i];
                elem[3] = p3[i];
            }
        }
        2 => {
            let p0 = &data[0..n_elements];
            let p1 = &data[n_elements..2 * n_elements];
            for (i, elem) in out[..grouped].chunks_exact_mut(2).enumerate() {
                elem[0] = p0[i];
                elem[1] = p1[i];
            }
        }
        _ => {
            for i in 0..n_elements {
                for j in 0..element_size {
                    out[i * element_size + j] = data[j * n_elements + i];
                }
            }
        }
    }
    out[grouped..].copy_from_slice(&data[grouped..]);
    Ok(())
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
    let grouped = n_elements * element_size;
    out[grouped..].copy_from_slice(&data[grouped..]);

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

    #[test]
    fn test_shuffle_roundtrip_preserves_trailing_bytes() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shuffled = shuffle(&data, 4).unwrap();
        let unshuffled = unshuffle(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, data);
    }
}
