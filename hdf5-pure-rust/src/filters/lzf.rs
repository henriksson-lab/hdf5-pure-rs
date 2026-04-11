use crate::error::{Error, Result};

/// Decompress LZF-compressed data.
///
/// LZF is a very fast, low-ratio compression algorithm.
/// Format: sequence of literal runs and back-references.
///
/// Each chunk starts with a control byte:
/// - If high 3 bits == 0: literal run of (control + 1) bytes follows
/// - Otherwise: back-reference: length = high 3 bits + 2 (or read next byte for long match),
///   offset = ((control & 0x1f) << 8) | next_byte + 1
pub fn decompress(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(expected_size);
    let mut ip = 0; // input position

    while ip < data.len() {
        let ctrl = data[ip] as usize;
        ip += 1;

        if ctrl < 32 {
            // Literal run: copy (ctrl + 1) bytes
            let count = ctrl + 1;
            if ip + count > data.len() {
                return Err(Error::InvalidFormat("lzf: literal run exceeds input".into()));
            }
            output.extend_from_slice(&data[ip..ip + count]);
            ip += count;
        } else {
            // Back-reference
            let mut length = (ctrl >> 5) as usize;
            let mut offset = ((ctrl & 0x1f) as usize) << 8;

            if length == 7 {
                // Long match: read additional length byte
                if ip >= data.len() {
                    return Err(Error::InvalidFormat("lzf: unexpected end in long match".into()));
                }
                length += data[ip] as usize;
                ip += 1;
            }
            length += 2; // minimum match length is 2

            if ip >= data.len() {
                return Err(Error::InvalidFormat("lzf: unexpected end reading offset".into()));
            }
            offset += data[ip] as usize + 1;
            ip += 1;

            if offset > output.len() {
                return Err(Error::InvalidFormat(format!(
                    "lzf: back-reference offset {} exceeds output size {}",
                    offset,
                    output.len()
                )));
            }

            let start = output.len() - offset;
            for i in 0..length {
                let byte = output[start + i];
                output.push(byte);
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lzf_literal_only() {
        // Control byte 0x04 = literal run of 5 bytes
        let compressed = vec![0x04, b'H', b'e', b'l', b'l', b'o'];
        let result = decompress(&compressed, 5).unwrap();
        assert_eq!(result, b"Hello");
    }

    #[test]
    fn test_lzf_with_backref() {
        // "abcabc" = literal "abc" + backref to position 0 length 3
        // literal: ctrl=2 (3-1), then 'a','b','c'
        // backref: length=3 (1 in high bits = 3-2=1, shifted = 0x20), offset=3
        //   ctrl = (1 << 5) | 0 = 0x20, next_byte = 2 (offset=0*256+2+1=3)
        let compressed = vec![0x02, b'a', b'b', b'c', 0x20, 0x02];
        let result = decompress(&compressed, 6).unwrap();
        assert_eq!(result, b"abcabc");
    }
}
