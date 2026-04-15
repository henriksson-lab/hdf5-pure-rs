use crate::error::{Error, Result};

const PARM_SCALETYPE: usize = 0;
const PARM_NELMTS: usize = 2;
const PARM_CLASS: usize = 3;
const PARM_SIZE: usize = 4;
const PARM_SIGN: usize = 5;
const PARM_ORDER: usize = 6;
const PARM_FILAVAIL: usize = 7;
const PARM_FILVAL: usize = 8;

const CLS_INTEGER: u32 = 0;
const CLS_FLOAT: u32 = 1;
const SIGN_UNSIGNED: u32 = 0;
const SIGN_TWOS: u32 = 1;
const ORDER_LE: u32 = 0;
const ORDER_BE: u32 = 1;
const HEADER_LEN: usize = 21;

#[derive(Debug, Clone, Copy)]
struct Parms {
    size: usize,
    minbits: usize,
    order: u32,
}

/// Decompress HDF5 ScaleOffset-filtered chunks using the datatype-aware
/// parameters stored in the filter pipeline.
pub fn decompress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    if client_data.len() <= PARM_ORDER {
        return Err(Error::InvalidFormat(
            "scaleoffset filter missing datatype parameters".into(),
        ));
    }

    let scale_type = client_data[PARM_SCALETYPE];
    let nelmts = client_data[PARM_NELMTS] as usize;
    let class = client_data[PARM_CLASS];
    let size = client_data[PARM_SIZE] as usize;
    let sign = client_data[PARM_SIGN];
    let order = client_data[PARM_ORDER];

    if size == 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset datatype size is zero".into(),
        ));
    }
    if order != ORDER_LE && order != ORDER_BE {
        return Err(Error::InvalidFormat(format!(
            "invalid scaleoffset byte order {order}"
        )));
    }
    if data.len() < HEADER_LEN {
        return Err(Error::InvalidFormat("scaleoffset data too short".into()));
    }

    let minbits = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let minval_size = data[4] as usize;
    if minval_size > 16 || data.len() < 5 + minval_size {
        return Err(Error::InvalidFormat(
            "invalid scaleoffset minimum value header".into(),
        ));
    }
    let minval = read_le_u128(&data[5..5 + minval_size]);

    let out_len = nelmts
        .checked_mul(size)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset output size overflow".into()))?;
    let mut out = vec![0u8; out_len];

    if minbits == size * 8 {
        let raw = data.get(HEADER_LEN..HEADER_LEN + out_len).ok_or_else(|| {
            Error::InvalidFormat("scaleoffset full-precision data too short".into())
        })?;
        out.copy_from_slice(raw);
    } else if minbits != 0 {
        let parms = Parms {
            size,
            minbits,
            order,
        };
        let mut stream = BitStream::new(&data[HEADER_LEN..]);
        for idx in 0..nelmts {
            decompress_atomic(&mut out, idx * size, &mut stream, parms)?;
        }
    }

    match class {
        CLS_INTEGER => {
            let fill = if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
                Some(read_fill_value(client_data, size, order))
            } else {
                None
            };
            postprocess_integer(&mut out, size, sign, order, minbits, minval, fill)?
        }
        CLS_FLOAT if scale_type == 0 => {
            postprocess_float(&mut out, size, order, minbits, minval, client_data)?
        }
        CLS_FLOAT => {
            return Err(Error::Unsupported(format!(
                "scaleoffset float scale type {scale_type}"
            )));
        }
        other => {
            return Err(Error::Unsupported(format!(
                "scaleoffset datatype class {other}"
            )));
        }
    }

    Ok(out)
}

fn decompress_atomic(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: Parms,
) -> Result<()> {
    let dtype_bits = parms.size * 8;
    if parms.minbits == 0 || parms.minbits > dtype_bits {
        return Err(Error::InvalidFormat(
            "invalid scaleoffset minimum bit count".into(),
        ));
    }

    if parms.order == ORDER_LE {
        let begin = parms.size - 1 - (dtype_bits - parms.minbits) / 8;
        for k in (0..=begin).rev() {
            decompress_byte(out, data_offset, k, begin, stream, parms, dtype_bits)?;
        }
    } else {
        let begin = (dtype_bits - parms.minbits) / 8;
        for k in begin..parms.size {
            decompress_byte(out, data_offset, k, begin, stream, parms, dtype_bits)?;
        }
    }
    Ok(())
}

fn decompress_byte(
    out: &mut [u8],
    data_offset: usize,
    k: usize,
    begin: usize,
    stream: &mut BitStream<'_>,
    parms: Parms,
    dtype_bits: usize,
) -> Result<()> {
    let bits_to_copy = if k == begin {
        8 - (dtype_bits - parms.minbits) % 8
    } else {
        8
    };
    let bits = stream.read_bits(bits_to_copy)? as u8;
    out[data_offset + k] = bits;
    Ok(())
}

fn postprocess_integer(
    out: &mut [u8],
    size: usize,
    sign: u32,
    order: u32,
    minbits: usize,
    minval: u128,
    fill: Option<u128>,
) -> Result<()> {
    let fill_marker = if minbits > 0 && minbits < 128 {
        Some((1u128 << minbits) - 1)
    } else if minbits == 128 {
        Some(u128::MAX)
    } else {
        None
    };

    for chunk in out.chunks_exact_mut(size) {
        let value = read_uint(chunk, order);
        let value = if let (Some(fill), Some(marker)) = (fill, fill_marker) {
            if value == marker {
                fill
            } else {
                minval.wrapping_add(value)
            }
        } else if minbits == 0 {
            minval
        } else {
            minval.wrapping_add(value)
        };
        write_uint(chunk, order, value);

        if sign != SIGN_UNSIGNED && sign != SIGN_TWOS {
            return Err(Error::InvalidFormat(format!(
                "invalid scaleoffset integer sign {sign}"
            )));
        }
    }
    Ok(())
}

fn postprocess_float(
    out: &mut [u8],
    size: usize,
    order: u32,
    minbits: usize,
    minval: u128,
    client_data: &[u32],
) -> Result<()> {
    let scale = client_data
        .get(1)
        .copied()
        .ok_or_else(|| Error::InvalidFormat("scaleoffset missing scale factor".into()))?
        as i32;
    let divisor = 10f64.powi(scale);
    let marker = if minbits > 0 && minbits < 128 {
        Some((1u128 << minbits) - 1)
    } else {
        None
    };
    let fill = if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
        Some(read_fill_value(client_data, size, order))
    } else {
        None
    };

    match size {
        4 => {
            let min = f32::from_le_bytes((minval as u32).to_le_bytes()) as f64;
            let fill = fill.map(|v| f32::from_le_bytes((v as u32).to_le_bytes()));
            for chunk in out.chunks_exact_mut(size) {
                let packed = read_uint(chunk, order);
                let value = if let (Some(marker), Some(fill)) = (marker, fill) {
                    if packed == marker {
                        fill
                    } else {
                        (packed as i64 as f64 / divisor + min) as f32
                    }
                } else {
                    (packed as i64 as f64 / divisor + min) as f32
                };
                write_float32(chunk, order, value);
            }
        }
        8 => {
            let min = f64::from_le_bytes((minval as u64).to_le_bytes());
            let fill = fill.map(|v| f64::from_le_bytes((v as u64).to_le_bytes()));
            for chunk in out.chunks_exact_mut(size) {
                let packed = read_uint(chunk, order);
                let value = if let (Some(marker), Some(fill)) = (marker, fill) {
                    if packed == marker {
                        fill
                    } else {
                        packed as i64 as f64 / divisor + min
                    }
                } else {
                    packed as i64 as f64 / divisor + min
                };
                write_float64(chunk, order, value);
            }
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "scaleoffset floating-point size {size}"
            )));
        }
    }

    Ok(())
}

fn read_uint(bytes: &[u8], order: u32) -> u128 {
    let mut value = 0u128;
    if order == ORDER_LE {
        for (idx, byte) in bytes.iter().take(16).enumerate() {
            value |= (*byte as u128) << (idx * 8);
        }
    } else {
        for byte in bytes.iter().take(16) {
            value = (value << 8) | (*byte as u128);
        }
    }
    value
}

fn write_uint(bytes: &mut [u8], order: u32, value: u128) {
    if order == ORDER_LE {
        for (idx, byte) in bytes.iter_mut().take(16).enumerate() {
            *byte = (value >> (idx * 8)) as u8;
        }
    } else {
        let n = bytes.len().min(16);
        for (idx, byte) in bytes.iter_mut().take(n).enumerate() {
            *byte = (value >> ((n - idx - 1) * 8)) as u8;
        }
    }
}

fn read_le_u128(bytes: &[u8]) -> u128 {
    let mut value = 0u128;
    for (idx, byte) in bytes.iter().take(16).enumerate() {
        value |= (*byte as u128) << (idx * 8);
    }
    value
}

fn read_fill_value(client_data: &[u32], size: usize, order: u32) -> u128 {
    let mut raw = vec![0u8; size];
    let mut pos = 0usize;
    for value in client_data.iter().skip(PARM_FILVAL) {
        let bytes = value.to_le_bytes();
        for byte in bytes {
            if pos < raw.len() {
                raw[pos] = byte;
                pos += 1;
            }
        }
    }
    read_uint(&raw, order)
}

fn write_float32(bytes: &mut [u8], order: u32, value: f32) {
    let raw = if order == ORDER_LE {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    bytes[..4].copy_from_slice(&raw);
}

fn write_float64(bytes: &mut [u8], order: u32, value: f64) {
    let raw = if order == ORDER_LE {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    bytes[..8].copy_from_slice(&raw);
}

struct BitStream<'a> {
    data: &'a [u8],
    byte: usize,
    bits_left: usize,
}

impl<'a> BitStream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte: 0,
            bits_left: 8,
        }
    }

    fn read_bits(&mut self, mut nbits: usize) -> Result<u16> {
        if nbits > 16 {
            return Err(Error::InvalidFormat("scaleoffset bit run too long".into()));
        }

        let mut value = 0u16;
        while nbits > 0 {
            let byte = *self
                .data
                .get(self.byte)
                .ok_or_else(|| Error::InvalidFormat("scaleoffset data too short".into()))?;
            let take = self.bits_left.min(nbits);
            let shift = self.bits_left - take;
            let mask = if take == 8 {
                0xff
            } else {
                ((1u16 << take) - 1) as u8
            };
            value = (value << take) | (((byte >> shift) & mask) as u16);
            self.bits_left -= take;
            nbits -= take;
            if self.bits_left == 0 {
                self.byte += 1;
                self.bits_left = 8;
            }
        }
        Ok(value)
    }
}
