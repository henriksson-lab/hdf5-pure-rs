use crate::error::{Error, Result};

const NBIT_ATOMIC: u32 = 1;
const NBIT_ARRAY: u32 = 2;
const NBIT_COMPOUND: u32 = 3;
const NBIT_NOOPTYPE: u32 = 4;
const NBIT_ORDER_LE: u32 = 0;
const NBIT_ORDER_BE: u32 = 1;

#[derive(Debug, Clone, Copy)]
struct AtomicParms {
    size: usize,
    order: u32,
    precision: usize,
    offset: usize,
}

/// Decompress HDF5 NBit-filtered data using the datatype-aware filter
/// parameters stored by HDF5's set-local callback.
pub fn decompress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    if client_data.len() < 5 {
        return Err(Error::InvalidFormat(
            "nbit filter missing datatype parameters".into(),
        ));
    }

    let nparams = client_data[0] as usize;
    if nparams != client_data.len() {
        return Err(Error::InvalidFormat(format!(
            "nbit parameter count mismatch: header says {nparams}, got {}",
            client_data.len()
        )));
    }

    if client_data[1] != 0 {
        return Ok(data.to_vec());
    }

    let nelmts = client_data[2] as usize;
    let dtype_size = client_data[4] as usize;
    let mut out =
        vec![
            0u8;
            nelmts
                .checked_mul(dtype_size)
                .ok_or_else(|| Error::InvalidFormat("nbit output size overflow".into()))?
        ];
    let mut stream = BitStream::new(data);

    match client_data[3] {
        NBIT_ATOMIC => {
            let parms = AtomicParms {
                size: dtype_size,
                order: *client_data
                    .get(5)
                    .ok_or_else(|| Error::InvalidFormat("nbit missing byte order".into()))?,
                precision: *client_data
                    .get(6)
                    .ok_or_else(|| Error::InvalidFormat("nbit missing precision".into()))?
                    as usize,
                offset: *client_data
                    .get(7)
                    .ok_or_else(|| Error::InvalidFormat("nbit missing bit offset".into()))?
                    as usize,
            };
            validate_atomic(parms)?;
            for idx in 0..nelmts {
                decompress_atomic(&mut out, idx * parms.size, &mut stream, parms)?;
            }
        }
        NBIT_ARRAY => {
            for idx in 0..nelmts {
                let mut pidx = 4usize;
                decompress_array(
                    &mut out,
                    idx * dtype_size,
                    &mut stream,
                    client_data,
                    &mut pidx,
                )?;
            }
        }
        NBIT_COMPOUND => {
            for idx in 0..nelmts {
                let mut pidx = 4usize;
                decompress_compound(
                    &mut out,
                    idx * dtype_size,
                    &mut stream,
                    client_data,
                    &mut pidx,
                )?;
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "nbit datatype class parameter {other}"
            )));
        }
    }

    Ok(out)
}

fn decompress_array(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: &[u32],
    pidx: &mut usize,
) -> Result<()> {
    let total_size = take(parms, pidx)? as usize;
    let base_class = take(parms, pidx)?;

    match base_class {
        NBIT_ATOMIC => {
            let p = AtomicParms {
                size: take(parms, pidx)? as usize,
                order: take(parms, pidx)?,
                precision: take(parms, pidx)? as usize,
                offset: take(parms, pidx)? as usize,
            };
            validate_atomic(p)?;
            let count = total_size / p.size;
            for idx in 0..count {
                decompress_atomic(out, data_offset + idx * p.size, stream, p)?;
            }
        }
        NBIT_ARRAY | NBIT_COMPOUND => {
            let base_size = *parms
                .get(*pidx)
                .ok_or_else(|| Error::InvalidFormat("nbit missing nested size".into()))?
                as usize;
            let count = total_size / base_size;
            let begin = *pidx;
            for idx in 0..count {
                *pidx = begin;
                if base_class == NBIT_ARRAY {
                    decompress_array(out, data_offset + idx * base_size, stream, parms, pidx)?;
                } else {
                    decompress_compound(out, data_offset + idx * base_size, stream, parms, pidx)?;
                }
            }
        }
        NBIT_NOOPTYPE => {
            let _size = take(parms, pidx)?;
            stream.copy_bytes(out, data_offset, total_size)?;
        }
        other => {
            return Err(Error::InvalidFormat(format!(
                "invalid nbit array base class {other}"
            )));
        }
    }

    Ok(())
}

fn decompress_compound(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: &[u32],
    pidx: &mut usize,
) -> Result<()> {
    let size = take(parms, pidx)? as usize;
    let nmembers = take(parms, pidx)? as usize;
    let mut used_size = 0usize;

    for _ in 0..nmembers {
        let member_offset = take(parms, pidx)? as usize;
        let member_class = take(parms, pidx)?;
        let member_size = *parms
            .get(*pidx)
            .ok_or_else(|| Error::InvalidFormat("nbit missing compound member size".into()))?
            as usize;

        used_size = used_size
            .checked_add(member_size)
            .ok_or_else(|| Error::InvalidFormat("nbit compound member size overflow".into()))?;
        if used_size > size || member_offset + member_size > size {
            return Err(Error::InvalidFormat(
                "nbit compound member exceeds compound bounds".into(),
            ));
        }

        match member_class {
            NBIT_ATOMIC => {
                let p = AtomicParms {
                    size: take(parms, pidx)? as usize,
                    order: take(parms, pidx)?,
                    precision: take(parms, pidx)? as usize,
                    offset: take(parms, pidx)? as usize,
                };
                validate_atomic(p)?;
                decompress_atomic(out, data_offset + member_offset, stream, p)?;
            }
            NBIT_ARRAY => {
                decompress_array(out, data_offset + member_offset, stream, parms, pidx)?;
            }
            NBIT_COMPOUND => {
                decompress_compound(out, data_offset + member_offset, stream, parms, pidx)?;
            }
            NBIT_NOOPTYPE => {
                let _size = take(parms, pidx)?;
                stream.copy_bytes(out, data_offset + member_offset, member_size)?;
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid nbit compound member class {other}"
                )));
            }
        }
    }

    Ok(())
}

fn decompress_atomic(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: AtomicParms,
) -> Result<()> {
    let dtype_bits = parms.size * 8;
    if parms.order == NBIT_ORDER_LE {
        let begin = if (parms.precision + parms.offset) % 8 != 0 {
            (parms.precision + parms.offset) / 8
        } else {
            (parms.precision + parms.offset) / 8 - 1
        };
        let end = parms.offset / 8;
        for k in (end..=begin).rev() {
            decompress_atomic_byte(out, data_offset, k, begin, end, stream, parms, dtype_bits)?;
        }
    } else if parms.order == NBIT_ORDER_BE {
        let begin = (dtype_bits - parms.precision - parms.offset) / 8;
        let end = if parms.offset % 8 != 0 {
            (dtype_bits - parms.offset) / 8
        } else {
            (dtype_bits - parms.offset) / 8 - 1
        };
        for k in begin..=end {
            decompress_atomic_byte(out, data_offset, k, begin, end, stream, parms, dtype_bits)?;
        }
    } else {
        return Err(Error::InvalidFormat(format!(
            "invalid nbit byte order {}",
            parms.order
        )));
    }

    Ok(())
}

fn decompress_atomic_byte(
    out: &mut [u8],
    data_offset: usize,
    k: usize,
    begin: usize,
    end: usize,
    stream: &mut BitStream<'_>,
    parms: AtomicParms,
    dtype_bits: usize,
) -> Result<()> {
    let (dat_offset, dat_len) = if begin != end {
        if k == begin {
            (0, 8 - (dtype_bits - parms.precision - parms.offset) % 8)
        } else if k == end {
            let len = 8 - parms.offset % 8;
            (8 - len, len)
        } else {
            (0, 8)
        }
    } else {
        (parms.offset % 8, parms.precision)
    };

    let bits = stream.read_bits(dat_len)? as u8;
    let out_idx = data_offset + k;
    if out_idx >= out.len() {
        return Err(Error::InvalidFormat(
            "nbit output offset out of range".into(),
        ));
    }
    out[out_idx] |= bits << dat_offset;
    Ok(())
}

fn validate_atomic(parms: AtomicParms) -> Result<()> {
    if parms.size == 0
        || parms.precision == 0
        || parms.precision > parms.size * 8
        || parms.precision + parms.offset > parms.size * 8
    {
        return Err(Error::InvalidFormat(
            "invalid nbit datatype precision/offset".into(),
        ));
    }
    Ok(())
}

fn take(parms: &[u32], pidx: &mut usize) -> Result<u32> {
    let value = *parms
        .get(*pidx)
        .ok_or_else(|| Error::InvalidFormat("truncated nbit parameters".into()))?;
    *pidx += 1;
    Ok(value)
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
            return Err(Error::InvalidFormat("nbit bit run too long".into()));
        }

        let mut value = 0u16;
        while nbits > 0 {
            let byte = *self
                .data
                .get(self.byte)
                .ok_or_else(|| Error::InvalidFormat("nbit data too short".into()))?;
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

    fn copy_bytes(&mut self, out: &mut [u8], offset: usize, size: usize) -> Result<()> {
        for idx in 0..size {
            out[offset + idx] = self.read_bits(8)? as u8;
        }
        Ok(())
    }
}
