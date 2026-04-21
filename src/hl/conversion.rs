use std::any::TypeId;

use crate::error::{Error, Result};
use crate::format::messages::datatype::{ByteOrder, DatatypeClass, DatatypeMessage};
use crate::hl::types::{self, H5Type};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ReadConversion {
    element_size: usize,
    byte_order: Option<ByteOrder>,
    kind: ConversionKind,
}

#[derive(Debug, Clone, Copy)]
enum ConversionKind {
    SameSizeBytes,
    Integer {
        src_size: usize,
        src_signed: bool,
        dst_size: usize,
        dst_signed: bool,
    },
    FloatToFloat {
        src_size: usize,
        dst_size: usize,
    },
    IntegerToFloat {
        src_size: usize,
        src_signed: bool,
        dst_size: usize,
    },
    FloatToInteger {
        src_size: usize,
        dst_size: usize,
        dst_signed: bool,
    },
}

impl ReadConversion {
    pub(crate) fn for_dataset<T: H5Type>(datatype: &DatatypeMessage) -> Result<Self> {
        if datatype.class == DatatypeClass::Time {
            return Err(Error::Unsupported(
                "typed reads for HDF5 time datatypes are not supported".into(),
            ));
        }
        let requested = T::type_size();
        let stored = datatype.size as usize;
        let byte_order = datatype.byte_order();

        // Dispatch on source class — each per-class helper picks the
        // matching `ConversionKind` (mirroring how libhdf5's
        // `H5T_path_find` registers per-class converters in
        // `H5T__conv_*`). Same-size byte copies are handled in the
        // dispatcher's fallthrough so each helper can stay focused on
        // the type-class-specific decisions.
        let kind = match datatype.class {
            DatatypeClass::FixedPoint | DatatypeClass::Enum | DatatypeClass::BitField => {
                Self::kind_for_integer_source::<T>(datatype, requested, stored)?
            }
            DatatypeClass::FloatingPoint => Self::kind_for_float_source::<T>(requested, stored)?,
            _ => Self::kind_for_passthrough(requested, stored)?,
        };

        Ok(Self {
            element_size: stored,
            byte_order,
            kind,
        })
    }

    /// Source class is FixedPoint / Enum / BitField. Mirrors libhdf5's
    /// `H5T__conv_i_i` / `H5T__conv_i_f` selection.
    fn kind_for_integer_source<T: H5Type>(
        datatype: &DatatypeMessage,
        requested: usize,
        stored: usize,
    ) -> Result<ConversionKind> {
        if let Some((dst_signed, dst_size)) = target_integer::<T>() {
            Ok(if requested == stored {
                ConversionKind::SameSizeBytes
            } else {
                ConversionKind::Integer {
                    src_size: stored,
                    src_signed: datatype.is_signed().unwrap_or(false),
                    dst_size,
                    dst_signed,
                }
            })
        } else if let Some(dst_size) = target_float::<T>() {
            Ok(ConversionKind::IntegerToFloat {
                src_size: stored,
                src_signed: datatype.is_signed().unwrap_or(false),
                dst_size,
            })
        } else {
            Err(Error::InvalidFormat(format!(
                "requested element size {requested} does not match dataset element size {stored}"
            )))
        }
    }

    /// Source class is FloatingPoint. Mirrors libhdf5's
    /// `H5T__conv_f_f` / `H5T__conv_f_i` selection.
    fn kind_for_float_source<T: H5Type>(requested: usize, stored: usize) -> Result<ConversionKind> {
        if let Some(dst_size) = target_float::<T>() {
            Ok(if requested == stored {
                ConversionKind::SameSizeBytes
            } else {
                ConversionKind::FloatToFloat {
                    src_size: stored,
                    dst_size,
                }
            })
        } else if let Some((dst_signed, dst_size)) = target_integer::<T>() {
            Ok(ConversionKind::FloatToInteger {
                src_size: stored,
                dst_size,
                dst_signed,
            })
        } else {
            Err(Error::InvalidFormat(format!(
                "requested element size {requested} does not match dataset element size {stored}"
            )))
        }
    }

    /// Source classes that fall through to a same-size byte copy
    /// (String / Opaque / Reference / Compound / Array / VarLen). The
    /// caller must pre-validate that a typed read is meaningful for the
    /// given source class.
    fn kind_for_passthrough(requested: usize, stored: usize) -> Result<ConversionKind> {
        if requested == stored {
            Ok(ConversionKind::SameSizeBytes)
        } else {
            Err(Error::InvalidFormat(format!(
                "requested element size {requested} does not match dataset element size {stored}"
            )))
        }
    }

    pub(crate) fn bytes_to_vec<T: H5Type>(&self, mut bytes: Vec<u8>) -> Result<Vec<T>> {
        match self.kind {
            ConversionKind::SameSizeBytes => {
                self.convert_bytes_in_place(&mut bytes);
                types::bytes_to_vec(bytes)
            }
            ConversionKind::Integer {
                src_size,
                src_signed,
                dst_size,
                dst_signed,
            } => {
                let converted = convert_integer_bytes(
                    &bytes,
                    src_size,
                    src_signed,
                    self.byte_order,
                    dst_size,
                    dst_signed,
                )?;
                types::bytes_to_vec(converted)
            }
            ConversionKind::FloatToFloat { src_size, dst_size } => {
                let converted = convert_float_bytes(&bytes, src_size, self.byte_order, dst_size)?;
                types::bytes_to_vec(converted)
            }
            ConversionKind::IntegerToFloat {
                src_size,
                src_signed,
                dst_size,
            } => {
                let converted = convert_integer_to_float_bytes(
                    &bytes,
                    src_size,
                    src_signed,
                    self.byte_order,
                    dst_size,
                )?;
                types::bytes_to_vec(converted)
            }
            ConversionKind::FloatToInteger {
                src_size,
                dst_size,
                dst_signed,
            } => {
                let converted = convert_float_to_integer_bytes(
                    &bytes,
                    src_size,
                    self.byte_order,
                    dst_size,
                    dst_signed,
                )?;
                types::bytes_to_vec(converted)
            }
        }
    }

    pub(crate) fn bytes_to_scalar<T: H5Type>(&self, bytes: Vec<u8>) -> Result<T> {
        let values = self.bytes_to_vec::<T>(bytes)?;
        values
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidFormat("no data for scalar read".into()))
    }

    fn convert_bytes_in_place(&self, bytes: &mut [u8]) {
        maybe_swap_elements(bytes, self.element_size, self.byte_order);
    }
}

fn target_integer<T: H5Type>() -> Option<(bool, usize)> {
    let type_id = TypeId::of::<T>();
    if type_id == TypeId::of::<i8>() {
        Some((true, 1))
    } else if type_id == TypeId::of::<i16>() {
        Some((true, 2))
    } else if type_id == TypeId::of::<i32>() {
        Some((true, 4))
    } else if type_id == TypeId::of::<i64>() {
        Some((true, 8))
    } else if type_id == TypeId::of::<u8>() {
        Some((false, 1))
    } else if type_id == TypeId::of::<u16>() {
        Some((false, 2))
    } else if type_id == TypeId::of::<u32>() {
        Some((false, 4))
    } else if type_id == TypeId::of::<u64>() {
        Some((false, 8))
    } else {
        None
    }
}

fn target_float<T: H5Type>() -> Option<usize> {
    let type_id = TypeId::of::<T>();
    if type_id == TypeId::of::<f32>() {
        Some(4)
    } else if type_id == TypeId::of::<f64>() {
        Some(8)
    } else {
        None
    }
}

fn convert_integer_bytes(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_signed: bool,
) -> Result<Vec<u8>> {
    if src_size == 0 || dst_size == 0 || src_size > 16 || dst_size > 16 {
        return Err(Error::Unsupported(
            "integer conversion supports 1..=16 byte integer payloads".into(),
        ));
    }
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source integer size {src_size}",
            bytes.len()
        )));
    }

    let mut out = vec![0u8; (bytes.len() / src_size) * dst_size];
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = if src_signed {
            IntegerValue::Signed(read_signed(chunk, src_order))
        } else {
            IntegerValue::Unsigned(read_unsigned(chunk, src_order))
        };
        let raw = clamp_integer(value, dst_size, dst_signed);
        write_native_uint(&mut out[idx * dst_size..(idx + 1) * dst_size], raw);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
enum IntegerValue {
    Signed(i128),
    Unsigned(u128),
}

fn clamp_integer(value: IntegerValue, dst_size: usize, dst_signed: bool) -> u128 {
    let bits = dst_size * 8;
    if dst_signed {
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        let clamped = match value {
            IntegerValue::Signed(value) => value.clamp(min, max),
            IntegerValue::Unsigned(value) => {
                if value > max as u128 {
                    max
                } else {
                    value as i128
                }
            }
        };
        signed_to_raw(clamped, bits)
    } else {
        let max = (1u128 << bits) - 1;
        match value {
            IntegerValue::Signed(value) => {
                if value <= 0 {
                    0
                } else {
                    (value as u128).min(max)
                }
            }
            IntegerValue::Unsigned(value) => value.min(max),
        }
    }
}

fn signed_to_raw(value: i128, bits: usize) -> u128 {
    if value >= 0 {
        value as u128
    } else {
        (1u128 << bits).wrapping_add(value as u128)
    }
}

fn read_unsigned(bytes: &[u8], byte_order: Option<ByteOrder>) -> u128 {
    let little = matches!(byte_order, Some(ByteOrder::LittleEndian) | None);
    let mut value = 0u128;
    if little {
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

fn read_signed(bytes: &[u8], byte_order: Option<ByteOrder>) -> i128 {
    let n = bytes.len().min(16);
    let unsigned = read_unsigned(bytes, byte_order);
    let bits = n * 8;
    let sign_bit = 1u128 << (bits - 1);
    if unsigned & sign_bit == 0 {
        unsigned as i128
    } else {
        (unsigned as i128) - (1i128 << bits)
    }
}

fn write_native_uint(bytes: &mut [u8], value: u128) {
    if cfg!(target_endian = "little") {
        for (idx, byte) in bytes.iter_mut().enumerate() {
            *byte = (value >> (idx * 8)) as u8;
        }
    } else {
        let n = bytes.len();
        for (idx, byte) in bytes.iter_mut().enumerate() {
            *byte = (value >> ((n - idx - 1) * 8)) as u8;
        }
    }
}

fn convert_float_bytes(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
) -> Result<Vec<u8>> {
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source float size {src_size}",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; (bytes.len() / src_size) * dst_size];
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = read_float(chunk, src_size, src_order)?;
        write_native_float(&mut out[idx * dst_size..(idx + 1) * dst_size], value)?;
    }
    Ok(out)
}

fn convert_integer_to_float_bytes(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    dst_size: usize,
) -> Result<Vec<u8>> {
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source integer size {src_size}",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; (bytes.len() / src_size) * dst_size];
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = if src_signed {
            read_signed(chunk, src_order) as f64
        } else {
            read_unsigned(chunk, src_order) as f64
        };
        write_native_float(&mut out[idx * dst_size..(idx + 1) * dst_size], value)?;
    }
    Ok(out)
}

fn convert_float_to_integer_bytes(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_signed: bool,
) -> Result<Vec<u8>> {
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source float size {src_size}",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; (bytes.len() / src_size) * dst_size];
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = read_float(chunk, src_size, src_order)?;
        let raw = clamp_float_to_integer(value, dst_size, dst_signed);
        write_native_uint(&mut out[idx * dst_size..(idx + 1) * dst_size], raw);
    }
    Ok(out)
}

fn read_float(bytes: &[u8], size: usize, byte_order: Option<ByteOrder>) -> Result<f64> {
    let mut raw = bytes[..size].to_vec();
    maybe_swap_elements(&mut raw, size, byte_order);
    match size {
        4 => Ok(f32::from_ne_bytes(raw.try_into().unwrap()) as f64),
        8 => Ok(f64::from_ne_bytes(raw.try_into().unwrap())),
        _ => Err(Error::Unsupported(format!(
            "floating-point conversion supports 4- and 8-byte payloads, got {size}"
        ))),
    }
}

fn write_native_float(bytes: &mut [u8], value: f64) -> Result<()> {
    match bytes.len() {
        4 => bytes.copy_from_slice(&(value as f32).to_ne_bytes()),
        8 => bytes.copy_from_slice(&value.to_ne_bytes()),
        size => {
            return Err(Error::Unsupported(format!(
                "floating-point conversion supports 4- and 8-byte targets, got {size}"
            )));
        }
    }
    Ok(())
}

fn clamp_float_to_integer(value: f64, dst_size: usize, dst_signed: bool) -> u128 {
    let bits = dst_size * 8;
    if dst_signed {
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        if value.is_nan() {
            return 0;
        }
        let clamped = if value.is_infinite() && value.is_sign_negative() {
            min
        } else if value.is_infinite() {
            max
        } else if value <= min as f64 {
            min
        } else if value >= max as f64 {
            max
        } else {
            value.trunc() as i128
        };
        signed_to_raw(clamped, bits)
    } else {
        let max = (1u128 << bits) - 1;
        if value.is_nan() || value <= 0.0 {
            0
        } else if value.is_infinite() || value >= max as f64 {
            max
        } else {
            value.trunc() as u128
        }
    }
}

pub(crate) fn maybe_swap_elements(
    bytes: &mut [u8],
    element_size: usize,
    byte_order: Option<ByteOrder>,
) {
    if element_size <= 1 {
        return;
    }

    let need_swap = match byte_order {
        Some(ByteOrder::BigEndian) => cfg!(target_endian = "little"),
        Some(ByteOrder::LittleEndian) => cfg!(target_endian = "big"),
        None => false,
    };

    if need_swap {
        for chunk in bytes.chunks_exact_mut(element_size) {
            chunk.reverse();
        }
    }
}
