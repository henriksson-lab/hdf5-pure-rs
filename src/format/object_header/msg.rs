//! Object header message decoding — mirrors libhdf5's `H5Omessage.c` plus
//! the per-message decode helpers in `H5Oint.c`. Iterates the on-disk
//! message stream within one chunk, dispatches NIL / HEADER_CONTINUATION
//! to the chunk layer, and forwards real messages back to the caller.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::chunk::reserve_continuation_range;
use super::{
    is_undefined_addr, read_le_uint, RawMessage, MSG_FLAG_SHARED, MSG_HEADER_CONTINUATION, MSG_NIL,
    MSG_SHARED_MSG_TABLE, SHARED_HEAP_ID_LEN, SHARED_MESSAGE_MAX_INDEXES,
    SHARED_MESSAGE_TABLE_VERSION, SHARED_REFERENCE_VERSION_1, SHARED_REFERENCE_VERSION_2,
    SHARED_REFERENCE_VERSION_3, SHARED_TYPE_COMMITTED, SHARED_TYPE_SOHM,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn read_v1_messages<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    chunk_end: u64,
    _num_messages: u16,
    messages: &mut Vec<RawMessage>,
    continuations: &mut Vec<(u64, u64)>,
    chunk_ranges: &mut Vec<(u64, u64)>,
    chunk_index: u16,
) -> Result<()> {
    while reader.position()? < chunk_end {
        let pos = reader.position()?;
        let header_end = checked_u64_add(pos, 8, "object header v1 message header")?;
        if header_end > chunk_end {
            let remaining = (chunk_end - pos) as usize;
            let padding = reader.read_bytes(remaining)?;
            if padding.iter().all(|&b| b == 0) {
                break;
            }
            return Err(Error::InvalidFormat(
                "object header v1 message header is truncated".into(),
            ));
        }

        let msg_type = reader.read_u16()?;
        let msg_size = reader.read_u16()? as u64;
        let msg_flags = reader.read_u8()?;
        // 3 reserved bytes
        reader.skip(3)?;

        // Aligned message size (v1 messages are 8-byte aligned)
        let aligned_size = msg_size
            .checked_add(7)
            .map(|n| n & !7)
            .ok_or_else(|| Error::InvalidFormat("object header message size overflow".into()))?;
        let data_start = checked_u64_add(pos, 8, "object header v1 message data start")?;
        let data_end = data_start
            .checked_add(aligned_size)
            .ok_or_else(|| Error::InvalidFormat("object header message range overflow".into()))?;
        if data_end > chunk_end {
            return Err(Error::InvalidFormat(
                "object header v1 message payload exceeds chunk".into(),
            ));
        }

        if msg_type == MSG_NIL {
            reader.skip(aligned_size)?;
            continue;
        }

        if msg_type == MSG_HEADER_CONTINUATION {
            // Continuation message: contains offset + length
            let used = reader.sizeof_addr() as u64 + reader.sizeof_size() as u64;
            if msg_size < used {
                return Err(Error::InvalidFormat(
                    "object header continuation message is truncated".into(),
                ));
            }
            let cont_offset = reader.read_addr()?;
            let cont_length = reader.read_length()?;
            reserve_continuation_range(reader, cont_offset, cont_length, 8, chunk_ranges)?;
            let remaining = aligned_size - used;
            if remaining > 0 {
                reader.skip(remaining)?;
            }
            continuations.push((cont_offset, cont_length));
            continue;
        }

        let data = reader.read_bytes(msg_size as usize)?;
        // Skip padding to alignment
        let padding = aligned_size - msg_size;
        if padding > 0 {
            reader.skip(padding)?;
        }
        validate_message_payload(
            msg_type,
            msg_flags,
            &data,
            reader.sizeof_addr(),
            reader.sizeof_size(),
        )?;

        messages.push(RawMessage {
            msg_type,
            flags: msg_flags,
            creation_index: None,
            chunk_index,
            data,
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn read_v2_messages<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    chunk_data_end: u64,
    has_crt_order: bool,
    messages: &mut Vec<RawMessage>,
    continuations: &mut Vec<(u64, u64)>,
    chunk_ranges: &mut Vec<(u64, u64)>,
    chunk_index: u16,
) -> Result<()> {
    while reader.position()? < chunk_data_end {
        let pos = reader.position()?;
        // Minimum message header size: 4 bytes (type:1, size:2, flags:1)
        let header_end = checked_u64_add(pos, 4, "object header v2 message header")?;
        if header_end > chunk_data_end {
            let remaining = (chunk_data_end - pos) as usize;
            let padding = reader.read_bytes(remaining)?;
            if padding.iter().all(|&b| b == 0) {
                break;
            }
            return Err(Error::InvalidFormat(
                "object header v2 message header is truncated".into(),
            ));
        }

        let msg_type = reader.read_u8()? as u16;
        let msg_size = reader.read_u16()? as u64;
        let msg_flags = reader.read_u8()?;

        let creation_index = if has_crt_order {
            let creation_order_end = checked_u64_add(
                reader.position()?,
                2,
                "object header v2 message creation order",
            )?;
            if creation_order_end > chunk_data_end {
                return Err(Error::InvalidFormat(
                    "object header v2 message creation order is truncated".into(),
                ));
            }
            Some(reader.read_u16()?)
        } else {
            None
        };
        let data_start = reader.position()?;
        let data_end = data_start.checked_add(msg_size).ok_or_else(|| {
            Error::InvalidFormat("object header v2 message range overflow".into())
        })?;
        if data_end > chunk_data_end {
            return Err(Error::InvalidFormat(
                "object header v2 message payload exceeds chunk".into(),
            ));
        }

        if msg_type == MSG_NIL {
            reader.skip(msg_size)?;
            continue;
        }

        if msg_type == MSG_HEADER_CONTINUATION {
            let used = reader.sizeof_addr() as u64 + reader.sizeof_size() as u64;
            if msg_size < used {
                return Err(Error::InvalidFormat(
                    "object header continuation message is truncated".into(),
                ));
            }
            let cont_offset = reader.read_addr()?;
            let cont_length = reader.read_length()?;
            reserve_continuation_range(reader, cont_offset, cont_length, 8, chunk_ranges)?;
            if msg_size > used {
                reader.skip(msg_size - used)?;
            }
            continuations.push((cont_offset, cont_length));
            continue;
        }

        let data = reader.read_bytes(msg_size as usize)?;
        validate_message_payload(
            msg_type,
            msg_flags,
            &data,
            reader.sizeof_addr(),
            reader.sizeof_size(),
        )?;

        messages.push(RawMessage {
            msg_type,
            flags: msg_flags,
            creation_index,
            chunk_index,
            data,
        });
    }

    Ok(())
}

fn validate_message_payload(
    msg_type: u16,
    msg_flags: u8,
    data: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    if msg_type == MSG_SHARED_MSG_TABLE {
        validate_shared_message_table(data, sizeof_addr)?;
    }
    if msg_flags & MSG_FLAG_SHARED != 0 {
        validate_shared_message_reference(data, sizeof_addr, sizeof_size)?;
    }
    Ok(())
}

fn validate_shared_message_table(data: &[u8], sizeof_addr: u8) -> Result<()> {
    let expected_len = 1usize
        .checked_add(sizeof_addr as usize)
        .and_then(|len| len.checked_add(1))
        .ok_or_else(|| Error::InvalidFormat("shared message table size overflow".into()))?;
    if data.len() != expected_len {
        return Err(Error::InvalidFormat(
            "shared message table payload has invalid length".into(),
        ));
    }
    let version = data[0];
    if version != SHARED_MESSAGE_TABLE_VERSION {
        return Err(Error::InvalidFormat(format!(
            "unsupported shared message table version: {version}"
        )));
    }
    let addr_end = checked_usize_add(1, sizeof_addr as usize, "shared message table address")?;
    let table_addr = read_le_uint(&data[1..addr_end])?;
    if is_undefined_addr(table_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "shared message table address is undefined".into(),
        ));
    }
    let nindexes = data[addr_end];
    if nindexes == 0 || nindexes > SHARED_MESSAGE_MAX_INDEXES {
        return Err(Error::InvalidFormat(
            "shared message table index count is invalid".into(),
        ));
    }
    Ok(())
}

fn validate_shared_message_reference(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<()> {
    if data.len() < 2 {
        return Err(Error::InvalidFormat(
            "shared object-header message reference is truncated".into(),
        ));
    }

    let version = data[0];
    match version {
        SHARED_REFERENCE_VERSION_1 => {
            let expected_len = 2usize
                .checked_add(6)
                .and_then(|len| len.checked_add(sizeof_size as usize))
                .and_then(|len| len.checked_add(sizeof_addr as usize))
                .ok_or_else(|| {
                    Error::InvalidFormat("shared message reference size overflow".into())
                })?;
            if data.len() != expected_len {
                return Err(Error::InvalidFormat(
                    "shared object-header message v1 reference has invalid length".into(),
                ));
            }
            let addr_start = checked_usize_add(
                checked_usize_add(2, 6, "shared message v1 reference prefix")?,
                sizeof_size as usize,
                "shared message v1 reference address",
            )?;
            let addr = read_le_uint(&data[addr_start..])?;
            if is_undefined_addr(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
        }
        SHARED_REFERENCE_VERSION_2 => {
            let expected_len = 2usize.checked_add(sizeof_addr as usize).ok_or_else(|| {
                Error::InvalidFormat("shared message reference size overflow".into())
            })?;
            if data.len() != expected_len {
                return Err(Error::InvalidFormat(
                    "shared object-header message v2 reference has invalid length".into(),
                ));
            }
            let addr = read_le_uint(&data[2..])?;
            if is_undefined_addr(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
        }
        SHARED_REFERENCE_VERSION_3 => match data[1] {
            SHARED_TYPE_SOHM => {
                let expected_len =
                    checked_usize_add(2, SHARED_HEAP_ID_LEN, "shared SOHM reference")?;
                if data.len() != expected_len {
                    return Err(Error::InvalidFormat(
                        "shared object-header message SOHM reference has invalid length".into(),
                    ));
                }
            }
            SHARED_TYPE_COMMITTED => {
                let expected_len = 2usize.checked_add(sizeof_addr as usize).ok_or_else(|| {
                    Error::InvalidFormat("shared message reference size overflow".into())
                })?;
                if data.len() != expected_len {
                    return Err(Error::InvalidFormat(
                        "shared object-header message committed reference has invalid length"
                            .into(),
                    ));
                }
                let addr = read_le_uint(&data[2..])?;
                if is_undefined_addr(addr, sizeof_addr)? {
                    return Err(Error::InvalidFormat(
                        "shared object-header message address is undefined".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::InvalidFormat(
                    "shared object-header message type is invalid".into(),
                ));
            }
        },
        _ => {
            return Err(Error::InvalidFormat(
                "shared object-header message version is invalid".into(),
            ));
        }
    }

    Ok(())
}

fn checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} size overflow")))
}

#[cfg(test)]
mod tests {
    use super::{checked_u64_add, checked_usize_add};

    #[test]
    fn object_header_checked_u64_add_rejects_overflow() {
        let err = checked_u64_add(u64::MAX, 1, "message header").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn object_header_checked_usize_add_rejects_overflow() {
        let err = checked_usize_add(usize::MAX, 1, "message size").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
