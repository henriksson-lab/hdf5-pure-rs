use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

/// Magic number for v2 object headers: "OHDR"
const OHDR_MAGIC: [u8; 4] = [b'O', b'H', b'D', b'R'];

/// Magic number for v2 continuation chunks: "OCHK"
const OCHK_MAGIC: [u8; 4] = [b'O', b'C', b'H', b'K'];

// Object header flags (v2)
pub const HDR_CHUNK0_SIZE_MASK: u8 = 0x03;
pub const HDR_ATTR_CRT_ORDER_TRACKED: u8 = 0x04;
pub const HDR_ATTR_CRT_ORDER_INDEXED: u8 = 0x08;
pub const HDR_ATTR_STORE_PHASE_CHANGE: u8 = 0x10;
pub const HDR_STORE_TIMES: u8 = 0x20;

const MSG_FLAG_SHARED: u8 = 0x02;
const SHARED_MESSAGE_TABLE_VERSION: u8 = 0;
const SHARED_MESSAGE_MAX_INDEXES: u8 = 8;
const SHARED_REFERENCE_VERSION_1: u8 = 1;
const SHARED_REFERENCE_VERSION_2: u8 = 2;
const SHARED_REFERENCE_VERSION_3: u8 = 3;
const SHARED_TYPE_SOHM: u8 = 1;
const SHARED_TYPE_COMMITTED: u8 = 2;
const SHARED_HEAP_ID_LEN: usize = 8;

// Message type IDs
pub const MSG_NIL: u16 = 0x0000;
pub const MSG_DATASPACE: u16 = 0x0001;
pub const MSG_LINK_INFO: u16 = 0x0002;
pub const MSG_DATATYPE: u16 = 0x0003;
pub const MSG_FILL_VALUE_OLD: u16 = 0x0004;
pub const MSG_FILL_VALUE: u16 = 0x0005;
pub const MSG_LINK: u16 = 0x0006;
pub const MSG_EXTERNAL_FILE_LIST: u16 = 0x0007;
pub const MSG_LAYOUT: u16 = 0x0008;
pub const MSG_BOGUS: u16 = 0x0009;
pub const MSG_GROUP_INFO: u16 = 0x000A;
pub const MSG_FILTER_PIPELINE: u16 = 0x000B;
pub const MSG_ATTRIBUTE: u16 = 0x000C;
pub const MSG_OBJ_COMMENT: u16 = 0x000D;
pub const MSG_OBJ_MOD_TIME_OLD: u16 = 0x000E;
pub const MSG_SHARED_MSG_TABLE: u16 = 0x000F;
pub const MSG_HEADER_CONTINUATION: u16 = 0x0010;
pub const MSG_SYMBOL_TABLE: u16 = 0x0011;
pub const MSG_OBJ_MOD_TIME: u16 = 0x0012;
pub const MSG_BTREE_K: u16 = 0x0013;
pub const MSG_DRIVER_INFO: u16 = 0x0014;
pub const MSG_ATTR_INFO: u16 = 0x0015;
pub const MSG_OBJ_REF_COUNT: u16 = 0x0016;
pub const MSG_FILE_SPACE_INFO: u16 = 0x0017;

/// A raw message from an object header.
#[derive(Debug, Clone)]
pub struct RawMessage {
    /// Message type ID.
    pub msg_type: u16,
    /// Message flags.
    pub flags: u8,
    /// Creation order index (v2 only, if tracked).
    pub creation_index: Option<u16>,
    /// Object header chunk this message was read from.
    pub chunk_index: u16,
    /// Raw message data bytes.
    pub data: Vec<u8>,
}

/// Parsed object header.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// Header flags (v2 only).
    pub flags: u8,
    /// Reference count.
    pub refcount: u32,
    /// Access time (v2, if HDR_STORE_TIMES).
    pub atime: Option<u32>,
    /// Modification time (v2, if HDR_STORE_TIMES).
    pub mtime: Option<u32>,
    /// Change time (v2, if HDR_STORE_TIMES).
    pub ctime: Option<u32>,
    /// Birth time (v2, if HDR_STORE_TIMES).
    pub btime: Option<u32>,
    /// Max compact attributes (v2, if HDR_ATTR_STORE_PHASE_CHANGE).
    pub max_compact_attrs: Option<u16>,
    /// Min dense attributes (v2, if HDR_ATTR_STORE_PHASE_CHANGE).
    pub min_dense_attrs: Option<u16>,
    /// All messages parsed from this header (including continuation chunks).
    pub messages: Vec<RawMessage>,
}

impl ObjectHeader {
    /// Read an object header at the given file address.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        // Peek at the first byte to determine version.
        // V2 headers start with "OHDR" magic; v1 starts with version byte (1).
        let first_bytes = reader.read_bytes(4)?;

        let result = if first_bytes == OHDR_MAGIC {
            Self::read_v2(reader, addr)
        } else {
            // Seek back and re-read as v1. The first byte is the version.
            reader.seek(addr)?;
            Self::read_v1(reader)
        };

        #[cfg(feature = "tracehash")]
        if let Ok(header) = &result {
            let traced_messages: Vec<_> = header
                .messages
                .iter()
                .filter(|message| message.chunk_index == 0)
                .collect();
            let mut th = tracehash::th_call!("hdf5.object_header.read");
            th.input_u64(addr);
            th.output_u64(header.version as u64);
            th.output_u64(header.flags as u64);
            th.output_u64(header.refcount as u64);
            th.output_u64(traced_messages.len() as u64);
            for message in traced_messages {
                th.output_u64(message.msg_type as u64);
                th.output_u64(message.data.len() as u64);
            }
            th.finish();
        }

        result
    }

    /// Read a v1 object header.
    fn read_v1<R: Read + Seek>(reader: &mut HdfReader<R>) -> Result<Self> {
        let header_start = reader.position()?;
        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected object header v1, got {version}"
            )));
        }

        // Reserved byte
        reader.skip(1)?;

        let num_messages = reader.read_u16()?;
        let refcount = reader.read_u32()?;
        let chunk_data_size = reader.read_u32()? as u64;

        // Reserved/padding to 8-byte boundary (v1 header is 12 bytes after version,
        // total prefix = 1+1+2+4+4 = 12, need to align to 8: 12 is already aligned to 4,
        // but the v1 header macro says H5O_ALIGN_OLD(12) = align to 8 = 16, so 4 padding bytes)
        reader.skip(4)?;

        // Now read messages from chunk data
        let chunk_start = reader.position()?;
        let chunk_end = chunk_start
            .checked_add(chunk_data_size)
            .ok_or_else(|| Error::InvalidFormat("object header v1 chunk size overflow".into()))?;

        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        let mut chunk_ranges = vec![(header_start, chunk_end)];

        Self::read_v1_messages(
            reader,
            chunk_end,
            num_messages,
            &mut messages,
            &mut continuations,
            &mut chunk_ranges,
            0,
        )?;

        // Process continuation chunks
        for (cont_addr, cont_len) in continuations {
            Self::read_v1_continuation(
                reader,
                cont_addr,
                cont_len,
                &mut messages,
                &mut chunk_ranges,
                1,
            )?;
        }

        Ok(ObjectHeader {
            version: 1,
            flags: 0,
            refcount,
            atime: None,
            mtime: None,
            ctime: None,
            btime: None,
            max_compact_attrs: None,
            min_dense_attrs: None,
            messages,
        })
    }

    fn read_v1_messages<R: Read + Seek>(
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
            if pos + 8 > chunk_end {
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
            let aligned_size = msg_size.checked_add(7).map(|n| n & !7).ok_or_else(|| {
                Error::InvalidFormat("object header message size overflow".into())
            })?;
            let data_start = pos + 8;
            let data_end = data_start.checked_add(aligned_size).ok_or_else(|| {
                Error::InvalidFormat("object header message range overflow".into())
            })?;
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
                Self::reserve_continuation_range(
                    reader,
                    cont_offset,
                    cont_length,
                    8,
                    chunk_ranges,
                )?;
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
            Self::validate_message_payload(
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

    fn read_v1_continuation<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        length: u64,
        messages: &mut Vec<RawMessage>,
        chunk_ranges: &mut Vec<(u64, u64)>,
        chunk_index: u16,
    ) -> Result<()> {
        reader.seek(addr)?;

        // V1 continuation chunks are just raw messages, no header.
        let chunk_end = addr.checked_add(length).ok_or_else(|| {
            Error::InvalidFormat("object header continuation range overflow".into())
        })?;
        let mut continuations = Vec::new();
        Self::read_v1_messages(
            reader,
            chunk_end,
            0,
            messages,
            &mut continuations,
            chunk_ranges,
            chunk_index,
        )?;

        for (cont_addr, cont_len) in continuations {
            Self::read_v1_continuation(
                reader,
                cont_addr,
                cont_len,
                messages,
                chunk_ranges,
                chunk_index + 1,
            )?;
        }

        Ok(())
    }

    /// Read a v2 object header. Assumes the "OHDR" magic has already been read.
    fn read_v2<R: Read + Seek>(reader: &mut HdfReader<R>, header_addr: u64) -> Result<Self> {
        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::InvalidFormat(format!(
                "expected object header v2, got {version}"
            )));
        }

        let flags = reader.read_u8()?;

        // Optional timestamps
        let (atime, mtime, ctime, btime) = if flags & HDR_STORE_TIMES != 0 {
            (
                Some(reader.read_u32()?),
                Some(reader.read_u32()?),
                Some(reader.read_u32()?),
                Some(reader.read_u32()?),
            )
        } else {
            (None, None, None, None)
        };

        // Optional attribute phase change
        let (max_compact_attrs, min_dense_attrs) = if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            (Some(reader.read_u16()?), Some(reader.read_u16()?))
        } else {
            (None, None)
        };

        // Chunk 0 data size (1, 2, 4, or 8 bytes based on flags)
        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;

        // Now we know where chunk 0 data starts and where its checksum is
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object header v2 chunk size overflow".into()))?;

        // Verify checksum: it covers from "OHDR" magic to just before the checksum
        let checksum_pos = chunk0_data_end;
        // Read the stored checksum
        reader.seek(checksum_pos)?;
        let stored_checksum = reader.read_u32()?;

        // Compute checksum over header_addr .. checksum_pos
        let check_len = (checksum_pos - header_addr) as usize;
        reader.seek(header_addr)?;
        let check_data = reader.read_bytes(check_len)?;
        let computed = checksum_metadata(&check_data);

        if stored_checksum != computed {
            return Err(Error::InvalidFormat(format!(
                "object header checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}"
            )));
        }

        // Now parse messages from chunk 0 data
        reader.seek(chunk0_data_start)?;

        let has_crt_order = flags & HDR_ATTR_CRT_ORDER_TRACKED != 0;
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        let chunk0_range_end = chunk0_data_end
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("object header v2 chunk range overflow".into()))?;
        let mut chunk_ranges = vec![(header_addr, chunk0_range_end)];

        Self::read_v2_messages(
            reader,
            chunk0_data_end,
            has_crt_order,
            &mut messages,
            &mut continuations,
            &mut chunk_ranges,
            0,
        )?;

        // Process continuation chunks
        for (cont_addr, cont_len) in continuations {
            Self::read_v2_continuation(
                reader,
                cont_addr,
                cont_len,
                has_crt_order,
                &mut messages,
                &mut chunk_ranges,
                1,
            )?;
        }

        Ok(ObjectHeader {
            version: 2,
            flags,
            refcount: 1, // v2 headers may have a separate refcount message
            atime,
            mtime,
            ctime,
            btime,
            max_compact_attrs,
            min_dense_attrs,
            messages,
        })
    }

    fn read_v2_messages<R: Read + Seek>(
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
            if pos + 4 > chunk_data_end {
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
                if reader.position()? + 2 > chunk_data_end {
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

            if msg_type == MSG_NIL as u16 {
                reader.skip(msg_size)?;
                continue;
            }

            if msg_type == MSG_HEADER_CONTINUATION as u16 {
                let used = reader.sizeof_addr() as u64 + reader.sizeof_size() as u64;
                if msg_size < used {
                    return Err(Error::InvalidFormat(
                        "object header continuation message is truncated".into(),
                    ));
                }
                let cont_offset = reader.read_addr()?;
                let cont_length = reader.read_length()?;
                Self::reserve_continuation_range(
                    reader,
                    cont_offset,
                    cont_length,
                    8,
                    chunk_ranges,
                )?;
                if msg_size > used {
                    reader.skip(msg_size - used)?;
                }
                continuations.push((cont_offset, cont_length));
                continue;
            }

            let data = reader.read_bytes(msg_size as usize)?;
            Self::validate_message_payload(
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

    fn read_v2_continuation<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        length: u64,
        has_crt_order: bool,
        messages: &mut Vec<RawMessage>,
        chunk_ranges: &mut Vec<(u64, u64)>,
        chunk_index: u16,
    ) -> Result<()> {
        reader.seek(addr)?;

        // V2 continuation chunks start with "OCHK" magic
        let magic = reader.read_bytes(4)?;
        if magic != OCHK_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid continuation chunk magic".into(),
            ));
        }

        // Data runs from after magic to before checksum
        let _data_start = reader.position()?;
        let data_end = addr
            .checked_add(length)
            .and_then(|end| end.checked_sub(4))
            .ok_or_else(|| {
                Error::InvalidFormat("object header continuation range overflow".into())
            })?; // minus checksum

        let mut continuations = Vec::new();
        Self::read_v2_messages(
            reader,
            data_end,
            has_crt_order,
            messages,
            &mut continuations,
            chunk_ranges,
            chunk_index,
        )?;

        // Verify checksum
        reader.seek(data_end)?;
        let stored_checksum = reader.read_u32()?;
        let check_len = (data_end - addr) as usize;
        reader.seek(addr)?;
        let check_data = reader.read_bytes(check_len)?;
        let computed = checksum_metadata(&check_data);

        if stored_checksum != computed {
            return Err(Error::InvalidFormat(
                "continuation chunk checksum mismatch".into(),
            ));
        }

        // Process nested continuations
        for (cont_addr, cont_len) in continuations {
            Self::read_v2_continuation(
                reader,
                cont_addr,
                cont_len,
                has_crt_order,
                messages,
                chunk_ranges,
                chunk_index + 1,
            )?;
        }

        Ok(())
    }

    fn reserve_continuation_range<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        length: u64,
        min_length: u64,
        chunk_ranges: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        if length < min_length {
            return Err(Error::InvalidFormat(
                "object header continuation chunk is too small".into(),
            ));
        }
        let end = addr.checked_add(length).ok_or_else(|| {
            Error::InvalidFormat("object header continuation range overflow".into())
        })?;
        let file_len = reader.len()?;
        if end > file_len {
            return Err(Error::InvalidFormat(
                "object header continuation range exceeds file size".into(),
            ));
        }
        if chunk_ranges
            .iter()
            .any(|&(range_start, range_end)| addr < range_end && range_start < end)
        {
            return Err(Error::InvalidFormat(
                "object header continuation range overlaps another metadata chunk".into(),
            ));
        }
        chunk_ranges.push((addr, end));
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
            Self::validate_shared_message_table(data, sizeof_addr)?;
        }
        if msg_flags & MSG_FLAG_SHARED != 0 {
            Self::validate_shared_message_reference(data, sizeof_addr, sizeof_size)?;
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
        let addr_end = 1 + sizeof_addr as usize;
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

    fn validate_shared_message_reference(
        data: &[u8],
        sizeof_addr: u8,
        sizeof_size: u8,
    ) -> Result<()> {
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
                let addr_start = 2 + 6 + sizeof_size as usize;
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
                    if data.len() != 2 + SHARED_HEAP_ID_LEN {
                        return Err(Error::InvalidFormat(
                            "shared object-header message SOHM reference has invalid length".into(),
                        ));
                    }
                }
                SHARED_TYPE_COMMITTED => {
                    let expected_len =
                        2usize.checked_add(sizeof_addr as usize).ok_or_else(|| {
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
}

fn read_le_uint(data: &[u8]) -> Result<u64> {
    if data.len() > 8 {
        return Err(Error::InvalidFormat(
            "integer payload is wider than u64".into(),
        ));
    }
    Ok(data.iter().enumerate().fold(0u64, |value, (idx, byte)| {
        value | ((*byte as u64) << (idx * 8))
    }))
}

fn is_undefined_addr(addr: u64, sizeof_addr: u8) -> Result<bool> {
    let bits = u32::from(sizeof_addr)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("address size overflow".into()))?;
    let undef = if bits == 64 {
        u64::MAX
    } else if bits < 64 {
        (1u64 << bits) - 1
    } else {
        return Err(Error::InvalidFormat(
            "address payload is wider than u64".into(),
        ));
    };
    Ok(addr == undef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_parse_v0_root_object_header() {
        let f = File::open("tests/data/simple_v0.h5").unwrap();
        let mut reader = HdfReader::new(BufReader::new(f));
        let sb = crate::format::superblock::Superblock::read(&mut reader).unwrap();

        let oh = ObjectHeader::read_at(&mut reader, sb.root_addr).unwrap();
        println!(
            "v0 root OH: version={}, refcount={}, messages:",
            oh.version, oh.refcount
        );
        for msg in &oh.messages {
            println!(
                "  type={:#06x}, flags={:#04x}, len={}",
                msg.msg_type,
                msg.flags,
                msg.data.len()
            );
        }

        assert_eq!(oh.version, 1);
        // Root group should have a symbol table message
        assert!(oh.messages.iter().any(|m| m.msg_type == MSG_SYMBOL_TABLE));
    }

    #[test]
    fn test_parse_v3_root_object_header() {
        let f = File::open("tests/data/simple_v2.h5").unwrap();
        let mut reader = HdfReader::new(BufReader::new(f));
        let sb = crate::format::superblock::Superblock::read(&mut reader).unwrap();

        let oh = ObjectHeader::read_at(&mut reader, sb.root_addr).unwrap();
        println!(
            "v3 root OH: version={}, flags={:#04x}, messages:",
            oh.version, oh.flags
        );
        for msg in &oh.messages {
            println!(
                "  type={:#06x}, flags={:#04x}, len={}",
                msg.msg_type,
                msg.flags,
                msg.data.len()
            );
        }

        assert_eq!(oh.version, 2);
        // V2 root group should have link messages or link info
        let has_links = oh
            .messages
            .iter()
            .any(|m| m.msg_type == MSG_LINK || m.msg_type == MSG_LINK_INFO);
        assert!(
            has_links,
            "v2 root group should have link or link info messages"
        );
    }
}
