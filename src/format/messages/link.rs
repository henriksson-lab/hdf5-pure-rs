use crate::error::{Error, Result};

/// Link type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    Hard,
    Soft,
    External,
    UserDefined(u8),
}

/// A parsed Link message (type 0x0006).
#[derive(Debug, Clone)]
pub struct LinkMessage {
    /// Link name.
    pub name: String,
    /// Link type.
    pub link_type: LinkType,
    /// Creation order (if tracked).
    pub creation_order: Option<u64>,
    /// Character encoding (0=ASCII, 1=UTF-8).
    pub char_encoding: u8,
    /// For hard links: object header address.
    pub hard_link_addr: Option<u64>,
    /// For soft links: target path.
    pub soft_link_target: Option<String>,
    /// For external links: (filename, obj_path).
    pub external_link: Option<(String, String)>,
}

impl LinkMessage {
    /// Decode a link message from raw bytes.
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        let mut pos = 0;

        // Version
        let version = read_u8(data, &mut pos, "link message version")?;
        if version != 1 {
            return Err(Error::InvalidFormat(format!(
                "link message version {version}"
            )));
        }

        // Flags
        let flags = read_u8(data, &mut pos, "link message flags")?;

        let size_of_len_of_link_name = flags & 0x03; // 2 bits
        let has_creation_order = flags & 0x04 != 0;
        let has_link_type = flags & 0x08 != 0;
        let has_char_encoding = flags & 0x10 != 0;

        // Link type (optional)
        let link_type = if has_link_type {
            let t = read_u8(data, &mut pos, "link message link type")?;
            match t {
                0 => LinkType::Hard,
                1 => LinkType::Soft,
                64 => LinkType::External,
                other => LinkType::UserDefined(other),
            }
        } else {
            LinkType::Hard // default
        };

        // Creation order (optional)
        let creation_order = if has_creation_order {
            let val = read_le_u64(data, &mut pos, 8, "link message creation order")?;
            Some(val)
        } else {
            None
        };

        // Character encoding (optional)
        let char_encoding = if has_char_encoding {
            read_u8(data, &mut pos, "link message character encoding")?
        } else {
            0 // ASCII
        };
        if char_encoding > 1 {
            return Err(Error::InvalidFormat(format!(
                "invalid link character encoding {char_encoding}"
            )));
        }

        // Length of link name
        let name_len_size = 1 << size_of_len_of_link_name; // 1, 2, 4, or 8
        let name_len = read_le_u64(data, &mut pos, name_len_size, "link message name length")?
            .try_into()
            .map_err(|_| Error::InvalidFormat("link name length overflows usize".into()))?;

        // Link name
        ensure_available(data, pos, name_len, "link name")?;
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Link value based on type
        let mut hard_link_addr = None;
        let mut soft_link_target = None;
        let mut external_link = None;

        match link_type {
            LinkType::Hard => {
                hard_link_addr = Some(read_le_u64(
                    data,
                    &mut pos,
                    sizeof_addr as usize,
                    "hard link address",
                )?);
            }
            LinkType::Soft => {
                let target_len =
                    read_le_u64(data, &mut pos, 2, "soft link target length")? as usize;
                ensure_available(data, pos, target_len, "soft link target")?;
                soft_link_target =
                    Some(String::from_utf8_lossy(&data[pos..pos + target_len]).to_string());
            }
            LinkType::External => {
                let info_len =
                    read_le_u64(data, &mut pos, 2, "external link info length")? as usize;
                if info_len < 2 {
                    return Err(Error::InvalidFormat(
                        "external link info is too short".into(),
                    ));
                }
                ensure_available(data, pos, info_len, "external link info")?;
                let ext_version = read_u8(data, &mut pos, "external link version")?;
                let mut info_consumed = 1;
                if ext_version >= 1 {
                    let _ext_flags = read_u8(data, &mut pos, "external link flags")?;
                    info_consumed += 1;
                }
                let end = pos + info_len - info_consumed;

                // Filename (null-terminated)
                let null_pos = data[pos..end]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(end - pos);
                let filename = String::from_utf8_lossy(&data[pos..pos + null_pos]).to_string();
                pos = (pos + null_pos + 1).min(end);

                // Object path (null-terminated)
                let null_pos2 = data[pos..end]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(end - pos);
                let obj_path = String::from_utf8_lossy(&data[pos..pos + null_pos2]).to_string();
                trace_external_link_resolve(&filename, &obj_path);

                external_link = Some((filename, obj_path));
            }
            LinkType::UserDefined(_) => {
                // Skip user-defined link data
            }
        }

        Ok(LinkMessage {
            name,
            link_type,
            creation_order,
            char_encoding,
            hard_link_addr,
            soft_link_target,
            external_link,
        })
    }
}

#[cfg(feature = "tracehash")]
fn trace_external_link_resolve(filename: &str, obj_path: &str) {
    let mut th = tracehash::th_call!("hdf5.external_link.resolve");
    th.input_bytes(filename.as_bytes());
    th.input_bytes(obj_path.as_bytes());
    th.output_bool(true);
    th.output_bytes(filename.as_bytes());
    th.output_bytes(obj_path.as_bytes());
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_external_link_resolve(_filename: &str, _obj_path: &str) {}

fn ensure_available(data: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    *pos += 1;
    Ok(value)
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    ensure_available(data, *pos, size, context)?;
    let mut val = 0u64;
    for i in 0..size {
        val |= (data[*pos + i] as u64) << (i * 8);
    }
    *pos += size;
    Ok(val)
}
