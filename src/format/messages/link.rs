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
        if data.is_empty() {
            return Err(Error::InvalidFormat("empty link message".into()));
        }

        let mut pos = 0;

        // Version
        let version = data[pos];
        pos += 1;
        if version != 1 {
            return Err(Error::Unsupported(format!(
                "link message version {version}"
            )));
        }

        // Flags
        let flags = data[pos];
        pos += 1;

        let size_of_len_of_link_name = flags & 0x03; // 2 bits
        let has_creation_order = flags & 0x04 != 0;
        let has_link_type = flags & 0x08 != 0;
        let has_char_encoding = flags & 0x10 != 0;

        // Link type (optional)
        let link_type = if has_link_type {
            let t = data[pos];
            pos += 1;
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
            let val = read_le_u64(&data[pos..], 8);
            pos += 8;
            Some(val)
        } else {
            None
        };

        // Character encoding (optional)
        let char_encoding = if has_char_encoding {
            let e = data[pos];
            pos += 1;
            e
        } else {
            0 // ASCII
        };

        // Length of link name
        let name_len_size = 1 << size_of_len_of_link_name; // 1, 2, 4, or 8
        let name_len = read_le_u64(&data[pos..], name_len_size) as usize;
        pos += name_len_size;

        // Link name
        if pos + name_len > data.len() {
            return Err(Error::InvalidFormat(
                "link name exceeds message bounds".into(),
            ));
        }
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Link value based on type
        let mut hard_link_addr = None;
        let mut soft_link_target = None;
        let mut external_link = None;

        match link_type {
            LinkType::Hard => {
                if pos + sizeof_addr as usize > data.len() {
                    return Err(Error::InvalidFormat(
                        "hard link addr exceeds message bounds".into(),
                    ));
                }
                hard_link_addr = Some(read_le_u64(&data[pos..], sizeof_addr as usize));
            }
            LinkType::Soft => {
                if pos + 2 > data.len() {
                    return Err(Error::InvalidFormat(
                        "soft link target length exceeds bounds".into(),
                    ));
                }
                let target_len = read_le_u64(&data[pos..], 2) as usize;
                pos += 2;
                if pos + target_len > data.len() {
                    return Err(Error::InvalidFormat(
                        "soft link target exceeds bounds".into(),
                    ));
                }
                soft_link_target =
                    Some(String::from_utf8_lossy(&data[pos..pos + target_len]).to_string());
            }
            LinkType::External => {
                if pos + 2 > data.len() {
                    return Err(Error::InvalidFormat(
                        "external link info length exceeds bounds".into(),
                    ));
                }
                let info_len = read_le_u64(&data[pos..], 2) as usize;
                pos += 2;
                if pos + info_len > data.len() || info_len < 2 {
                    return Err(Error::InvalidFormat(
                        "external link info exceeds bounds".into(),
                    ));
                }
                let ext_version = data[pos];
                pos += 1;
                let mut info_consumed = 1;
                if ext_version >= 1 {
                    let _ext_flags = data[pos];
                    pos += 1;
                    info_consumed += 1;
                }
                let end = (pos + info_len - info_consumed).min(data.len());

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

fn read_le_u64(data: &[u8], size: usize) -> u64 {
    let mut val = 0u64;
    for i in 0..size.min(8).min(data.len()) {
        val |= (data[i] as u64) << (i * 8);
    }
    val
}
