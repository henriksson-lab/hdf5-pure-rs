use crate::error::{Error, Result};

const MAX_LAYOUT_RANK: usize = 32;

/// Storage layout type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutClass {
    Compact,    // 0
    Contiguous, // 1
    Chunked,    // 2
    Virtual,    // 3
}

/// Chunk index type (v4+ layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIndexType {
    BTreeV1,         // 0 - legacy
    SingleChunk,     // 1
    Implicit,        // 2
    FixedArray,      // 3
    ExtensibleArray, // 4
    BTreeV2,         // 5
}

/// Parsed Data Layout message (type 0x0008).
#[derive(Debug, Clone)]
pub struct DataLayoutMessage {
    pub version: u8,
    pub layout_class: LayoutClass,
    /// For compact: raw data bytes.
    pub compact_data: Option<Vec<u8>>,
    /// For contiguous: data address.
    pub contiguous_addr: Option<u64>,
    /// For contiguous: data size.
    pub contiguous_size: Option<u64>,
    /// For chunked: chunk dimensions.
    pub chunk_dims: Option<Vec<u64>>,
    /// For chunked (v3): address of the chunk index (B-tree).
    pub chunk_index_addr: Option<u64>,
    /// For chunked (v4+): chunk index type.
    pub chunk_index_type: Option<ChunkIndexType>,
    /// For chunked: data element size stored in chunk dims (v1/v2).
    pub chunk_element_size: Option<u32>,
    /// For chunked (v4): flags.
    pub chunk_flags: Option<u8>,
    /// For chunked (v4): encoded chunk dimensions.
    pub chunk_encoded_dims: Option<Vec<u64>>,
    /// For single chunk (v4): filtered chunk size.
    pub single_chunk_filtered_size: Option<u64>,
    /// For single chunk (v4): filter mask.
    pub single_chunk_filter_mask: Option<u32>,
    /// For single chunk or contiguous with address.
    pub data_addr: Option<u64>,
    /// For virtual datasets: address of the global heap storing virtual mapping.
    pub virtual_heap_addr: Option<u64>,
    /// For virtual datasets: index into the global heap.
    pub virtual_heap_index: Option<u32>,
}

impl DataLayoutMessage {
    pub fn decode(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidFormat("empty data layout message".into()));
        }

        let version = data[0];
        let result = match version {
            1 | 2 => Self::decode_v1_v2(data, version, sizeof_addr),
            3 => Self::decode_v3(data, sizeof_addr, sizeof_size),
            4 => Self::decode_v4(data, sizeof_addr, sizeof_size),
            _ => Err(Error::InvalidFormat(format!(
                "data layout message version {version}"
            ))),
        };

        #[cfg(feature = "tracehash")]
        if let Ok(message) = &result {
            let mut th = tracehash::th_call!("hdf5.data_layout.decode");
            th.input_bytes(data);
            th.output_bool(true);
            th.output_u64(message.version as u64);
            th.output_u64(message.layout_class as u64);
            th.output_u64(message.chunk_index_type.map(|t| t as u64).unwrap_or(0));
            th.finish();
        }

        result
    }

    fn decode_v1_v2(data: &[u8], version: u8, sizeof_addr: u8) -> Result<Self> {
        let sa = sizeof_addr as usize;
        ensure_available(data, 0, 8, "data layout v1/v2 header")?;
        let ndims = data[1] as usize;
        if ndims > MAX_LAYOUT_RANK {
            return Err(Error::InvalidFormat(format!(
                "data layout rank {ndims} exceeds supported maximum {MAX_LAYOUT_RANK}"
            )));
        }
        let layout_class_val = data[2];
        // 5 reserved bytes in v1/v2
        let mut pos = 8;

        let layout_class = match layout_class_val {
            0 => LayoutClass::Compact,
            1 => LayoutClass::Contiguous,
            2 => LayoutClass::Chunked,
            _ => {
                return Err(Error::InvalidFormat(format!(
                    "unknown layout class {layout_class_val}"
                )))
            }
        };

        let data_addr = if layout_class != LayoutClass::Compact {
            let addr = read_le_u64(data, &mut pos, sa, "data layout v1/v2 address")?;
            Some(addr)
        } else {
            None
        };

        // Dimension sizes (ndims * 4 bytes for v1/v2, last dim is element size for chunked)
        let mut dims = Vec::new();
        for _ in 0..ndims {
            let d = read_u32_le(data, &mut pos, "data layout v1/v2 dimensions")? as u64;
            dims.push(d);
        }

        let mut result = DataLayoutMessage {
            version,
            layout_class,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: None,
            chunk_index_addr: None,
            chunk_index_type: None,
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr,
            virtual_heap_addr: None,
            virtual_heap_index: None,
        };

        match layout_class {
            LayoutClass::Compact => {
                let compact_size =
                    read_u32_le(data, &mut pos, "data layout v1/v2 compact size")? as usize;
                ensure_available(data, pos, compact_size, "data layout v1/v2 compact data")?;
                result.compact_data = Some(data[pos..pos + compact_size].to_vec());
            }
            LayoutClass::Contiguous => {
                // For v1/v2, size is in the dimensions
                result.contiguous_addr = data_addr;
                if !dims.is_empty() {
                    result.contiguous_size = Some(dims.iter().try_fold(1u64, |acc, &dim| {
                        acc.checked_mul(dim).ok_or_else(|| {
                            Error::InvalidFormat(
                                "data layout v1/v2 contiguous size overflow".into(),
                            )
                        })
                    })?);
                }
            }
            LayoutClass::Chunked => {
                result.chunk_index_addr = data_addr;
                if let Some(&last) = dims.last() {
                    result.chunk_element_size = Some(last as u32);
                    result.chunk_dims = Some(dims[..dims.len() - 1].to_vec());
                }
            }
            _ => {}
        }

        Ok(result)
    }

    fn decode_v3(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<Self> {
        let sa = sizeof_addr as usize;
        let ss = sizeof_size as usize;
        ensure_available(data, 0, 2, "data layout v3 header")?;
        let layout_class_val = data[1];
        let mut pos = 2;

        let layout_class = match layout_class_val {
            0 => LayoutClass::Compact,
            1 => LayoutClass::Contiguous,
            2 => LayoutClass::Chunked,
            3 => LayoutClass::Virtual,
            _ => {
                return Err(Error::InvalidFormat(format!(
                    "unknown layout class {layout_class_val}"
                )))
            }
        };

        let mut result = DataLayoutMessage {
            version: 3,
            layout_class,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: None,
            chunk_index_addr: None,
            chunk_index_type: None,
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr: None,
            virtual_heap_addr: None,
            virtual_heap_index: None,
        };

        match layout_class {
            LayoutClass::Compact => {
                let size = read_u16_le(data, &mut pos, "data layout v3 compact size")? as usize;
                ensure_available(data, pos, size, "data layout v3 compact data")?;
                result.compact_data = Some(data[pos..pos + size].to_vec());
            }
            LayoutClass::Contiguous => {
                let addr = read_le_u64(data, &mut pos, sa, "data layout v3 contiguous address")?;
                let size = read_le_u64(data, &mut pos, ss, "data layout v3 contiguous size")?;
                result.contiguous_addr = Some(addr);
                result.contiguous_size = Some(size);
                result.data_addr = Some(addr);
            }
            LayoutClass::Chunked => {
                // v3 chunked: ndims(1) + dims(ndims*4) + btree_addr(sizeof_addr)
                let ndims = read_u8(data, &mut pos, "data layout v3 chunk rank")? as usize;
                if ndims > MAX_LAYOUT_RANK {
                    return Err(Error::InvalidFormat(format!(
                        "data layout v3 chunk rank {ndims} exceeds supported maximum {MAX_LAYOUT_RANK}"
                    )));
                }
                let addr = read_le_u64(data, &mut pos, sa, "data layout v3 chunk index address")?;

                let mut dims = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    let d = read_u32_le(data, &mut pos, "data layout v3 chunk dimensions")? as u64;
                    dims.push(d);
                }

                // Last dimension is the element size
                if let Some(&last) = dims.last() {
                    result.chunk_element_size = Some(last as u32);
                    result.chunk_dims = Some(dims[..dims.len() - 1].to_vec());
                }

                result.chunk_index_addr = Some(addr);
                result.data_addr = Some(addr);
            }
            LayoutClass::Virtual => {
                // v3 virtual: global_heap_addr(sizeof_addr) + index(4)
                let addr = read_le_u64(data, &mut pos, sa, "data layout v3 virtual heap address")?;
                let index = read_u32_le(data, &mut pos, "data layout v3 virtual heap index")?;
                result.virtual_heap_addr = Some(addr);
                result.virtual_heap_index = Some(index);
            }
        }

        Ok(result)
    }

    fn decode_v4(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<Self> {
        let sa = sizeof_addr as usize;
        let ss = sizeof_size as usize;
        ensure_available(data, 0, 2, "data layout v4 header")?;
        let layout_class_val = data[1];
        let mut pos = 2;

        let layout_class = match layout_class_val {
            0 => LayoutClass::Compact,
            1 => LayoutClass::Contiguous,
            2 => LayoutClass::Chunked,
            3 => LayoutClass::Virtual,
            _ => {
                return Err(Error::InvalidFormat(format!(
                    "unknown layout class {layout_class_val}"
                )))
            }
        };

        let mut result = DataLayoutMessage {
            version: 4,
            layout_class,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: None,
            chunk_index_addr: None,
            chunk_index_type: None,
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr: None,
            virtual_heap_addr: None,
            virtual_heap_index: None,
        };

        match layout_class {
            LayoutClass::Compact => {
                let size = read_u16_le(data, &mut pos, "data layout v4 compact size")? as usize;
                ensure_available(data, pos, size, "data layout v4 compact data")?;
                result.compact_data = Some(data[pos..pos + size].to_vec());
            }
            LayoutClass::Contiguous => {
                let addr = read_le_u64(data, &mut pos, sa, "data layout v4 contiguous address")?;
                let size = read_le_u64(data, &mut pos, ss, "data layout v4 contiguous size")?;
                result.contiguous_addr = Some(addr);
                result.contiguous_size = Some(size);
                result.data_addr = Some(addr);
            }
            LayoutClass::Chunked => {
                let flags = read_u8(data, &mut pos, "data layout v4 chunk flags")?;
                let ndims = read_u8(data, &mut pos, "data layout v4 chunk rank")? as usize;
                if ndims > MAX_LAYOUT_RANK {
                    return Err(Error::InvalidFormat(format!(
                        "data layout v4 chunk rank {ndims} exceeds supported maximum {MAX_LAYOUT_RANK}"
                    )));
                }
                let enc_bytes_per_dim =
                    read_u8(data, &mut pos, "data layout v4 encoded dimension size")? as usize;

                // Read chunk dimensions (variable-width encoded)
                let mut dims = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    let d = read_le_u64(
                        data,
                        &mut pos,
                        enc_bytes_per_dim,
                        "data layout v4 chunk dimensions",
                    )?;
                    dims.push(d);
                }
                result.chunk_dims = Some(dims);
                result.chunk_flags = Some(flags);

                // Chunk index type
                let idx_type_val = read_u8(data, &mut pos, "data layout v4 chunk index type")?;

                let idx_type = match idx_type_val {
                    0 => ChunkIndexType::BTreeV1,
                    1 => ChunkIndexType::SingleChunk,
                    2 => ChunkIndexType::Implicit,
                    3 => ChunkIndexType::FixedArray,
                    4 => ChunkIndexType::ExtensibleArray,
                    5 => ChunkIndexType::BTreeV2,
                    _ => {
                        return Err(Error::Unsupported(format!(
                            "chunk index type {idx_type_val}"
                        )))
                    }
                };
                result.chunk_index_type = Some(idx_type);

                // Index-specific data
                match idx_type {
                    ChunkIndexType::SingleChunk => {
                        if flags & 0x02 != 0 {
                            // Single chunk with filter
                            let filtered_size = read_le_u64(
                                data,
                                &mut pos,
                                ss,
                                "data layout v4 single chunk filtered size",
                            )?;
                            let filter_mask =
                                read_u32_le(data, &mut pos, "data layout v4 single chunk mask")?;
                            result.single_chunk_filtered_size = Some(filtered_size);
                            result.single_chunk_filter_mask = Some(filter_mask);
                        }
                        let addr =
                            read_le_u64(data, &mut pos, sa, "data layout v4 single chunk address")?;
                        result.chunk_index_addr = Some(addr);
                        result.data_addr = Some(addr);
                    }
                    ChunkIndexType::Implicit => {
                        let addr =
                            read_le_u64(data, &mut pos, sa, "data layout v4 implicit address")?;
                        result.chunk_index_addr = Some(addr);
                        result.data_addr = Some(addr);
                    }
                    ChunkIndexType::FixedArray => {
                        // Creation parameters: page_bits(1)
                        let _page_bits =
                            read_u8(data, &mut pos, "data layout v4 fixed array page bits")?;
                        let addr =
                            read_le_u64(data, &mut pos, sa, "data layout v4 fixed array address")?;
                        result.chunk_index_addr = Some(addr);
                        result.data_addr = Some(addr);
                    }
                    ChunkIndexType::ExtensibleArray => {
                        // Creation parameters: several bytes
                        ensure_available(
                            data,
                            pos,
                            5,
                            "data layout v4 extensible array parameters",
                        )?;
                        pos += 5;
                        let addr = read_le_u64(
                            data,
                            &mut pos,
                            sa,
                            "data layout v4 extensible array address",
                        )?;
                        result.chunk_index_addr = Some(addr);
                        result.data_addr = Some(addr);
                    }
                    ChunkIndexType::BTreeV2 => {
                        let _node_size =
                            read_u32_le(data, &mut pos, "data layout v4 btree2 node size")?;
                        let _split_percent =
                            read_u8(data, &mut pos, "data layout v4 btree2 split percent")?;
                        let _merge_percent =
                            read_u8(data, &mut pos, "data layout v4 btree2 merge percent")?;
                        let addr =
                            read_le_u64(data, &mut pos, sa, "data layout v4 btree2 address")?;
                        result.chunk_index_addr = Some(addr);
                        result.data_addr = Some(addr);
                    }
                    ChunkIndexType::BTreeV1 => {
                        let addr =
                            read_le_u64(data, &mut pos, sa, "data layout v4 btree1 address")?;
                        result.chunk_index_addr = Some(addr);
                        result.data_addr = Some(addr);
                    }
                }
            }
            LayoutClass::Virtual => {
                // v4 virtual: global_heap_addr(sizeof_addr) + index(4)
                let addr = read_le_u64(data, &mut pos, sa, "data layout v4 virtual heap address")?;
                let index = read_u32_le(data, &mut pos, "data layout v4 virtual heap index")?;
                result.virtual_heap_addr = Some(addr);
                result.virtual_heap_index = Some(index);
            }
        }

        Ok(result)
    }
}

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

fn read_u16_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u16> {
    ensure_available(data, *pos, 2, context)?;
    let value = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(value)
}

fn read_u32_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u32> {
    ensure_available(data, *pos, 4, context)?;
    let value = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
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
