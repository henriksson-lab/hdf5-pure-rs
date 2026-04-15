pub mod blosc;
pub mod deflate;
pub mod fletcher32;
pub mod lzf;
pub mod nbit;
pub mod scaleoffset;
pub mod shuffle;
pub mod szip;

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP,
};

/// Apply filter pipeline in reverse (for reading/decompression).
/// Filters are applied in reverse order of their definition.
pub fn apply_pipeline_reverse(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
) -> Result<Vec<u8>> {
    apply_pipeline_reverse_with_mask(data, pipeline, element_size, 0)
}

/// Apply filter pipeline in reverse while honoring an HDF5 per-chunk filter mask.
/// Bit `i` set means filter `i` in the stored pipeline was not applied to the chunk.
pub fn apply_pipeline_reverse_with_mask(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
) -> Result<Vec<u8>> {
    if pipeline.filters.len() > u32::BITS as usize {
        return Err(Error::InvalidFormat(format!(
            "filter pipeline length {} exceeds 32-bit chunk filter mask",
            pipeline.filters.len()
        )));
    }

    let valid_mask = if pipeline.filters.len() >= u32::BITS as usize {
        u32::MAX
    } else {
        (1u32 << pipeline.filters.len()) - 1
    };
    if filter_mask & !valid_mask != 0 {
        return Err(Error::InvalidFormat(format!(
            "filter mask {filter_mask:#x} references filters outside pipeline length {}",
            pipeline.filters.len()
        )));
    }

    #[cfg(feature = "tracehash")]
    let mut th = {
        let mut th = tracehash::th_call!("hdf5.filter_pipeline.apply");
        th.input_u64(pipeline.filters.len() as u64);
        th.input_u64(0x0100);
        th.input_u64(filter_mask as u64);
        th.input_u64(data.len() as u64);
        th
    };

    let mut buf = data.to_vec();

    // Apply filters in reverse order
    for (index, filter) in pipeline.filters.iter().enumerate().rev() {
        if filter_mask & (1u32 << index) != 0 {
            continue;
        }
        buf = apply_filter_reverse(&buf, filter, element_size)?;
    }

    #[cfg(feature = "tracehash")]
    {
        th.output_bool(true);
        th.output_u64(0);
        th.output_u64(buf.len() as u64);
        th.finish();
    }

    Ok(buf)
}

fn apply_filter_reverse(data: &[u8], filter: &FilterDesc, element_size: usize) -> Result<Vec<u8>> {
    match filter.id {
        FILTER_DEFLATE => deflate::decompress(data),
        FILTER_SHUFFLE => shuffle::unshuffle(data, element_size),
        FILTER_FLETCHER32 => fletcher32::verify_and_strip(data),
        FILTER_NBIT => nbit::decompress(data, &filter.client_data),
        FILTER_SCALEOFFSET => scaleoffset::decompress(data, &filter.client_data),
        FILTER_SZIP => szip::decompress(data),
        32001 => blosc::decompress(data), // HDF5 Blosc filter ID
        32000 => {
            // LZF filter -- need the uncompressed size
            // LZF stores the original size in the first client_data parameter
            let expected = filter
                .client_data
                .first()
                .copied()
                .unwrap_or(data.len() as u32 * 2);
            lzf::decompress(data, expected as usize)
        }
        _ => Err(Error::Unsupported(format!(
            "filter {} not implemented",
            filter.id
        ))),
    }
}
