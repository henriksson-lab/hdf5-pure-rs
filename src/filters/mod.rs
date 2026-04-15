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
    let mut buf = data.to_vec();

    // Apply filters in reverse order
    for filter in pipeline.filters.iter().rev() {
        buf = apply_filter_reverse(&buf, filter, element_size)?;
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
