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
    apply_pipeline_reverse_with_mask_and_expected(data, pipeline, element_size, filter_mask, None)
}

pub fn apply_pipeline_reverse_with_mask_expected(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: usize,
) -> Result<Vec<u8>> {
    apply_pipeline_reverse_with_mask_and_expected(
        data,
        pipeline,
        element_size,
        filter_mask,
        Some(expected_len),
    )
}

fn apply_pipeline_reverse_with_mask_and_expected(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: Option<usize>,
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

    if let Some(expected_len) = expected_len {
        if buf.len() != expected_len {
            return Err(Error::InvalidFormat(format!(
                "filter pipeline output length mismatch: expected {expected_len}, got {}",
                buf.len()
            )));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn deflate_pipeline() -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_DEFLATE,
                name: Some("deflate".into()),
                flags: 0,
                client_data: vec![4],
            }],
        }
    }

    fn unknown_pipeline(flags: u16) -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: 32_099,
                name: Some("unknown".into()),
                flags,
                client_data: Vec::new(),
            }],
        }
    }

    #[test]
    fn expected_length_accepts_exact_filter_output() {
        let compressed = deflate::compress(b"abcd", 4).unwrap();
        let out =
            apply_pipeline_reverse_with_mask_expected(&compressed, &deflate_pipeline(), 1, 0, 4)
                .unwrap();
        assert_eq!(out, b"abcd");
    }

    #[test]
    fn expected_length_rejects_filter_output_mismatch() {
        let compressed = deflate::compress(b"abcd", 4).unwrap();
        let err =
            apply_pipeline_reverse_with_mask_expected(&compressed, &deflate_pipeline(), 1, 0, 3)
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("filter pipeline output length mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn masked_unknown_optional_filter_is_skipped() {
        let out = apply_pipeline_reverse_with_mask(b"abcd", &unknown_pipeline(1), 1, 0b1).unwrap();
        assert_eq!(out, b"abcd");
    }

    #[test]
    fn unmasked_unknown_optional_filter_fails() {
        let err =
            apply_pipeline_reverse_with_mask(b"abcd", &unknown_pipeline(1), 1, 0).unwrap_err();
        assert!(
            err.to_string().contains("filter 32099 not implemented"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unmasked_unknown_required_filter_fails() {
        let err =
            apply_pipeline_reverse_with_mask(b"abcd", &unknown_pipeline(0), 1, 0).unwrap_err();
        assert!(
            err.to_string().contains("filter 32099 not implemented"),
            "unexpected error: {err}"
        );
    }
}
