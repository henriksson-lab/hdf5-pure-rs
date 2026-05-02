use crate::error::{Error, Result};
use crate::format::messages::data_layout::LayoutClass;

use super::{usize_from_u64, Dataset};

impl Dataset {
    /// Read a subset of the dataset using a selection.
    ///
    /// Example: `ds.read_slice::<f64>(10..20)` reads elements 10-19 from a 1D dataset.
    pub fn read_slice<T: crate::hl::types::H5Type, S: crate::hl::selection::IntoSelection>(
        &self,
        sel: S,
    ) -> Result<Vec<T>> {
        let shape = self.shape()?;
        let selection = sel.into_selection(&shape);
        if matches!(selection, crate::hl::selection::Selection::None) {
            return Ok(Vec::new());
        }
        if let crate::hl::selection::Selection::Points(points) = &selection {
            Self::validate_selection_points(&shape, points)?;
            let all_data = self.read::<T>()?;
            return Self::extract_point_selection(&all_data, &shape, points);
        }
        if let crate::hl::selection::Selection::Hyperslab(dims) = &selection {
            Self::validate_hyperslab_selection(&shape, dims)?;
            let out_shape = selection.output_shape(&shape);
            let total_out = Self::selection_output_elements(&out_shape)?;
            if total_out == 0 {
                return Ok(Vec::new());
            }
            let total_out_usize = usize_from_u64(total_out, "hyperslab selection element count")?;
            let all_data = self.read::<T>()?;
            return Self::extract_hyperslab_selection(
                &all_data,
                &shape,
                dims,
                &out_shape,
                total_out,
                total_out_usize,
            );
        }

        let slices = selection.to_slices(&shape);
        Self::validate_selection_slices(&shape, &slices)?;
        let out_shape = selection.output_shape(&shape);
        let total_out = Self::selection_output_elements(&out_shape)?;
        let total_out_usize = usize_from_u64(total_out, "selection element count")?;
        let elem_size = T::type_size();

        if total_out == 0 {
            return Ok(Vec::new());
        }

        if let Some(raw) = self.try_read_slice_contiguous_1d(&shape, &slices, elem_size)? {
            return crate::hl::types::bytes_to_vec(raw);
        }

        let all_data = self.read::<T>()?;
        if shape.is_empty() {
            return Ok(all_data);
        }

        if shape.len() == 1 && slices.len() == 1 {
            return Self::extract_1d_selection(&all_data, &slices[0]);
        }

        Self::extract_nd_selection(
            &all_data,
            &shape,
            &slices,
            &out_shape,
            total_out,
            total_out_usize,
        )
    }

    fn selection_output_elements(out_shape: &[u64]) -> Result<u64> {
        if out_shape.is_empty() {
            return Ok(1);
        }
        out_shape.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("selection element count overflow".into()))
        })
    }

    fn validate_selection_slices(
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
    ) -> Result<()> {
        if slices.len() != shape.len() {
            return Err(Error::InvalidFormat(format!(
                "selection rank {} does not match dataset rank {}",
                slices.len(),
                shape.len()
            )));
        }
        for (dim, (slice, &extent)) in slices.iter().zip(shape).enumerate() {
            if slice.step == 0 {
                return Err(Error::InvalidFormat(format!(
                    "selection dimension {dim} has zero step"
                )));
            }
            if slice.start > extent || slice.end > extent {
                return Err(Error::InvalidFormat(format!(
                    "selection dimension {dim} range {}..{} exceeds extent {extent}",
                    slice.start, slice.end
                )));
            }
        }
        Ok(())
    }

    fn validate_hyperslab_selection(
        shape: &[u64],
        dims: &[crate::hl::selection::HyperslabDim],
    ) -> Result<()> {
        if dims.len() != shape.len() {
            return Err(Error::InvalidFormat(format!(
                "hyperslab rank {} does not match dataset rank {}",
                dims.len(),
                shape.len()
            )));
        }
        for (dim, (selection, &extent)) in dims.iter().zip(shape).enumerate() {
            if selection.stride == 0 {
                return Err(Error::InvalidFormat(format!(
                    "hyperslab dimension {dim} has zero stride"
                )));
            }
            if selection.block == 0 && selection.count != 0 {
                return Err(Error::InvalidFormat(format!(
                    "hyperslab dimension {dim} has zero block"
                )));
            }
            if selection.count == 0 || selection.block == 0 {
                continue;
            }
            let span_start = selection
                .count
                .checked_sub(1)
                .and_then(|count_minus_one| count_minus_one.checked_mul(selection.stride))
                .and_then(|offset| selection.start.checked_add(offset))
                .ok_or_else(|| Error::InvalidFormat("hyperslab extent overflow".into()))?;
            let span_end = span_start
                .checked_add(selection.block)
                .ok_or_else(|| Error::InvalidFormat("hyperslab extent overflow".into()))?;
            if span_end > extent {
                return Err(Error::InvalidFormat(format!(
                    "hyperslab dimension {dim} exceeds extent {extent}"
                )));
            }
        }
        Ok(())
    }

    fn validate_selection_points(shape: &[u64], points: &[Vec<u64>]) -> Result<()> {
        for point in points {
            if point.len() != shape.len() {
                return Err(Error::InvalidFormat(format!(
                    "point selection rank {} does not match dataset rank {}",
                    point.len(),
                    shape.len()
                )));
            }
            for (dim, (&coord, &extent)) in point.iter().zip(shape).enumerate() {
                if coord >= extent {
                    return Err(Error::InvalidFormat(format!(
                        "point selection coordinate {coord} in dimension {dim} exceeds extent {extent}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn extract_point_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        points: &[Vec<u64>],
    ) -> Result<Vec<T>> {
        let strides = Self::row_major_strides(shape)?;
        let mut result = Vec::with_capacity(points.len());
        for point in points {
            let index = Self::linear_index(point, &strides)?;
            if index < all_data.len() {
                result.push(all_data[index]);
            }
        }
        Ok(result)
    }

    fn extract_hyperslab_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        dims: &[crate::hl::selection::HyperslabDim],
        out_shape: &[u64],
        total_out: u64,
        total_out_usize: usize,
    ) -> Result<Vec<T>> {
        let strides = Self::row_major_strides(shape)?;
        let ndims = shape.len();
        let mut result = Vec::with_capacity(total_out_usize);
        let mut out_idx = vec![0u64; ndims];
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for dim in 0..ndims {
                let selected_block = out_idx[dim] / dims[dim].block;
                let selected_offset = out_idx[dim] % dims[dim].block;
                let in_d =
                    dims[dim]
                        .start
                        .checked_add(selected_block.checked_mul(dims[dim].stride).ok_or_else(
                            || Error::InvalidFormat("hyperslab coordinate overflow".into()),
                        )?)
                        .and_then(|coord| coord.checked_add(selected_offset))
                        .ok_or_else(|| {
                            Error::InvalidFormat("hyperslab coordinate overflow".into())
                        })?;
                let term = usize_from_u64(in_d, "hyperslab input index")?
                    .checked_mul(strides[dim])
                    .ok_or_else(|| {
                        Error::InvalidFormat("hyperslab linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("hyperslab linear index overflow".into())
                })?;
            }
            if in_linear < all_data.len() {
                result.push(all_data[in_linear]);
            }
            for dim in (0..ndims).rev() {
                out_idx[dim] = out_idx[dim].checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("hyperslab output index overflow".into())
                })?;
                if out_idx[dim] < out_shape[dim] {
                    break;
                }
                out_idx[dim] = 0;
            }
        }
        Ok(result)
    }

    fn try_read_slice_contiguous_1d(
        &self,
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        elem_size: usize,
    ) -> Result<Option<Vec<u8>>> {
        if !(shape.len() == 1 && slices.len() == 1 && slices[0].step == 1) {
            return Ok(None);
        }

        let info = self.info()?;
        if info.layout.layout_class != LayoutClass::Contiguous {
            return Ok(None);
        }

        let Some(addr) = info.layout.contiguous_addr else {
            return Ok(None);
        };
        if crate::io::reader::is_undef_addr(addr) {
            return Ok(None);
        }

        let start_byte = usize_from_u64(slices[0].start, "selection start")?
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("selection byte offset overflow".into()))?;
        let nbytes = usize_from_u64(slices[0].count(), "selection count")?
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("selection byte count overflow".into()))?;
        let read_addr = addr
            .checked_add(start_byte as u64)
            .ok_or_else(|| Error::InvalidFormat("selection read address overflow".into()))?;

        let mut guard = self.inner.lock();
        guard.reader.seek(read_addr)?;
        Ok(Some(guard.reader.read_bytes(nbytes)?))
    }

    fn extract_1d_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        slice: &crate::hl::selection::SliceInfo,
    ) -> Result<Vec<T>> {
        let start = usize_from_u64(slice.start, "selection start")?;
        let end = usize_from_u64(slice.end, "selection end")?;
        if start > all_data.len() {
            return Ok(Vec::new());
        }
        let end = end.min(all_data.len());
        if slice.step == 1 {
            return Ok(all_data[start..end].to_vec());
        }
        let step = usize_from_u64(slice.step, "selection step")?;
        Ok(all_data[start..end].iter().step_by(step).copied().collect())
    }

    fn extract_nd_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        out_shape: &[u64],
        total_out: u64,
        total_out_usize: usize,
    ) -> Result<Vec<T>> {
        let mut result = Vec::with_capacity(total_out_usize);
        let ndims = shape.len();

        let mut in_strides = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            in_strides[d] = in_strides[d + 1]
                .checked_mul(usize_from_u64(shape[d + 1], "selection shape")?)
                .ok_or_else(|| Error::InvalidFormat("selection stride overflow".into()))?;
        }

        let mut out_idx = vec![0u64; ndims];
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for d in 0..ndims {
                let in_d = out_idx[d]
                    .checked_mul(slices[d].step)
                    .and_then(|offset| slices[d].start.checked_add(offset))
                    .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))?;
                let term = usize_from_u64(in_d, "selection input index")?
                    .checked_mul(in_strides[d])
                    .ok_or_else(|| {
                        Error::InvalidFormat("selection linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("selection linear index overflow".into())
                })?;
            }

            if in_linear < all_data.len() {
                result.push(all_data[in_linear]);
            }

            for d in (0..ndims).rev() {
                out_idx[d] = out_idx[d].checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("selection output index overflow".into())
                })?;
                if out_idx[d] < out_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        Ok(result)
    }

    pub(super) fn row_major_strides(dims: &[u64]) -> Result<Vec<usize>> {
        let mut strides = vec![1usize; dims.len()];
        for dim in (0..dims.len().saturating_sub(1)).rev() {
            strides[dim] = strides[dim + 1]
                .checked_mul(usize_from_u64(dims[dim + 1], "dataspace dimension")?)
                .ok_or_else(|| Error::InvalidFormat("dataspace stride overflow".into()))?;
        }
        Ok(strides)
    }

    pub(super) fn linear_index(coords: &[u64], strides: &[usize]) -> Result<usize> {
        coords
            .iter()
            .zip(strides)
            .try_fold(0usize, |acc, (&coord, &stride)| {
                acc.checked_add(
                    usize_from_u64(coord, "dataspace coordinate")
                        .ok()?
                        .checked_mul(stride)?,
                )
            })
            .ok_or_else(|| Error::InvalidFormat("linear index overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hl::selection::SliceInfo;

    #[test]
    fn nd_slice_extraction_rejects_coordinate_overflow() {
        let slices = [SliceInfo {
            start: u64::MAX - 1,
            end: u64::MAX,
            step: 2,
        }];
        let err = Dataset::extract_nd_selection::<u8>(&[0], &[u64::MAX], &slices, &[2], 2, 2)
            .unwrap_err();
        assert!(
            err.to_string().contains("selection coordinate overflow"),
            "unexpected error: {err}"
        );
    }
}
