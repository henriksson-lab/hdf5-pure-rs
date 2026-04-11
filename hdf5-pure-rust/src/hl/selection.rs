use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// A selection specifies which elements to read from a dataset.
#[derive(Debug, Clone)]
pub enum Selection {
    /// All elements.
    All,
    /// A contiguous range along each dimension.
    Slice(Vec<SliceInfo>),
}

/// Slice specification for one dimension.
#[derive(Debug, Clone)]
pub struct SliceInfo {
    pub start: u64,
    pub end: u64,    // exclusive
    pub step: u64,   // must be 1 for now
}

impl SliceInfo {
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end, step: 1 }
    }

    pub fn count(&self) -> u64 {
        if self.end > self.start {
            (self.end - self.start + self.step - 1) / self.step
        } else {
            0
        }
    }
}

/// Trait for types that can be converted to a Selection.
pub trait IntoSelection {
    fn into_selection(self, shape: &[u64]) -> Selection;
}

impl IntoSelection for Selection {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        self
    }
}

// For 1D: Range<usize>
impl IntoSelection for Range<usize> {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::Slice(vec![SliceInfo::new(self.start as u64, self.end as u64)])
    }
}

// For 1D: RangeFull (..)
impl IntoSelection for RangeFull {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::All
    }
}

// For 1D: RangeFrom (start..)
impl IntoSelection for RangeFrom<usize> {
    fn into_selection(self, shape: &[u64]) -> Selection {
        let end = if shape.is_empty() { 0 } else { shape[0] };
        Selection::Slice(vec![SliceInfo::new(self.start as u64, end)])
    }
}

// For 1D: RangeTo (..end)
impl IntoSelection for RangeTo<usize> {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::Slice(vec![SliceInfo::new(0, self.end as u64)])
    }
}

// Tuples for multi-D: (Range, Range)
impl IntoSelection for (Range<usize>, Range<usize>) {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::Slice(vec![
            SliceInfo::new(self.0.start as u64, self.0.end as u64),
            SliceInfo::new(self.1.start as u64, self.1.end as u64),
        ])
    }
}

impl Selection {
    /// Compute the output shape for this selection given the dataset shape.
    pub fn output_shape(&self, ds_shape: &[u64]) -> Vec<u64> {
        match self {
            Selection::All => ds_shape.to_vec(),
            Selection::Slice(slices) => {
                slices.iter().map(|s| s.count()).collect()
            }
        }
    }

    /// Normalize into concrete SliceInfo per dimension.
    pub fn to_slices(&self, ds_shape: &[u64]) -> Vec<SliceInfo> {
        match self {
            Selection::All => {
                ds_shape.iter().map(|&d| SliceInfo::new(0, d)).collect()
            }
            Selection::Slice(slices) => slices.clone(),
        }
    }
}
