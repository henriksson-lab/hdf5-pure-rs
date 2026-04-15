use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};

/// High-level dataspace descriptor.
#[derive(Debug, Clone)]
pub struct Dataspace {
    msg: DataspaceMessage,
}

impl Dataspace {
    pub(crate) fn from_message(msg: DataspaceMessage) -> Self {
        Self { msg }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.msg.ndims as usize
    }

    /// Current dimension sizes.
    pub fn shape(&self) -> &[u64] {
        &self.msg.dims
    }

    /// Maximum dimension sizes (None if same as current).
    pub fn maxdims(&self) -> Option<&[u64]> {
        self.msg.max_dims.as_deref()
    }

    /// Total number of elements.
    pub fn size(&self) -> u64 {
        if self.msg.dims.is_empty() {
            if self.msg.space_type == DataspaceType::Scalar {
                1
            } else {
                0
            }
        } else {
            self.msg.dims.iter().product()
        }
    }

    /// Whether this is a scalar dataspace.
    pub fn is_scalar(&self) -> bool {
        self.msg.space_type == DataspaceType::Scalar
    }

    /// Whether this is a null dataspace (no data).
    pub fn is_null(&self) -> bool {
        self.msg.space_type == DataspaceType::Null
    }

    /// Whether this is a simple (N-dimensional) dataspace.
    pub fn is_simple(&self) -> bool {
        self.msg.space_type == DataspaceType::Simple
    }

    /// Whether any dimension is resizable (has unlimited max dim).
    pub fn is_resizable(&self) -> bool {
        if let Some(maxdims) = &self.msg.max_dims {
            maxdims.iter().any(|&d| d == u64::MAX)
        } else {
            false
        }
    }
}
