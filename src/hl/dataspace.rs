use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};

/// High-level dataspace descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataspace {
    msg: DataspaceMessage,
}

impl Dataspace {
    /// Initialize dataspace package support.
    pub fn init() -> bool {
        true
    }

    /// Internal dataspace package initialization alias.
    pub fn init_package() -> bool {
        Self::init()
    }

    /// Top-level dataspace package termination alias.
    pub fn top_term_package() {}

    /// Dataspace package termination alias.
    pub fn term_package() {}

    /// Create a scalar, simple, or null dataspace with no simple dimensions.
    pub fn create(space_type: DataspaceType) -> Self {
        let dims = Vec::new();
        Self {
            msg: DataspaceMessage {
                version: 2,
                space_type,
                ndims: 0,
                dims,
                max_dims: None,
            },
        }
    }

    /// Create a scalar dataspace.
    pub fn scalar() -> Self {
        Self::create(DataspaceType::Scalar)
    }

    /// Create a null dataspace.
    pub fn null() -> Self {
        Self::create(DataspaceType::Null)
    }

    /// Create a simple dataspace from current and optional maximum dimensions.
    pub fn simple(dims: Vec<u64>, max_dims: Option<Vec<u64>>) -> crate::Result<Self> {
        let ndims = u8::try_from(dims.len())
            .map_err(|_| crate::Error::InvalidFormat("dataspace rank exceeds u8::MAX".into()))?;
        if let Some(max_dims) = max_dims.as_ref() {
            if max_dims.len() != dims.len() {
                return Err(crate::Error::InvalidFormat(
                    "dataspace max dimensions rank does not match current dimensions".into(),
                ));
            }
        }
        Ok(Self {
            msg: DataspaceMessage {
                version: 2,
                space_type: DataspaceType::Simple,
                ndims,
                dims,
                max_dims,
            },
        })
    }

    pub(crate) fn from_message(msg: DataspaceMessage) -> Self {
        Self { msg }
    }

    /// Close callback alias. The pure Rust dataspace is consumed.
    pub fn close_cb(self) {}

    /// Return the parsed low-level dataspace message.
    pub fn raw_message(&self) -> DataspaceMessage {
        self.msg.clone()
    }

    /// Return the parsed simple extent metadata.
    pub fn simple_extent(&self) -> DataspaceMessage {
        self.raw_message()
    }

    /// Explicit dataspace copy operation.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.msg.ndims as usize
    }

    /// Internal simple-extent rank helper.
    pub fn simple_extent_ndims(&self) -> usize {
        self.ndim()
    }

    /// Current dimension sizes.
    pub fn shape(&self) -> &[u64] {
        &self.msg.dims
    }

    /// Maximum dimension sizes (None if same as current).
    pub fn maxdims(&self) -> Option<&[u64]> {
        self.msg.max_dims.as_deref()
    }

    /// Return `(current_dims, max_dims)` extent vectors.
    pub fn extent_dims(&self) -> (&[u64], Option<&[u64]>) {
        (self.shape(), self.maxdims())
    }

    /// Simple-extent dimension helper.
    pub fn simple_extent_dims(&self) -> (&[u64], Option<&[u64]>) {
        self.extent_dims()
    }

    /// Return the dataspace extent type.
    pub fn extent_type(&self) -> DataspaceType {
        self.msg.space_type
    }

    /// Simple-extent type helper.
    pub fn simple_extent_type(&self) -> DataspaceType {
        self.extent_type()
    }

    /// Whether this dataspace has an extent.
    pub fn has_extent(&self) -> bool {
        true
    }

    /// Validate and return a simple dataspace offset vector.
    pub fn offset_simple(&self, offsets: &[i64]) -> crate::Result<Vec<i64>> {
        if offsets.len() != self.ndim() {
            return Err(crate::Error::InvalidFormat(format!(
                "dataspace offset rank {} does not match dataspace rank {}",
                offsets.len(),
                self.ndim()
            )));
        }
        Ok(offsets.to_vec())
    }

    /// Replace this dataspace with a simple extent.
    pub fn set_extent_simple(
        &mut self,
        dims: Vec<u64>,
        max_dims: Option<Vec<u64>>,
    ) -> crate::Result<()> {
        *self = Self::simple(dims, max_dims)?;
        Ok(())
    }

    /// Internal real extent mutation helper.
    pub fn set_extent_real(
        &mut self,
        dims: Vec<u64>,
        max_dims: Option<Vec<u64>>,
    ) -> crate::Result<()> {
        self.set_extent_simple(dims, max_dims)
    }

    /// Total number of elements.
    ///
    /// If the dimension product exceeds `u64::MAX`, this returns `u64::MAX`.
    /// Fallible read/write paths validate shape products and return errors
    /// instead of relying on this display-oriented helper.
    pub fn size(&self) -> u64 {
        if self.msg.dims.is_empty() {
            if self.msg.space_type == DataspaceType::Scalar {
                1
            } else {
                0
            }
        } else {
            self.msg
                .dims
                .iter()
                .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
                .unwrap_or(u64::MAX)
        }
    }

    /// Return the extent element count.
    pub fn extent_nelem(&self) -> u64 {
        self.size()
    }

    /// Return the maximum possible element count if all max dimensions are
    /// finite; returns `u64::MAX` when any max dimension is unlimited or the
    /// product overflows.
    pub fn npoints_max(&self) -> u64 {
        let dims = self.msg.max_dims.as_ref().unwrap_or(&self.msg.dims);
        if dims.iter().any(|&dim| dim == u64::MAX) {
            return u64::MAX;
        }
        if dims.is_empty() {
            return if self.is_scalar() { 1 } else { 0 };
        }
        dims.iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .unwrap_or(u64::MAX)
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

    /// Internal simple-dataspace predicate alias.
    pub fn is_simple_internal(&self) -> bool {
        self.is_simple()
    }

    /// Return the local dataspace category used before MPI-specific I/O.
    pub fn mpio_space_type(&self) -> DataspaceType {
        self.msg.space_type
    }

    /// Return the dataspace category used when obtaining transfer datatype
    /// context in the C selection I/O path.
    pub fn obtain_datatype(&self) -> DataspaceType {
        self.msg.space_type
    }

    /// Debug representation for dataspace diagnostics.
    pub fn debug(&self) -> String {
        format!("{:?}", self.msg)
    }

    /// Whether any dimension is resizable (has unlimited max dim).
    pub fn is_resizable(&self) -> bool {
        if let Some(maxdims) = &self.msg.max_dims {
            maxdims.iter().any(|&d| d == u64::MAX)
        } else {
            false
        }
    }

    /// Compare two dataspaces' extents.
    pub fn extent_equal(&self, other: &Self) -> bool {
        self.msg.space_type == other.msg.space_type
            && self.msg.dims == other.msg.dims
            && self.msg.max_dims == other.msg.max_dims
    }

    /// Internal extent equality helper.
    pub fn extent_equal_internal(&self, other: &Self) -> bool {
        self.extent_equal(other)
    }

    /// Set the serialized dataspace message version used for writer-side
    /// metadata.
    pub fn set_version(&mut self, version: u8) -> crate::Result<()> {
        if !matches!(version, 1 | 2) {
            return Err(crate::Error::InvalidFormat(format!(
                "dataspace message version {version}"
            )));
        }
        self.msg.version = version;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataspace_package_aliases_roundtrip() {
        assert!(Dataspace::init());
        assert!(Dataspace::init_package());
        Dataspace::top_term_package();
        Dataspace::term_package();

        let space = Dataspace::simple(vec![2, 3], None).unwrap();
        assert!(space.is_simple_internal());
        assert_eq!(space.mpio_space_type(), DataspaceType::Simple);
        assert_eq!(space.obtain_datatype(), DataspaceType::Simple);
        assert!(space.debug().contains("Simple"));
        assert_eq!(space.offset_simple(&[1, -1]).unwrap(), vec![1, -1]);
        assert!(space.offset_simple(&[0]).is_err());
        space.close_cb();
    }
}
