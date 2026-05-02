use std::path::{Path, PathBuf};

/// View policy for virtual datasets with unlimited dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdsView {
    LastAvailable,
    FirstMissing,
}

/// Policy for virtual dataset mappings whose source files are absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdsMissingSourcePolicy {
    Error,
    Fill,
}

/// Dataset access properties used by high-level reads.
///
/// This currently models the VDS access properties that affect pure-Rust
/// dataset reads: virtual view policy and explicit virtual source prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetAccess {
    pub(super) virtual_view: VdsView,
    pub(super) virtual_prefix: Option<PathBuf>,
    pub(super) virtual_missing_source_policy: VdsMissingSourcePolicy,
    append_flush: bool,
    efile_prefix: Option<PathBuf>,
}

impl Default for DatasetAccess {
    fn default() -> Self {
        Self {
            virtual_view: VdsView::LastAvailable,
            virtual_prefix: None,
            virtual_missing_source_policy: VdsMissingSourcePolicy::Error,
            append_flush: false,
            efile_prefix: None,
        }
    }
}

impl DatasetAccess {
    /// Create default dataset access properties.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the VDS view policy.
    pub fn with_virtual_view(mut self, view: VdsView) -> Self {
        self.virtual_view = view;
        self
    }

    /// Set the VDS view policy in place.
    pub fn set_virtual_view(&mut self, view: VdsView) {
        self.virtual_view = view;
    }

    /// Set an explicit VDS source-file prefix.
    ///
    /// `${ORIGIN}` is expanded to the virtual dataset file's containing
    /// directory, matching HDF5's virtual-prefix convention.
    pub fn with_virtual_prefix<P: Into<PathBuf>>(mut self, prefix: P) -> Self {
        self.virtual_prefix = Some(prefix.into());
        self
    }

    /// Set an explicit VDS source-file prefix in place.
    pub fn set_virtual_prefix<P: Into<PathBuf>>(&mut self, prefix: Option<P>) {
        self.virtual_prefix = prefix.map(Into::into);
    }

    /// Set the behavior for absent VDS source files.
    pub fn with_virtual_missing_source_policy(mut self, policy: VdsMissingSourcePolicy) -> Self {
        self.virtual_missing_source_policy = policy;
        self
    }

    pub fn virtual_view(&self) -> VdsView {
        self.virtual_view
    }

    pub fn virtual_prefix(&self) -> Option<&Path> {
        self.virtual_prefix.as_deref()
    }

    pub fn virtual_missing_source_policy(&self) -> VdsMissingSourcePolicy {
        self.virtual_missing_source_policy
    }

    /// Whether append-flush callback state is installed.
    pub fn append_flush(&self) -> bool {
        self.append_flush
    }

    /// Set append-flush callback presence.
    pub fn set_append_flush(&mut self, installed: bool) {
        self.append_flush = installed;
    }

    /// External raw-storage file prefix.
    pub fn efile_prefix(&self) -> Option<&Path> {
        self.efile_prefix.as_deref()
    }

    /// Set external raw-storage file prefix.
    pub fn set_efile_prefix<P: Into<PathBuf>>(&mut self, prefix: Option<P>) {
        self.efile_prefix = prefix.map(Into::into);
    }
}
