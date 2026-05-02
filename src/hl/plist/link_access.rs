use crate::hl::plist::file_access::FileAccess;

/// Link access properties used for path traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkAccess {
    max_link_traversals: usize,
    external_link_prefix: Option<String>,
    external_link_access: FileAccess,
    external_link_acc_flags: u32,
    external_link_callback: bool,
    external_link_file_cache_size: usize,
}

impl Default for LinkAccess {
    fn default() -> Self {
        Self {
            max_link_traversals: 40,
            external_link_prefix: None,
            external_link_access: FileAccess::default(),
            external_link_acc_flags: 0,
            external_link_callback: false,
            external_link_file_cache_size: 0,
        }
    }
}

impl LinkAccess {
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum number of soft/external link traversals.
    pub fn nlinks(&self) -> usize {
        self.max_link_traversals
    }

    /// Set maximum number of soft/external link traversals.
    pub fn set_nlinks(&mut self, nlinks: usize) {
        self.max_link_traversals = nlinks;
    }

    /// Prefix applied to external-link filenames.
    pub fn elink_prefix(&self) -> Option<&str> {
        self.external_link_prefix.as_deref()
    }

    /// Set prefix applied to external-link filenames.
    pub fn set_elink_prefix<S: Into<String>>(&mut self, prefix: Option<S>) {
        self.external_link_prefix = prefix.map(Into::into);
    }

    /// File access properties used when opening external-link target files.
    pub fn elink_fapl(&self) -> &FileAccess {
        &self.external_link_access
    }

    /// Set file access properties used for external-link target files.
    pub fn set_elink_fapl(&mut self, fapl: FileAccess) {
        self.external_link_access = fapl;
    }

    /// File access flags used when opening external-link target files.
    pub fn elink_acc_flags(&self) -> u32 {
        self.external_link_acc_flags
    }

    /// Set file access flags used for external-link target files.
    pub fn set_elink_acc_flags(&mut self, flags: u32) {
        self.external_link_acc_flags = flags;
    }

    /// External-link traversal callback. This reader does not install one.
    pub fn elink_cb(&self) -> Option<()> {
        self.external_link_callback.then_some(())
    }

    /// Set external-link traversal callback presence.
    pub fn set_elink_cb(&mut self, installed: bool) {
        self.external_link_callback = installed;
    }

    /// External-link file cache size. The pure Rust resolver opens targets
    /// directly and does not maintain a cache.
    pub fn elink_file_cache_size(&self) -> usize {
        self.external_link_file_cache_size
    }

    /// Set external-link file cache size.
    pub fn set_elink_file_cache_size(&mut self, size: usize) {
        self.external_link_file_cache_size = size;
    }
}
