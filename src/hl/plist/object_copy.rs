/// Object-copy properties.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ObjectCopy {
    copy_object: u32,
    mcdt_search_cb: bool,
}

impl ObjectCopy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Object-copy option flags.
    pub fn copy_object(&self) -> u32 {
        self.copy_object
    }

    /// Set object-copy option flags.
    pub fn set_copy_object(&mut self, flags: u32) {
        self.copy_object = flags;
    }

    /// Committed-datatype search callback.
    pub fn mcdt_search_cb(&self) -> Option<()> {
        self.mcdt_search_cb.then_some(())
    }

    /// Set committed-datatype search callback presence.
    pub fn set_mcdt_search_cb(&mut self, installed: bool) {
        self.mcdt_search_cb = installed;
    }
}
