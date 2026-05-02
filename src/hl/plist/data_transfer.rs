/// Dataset transfer properties used by high-level reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTransfer {
    buffer_size: usize,
    preserve: bool,
    data_transform: Option<String>,
    edc_check: bool,
    filter_callback: bool,
    type_conv_cb: bool,
    vlen_mem_manager: bool,
    btree_ratios: (u64, u64, u64),
    hyper_vector_size: usize,
    modify_write_buf: bool,
}

impl Default for DataTransfer {
    fn default() -> Self {
        Self {
            buffer_size: 0,
            preserve: false,
            data_transform: None,
            edc_check: true,
            filter_callback: false,
            type_conv_cb: false,
            vlen_mem_manager: false,
            btree_ratios: (0.1f64.to_bits(), 0.5f64.to_bits(), 0.9f64.to_bits()),
            hyper_vector_size: 1024,
            modify_write_buf: false,
        }
    }
}

impl DataTransfer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Type-conversion/background buffer configuration.
    pub fn buffer(&self) -> (usize, Option<()>, Option<()>) {
        (self.buffer_size, None, None)
    }

    /// Set type-conversion/background buffer size.
    pub fn set_buffer(&mut self, size: usize) {
        self.buffer_size = size;
    }

    /// Whether compound-field preservation is enabled during conversion.
    pub fn preserve(&self) -> bool {
        self.preserve
    }

    /// Set compound-field preservation during conversion.
    pub fn set_preserve(&mut self, preserve: bool) {
        self.preserve = preserve;
    }

    /// Type-conversion exception callback.
    pub fn type_conv_cb(&self) -> Option<()> {
        self.type_conv_cb.then_some(())
    }

    /// Set type-conversion exception callback presence.
    pub fn set_type_conv_cb(&mut self, installed: bool) {
        self.type_conv_cb = installed;
    }

    /// Variable-length memory manager callbacks.
    pub fn vlen_mem_manager(&self) -> (Option<()>, Option<()>) {
        if self.vlen_mem_manager {
            (Some(()), Some(()))
        } else {
            (None, None)
        }
    }

    /// Set variable-length memory manager callback presence.
    pub fn set_vlen_mem_manager(&mut self, installed: bool) {
        self.vlen_mem_manager = installed;
    }

    /// Data transform expression.
    pub fn data_transform(&self) -> Option<&str> {
        self.data_transform.as_deref()
    }

    /// Set data transform expression.
    pub fn set_data_transform<S: Into<String>>(&mut self, transform: Option<S>) {
        self.data_transform = transform.map(Into::into);
    }

    /// Whether error-detection checks are enabled.
    pub fn edc_check(&self) -> bool {
        self.edc_check
    }

    /// Set error-detection check policy.
    pub fn set_edc_check(&mut self, enabled: bool) {
        self.edc_check = enabled;
    }

    /// Filter callback presence.
    pub fn filter_callback(&self) -> Option<()> {
        self.filter_callback.then_some(())
    }

    /// Set filter callback presence.
    pub fn set_filter_callback(&mut self, installed: bool) {
        self.filter_callback = installed;
    }

    /// B-tree split ratios `(left, middle, right)`.
    pub fn btree_ratios(&self) -> (f64, f64, f64) {
        (
            f64::from_bits(self.btree_ratios.0),
            f64::from_bits(self.btree_ratios.1),
            f64::from_bits(self.btree_ratios.2),
        )
    }

    /// Set B-tree split ratios.
    pub fn set_btree_ratios(&mut self, left: f64, middle: f64, right: f64) {
        self.btree_ratios = (left.to_bits(), middle.to_bits(), right.to_bits());
    }

    /// Hyperslab vector size hint.
    pub fn hyper_vector_size(&self) -> usize {
        self.hyper_vector_size
    }

    /// Set hyperslab vector size hint.
    pub fn set_hyper_vector_size(&mut self, size: usize) {
        self.hyper_vector_size = size;
    }

    /// Whether write buffers may be modified.
    pub fn modify_write_buf(&self) -> bool {
        self.modify_write_buf
    }

    /// Set whether write buffers may be modified.
    pub fn set_modify_write_buf(&mut self, enabled: bool) {
        self.modify_write_buf = enabled;
    }

    /// Actual MPI chunk optimization mode. MPI is intentionally not used.
    pub fn mpio_actual_chunk_opt_mode(&self) -> Option<()> {
        None
    }

    /// Actual MPI I/O mode. MPI is intentionally not used.
    pub fn mpio_actual_io_mode(&self) -> Option<()> {
        None
    }

    /// Reasons collective MPI I/O was not used.
    pub fn mpio_no_collective_cause(&self) -> (u32, u32) {
        (0, 0)
    }
}
