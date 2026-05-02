/// Object creation properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectCreate {
    track_times: bool,
    attr_phase_change: (u32, u32),
    attr_creation_order: u32,
    local_heap_size_hint: usize,
    link_phase_change: (u32, u32),
    est_link_info: (u32, u32),
}

impl Default for ObjectCreate {
    fn default() -> Self {
        Self {
            track_times: true,
            attr_phase_change: (8, 6),
            attr_creation_order: 0,
            local_heap_size_hint: 0,
            link_phase_change: (8, 6),
            est_link_info: (4, 8),
        }
    }
}

impl ObjectCreate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether object timestamps are tracked.
    pub fn obj_track_times(&self) -> bool {
        self.track_times
    }

    /// Set whether object timestamps are tracked.
    pub fn set_obj_track_times(&mut self, track_times: bool) {
        self.track_times = track_times;
    }

    /// Attribute compact/dense phase-change thresholds.
    pub fn attr_phase_change(&self) -> (u32, u32) {
        self.attr_phase_change
    }

    /// Set attribute compact/dense phase-change thresholds.
    pub fn set_attr_phase_change(&mut self, max_compact: u32, min_dense: u32) {
        self.attr_phase_change = (max_compact, min_dense);
    }

    /// Attribute creation-order tracking/indexing flags.
    pub fn attr_creation_order(&self) -> u32 {
        self.attr_creation_order
    }

    /// Set attribute creation-order tracking/indexing flags.
    pub fn set_attr_creation_order(&mut self, flags: u32) {
        self.attr_creation_order = flags;
    }

    /// Local heap size hint for old-style groups.
    pub fn local_heap_size_hint(&self) -> usize {
        self.local_heap_size_hint
    }

    /// Set local heap size hint for old-style groups.
    pub fn set_local_heap_size_hint(&mut self, size_hint: usize) {
        self.local_heap_size_hint = size_hint;
    }

    /// Link compact/dense phase-change thresholds.
    pub fn link_phase_change(&self) -> (u32, u32) {
        self.link_phase_change
    }

    /// Set link compact/dense phase-change thresholds.
    pub fn set_link_phase_change(&mut self, max_compact: u32, min_dense: u32) {
        self.link_phase_change = (max_compact, min_dense);
    }

    /// Estimated link count and average name length.
    pub fn est_link_info(&self) -> (u32, u32) {
        self.est_link_info
    }

    /// Set estimated link count and average name length.
    pub fn set_est_link_info(&mut self, link_count: u32, name_length: u32) {
        self.est_link_info = (link_count, name_length);
    }
}
