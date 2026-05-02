use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileOpenTracker {
    opened: BTreeMap<u64, usize>,
    marked: BTreeSet<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TopOpenTracker {
    count: usize,
}

impl FileOpenTracker {
    pub fn create() -> Self {
        Self::default()
    }

    pub fn opened(&self, addr: u64) -> bool {
        self.opened.contains_key(&addr)
    }

    pub fn insert(&mut self, addr: u64) {
        *self.opened.entry(addr).or_insert(0) += 1;
    }

    pub fn delete(&mut self, addr: u64) -> Result<()> {
        let count = self
            .opened
            .get_mut(&addr)
            .ok_or_else(|| Error::InvalidFormat(format!("open object {addr:#x} not tracked")))?;
        *count -= 1;
        if *count == 0 {
            self.opened.remove(&addr);
            self.marked.remove(&addr);
        }
        Ok(())
    }

    pub fn mark(&mut self, addr: u64) {
        self.marked.insert(addr);
    }

    pub fn marked(&self, addr: u64) -> bool {
        self.marked.contains(&addr)
    }

    pub fn dest(self) {}
}

impl TopOpenTracker {
    pub fn top_create() -> Self {
        Self::default()
    }

    pub fn top_incr(&mut self) {
        self.count = self.count.saturating_add(1);
    }

    pub fn top_decr(&mut self) -> Result<()> {
        if self.count == 0 {
            return Err(Error::InvalidFormat(
                "top open-object counter underflow".into(),
            ));
        }
        self.count -= 1;
        Ok(())
    }

    pub fn top_count(&self) -> usize {
        self.count
    }

    pub fn top_dest(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_tracker_counts_and_marks() {
        let mut tracker = FileOpenTracker::create();
        tracker.insert(10);
        tracker.mark(10);
        assert!(tracker.opened(10));
        assert!(tracker.marked(10));
        tracker.delete(10).unwrap();
        assert!(!tracker.opened(10));
    }
}
