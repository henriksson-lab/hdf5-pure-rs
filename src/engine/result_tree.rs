/// Inclusive integer range result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultRange {
    pub start: u64,
    pub end: u64,
}

/// Ordered range result tree analogue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResultTree {
    ranges: Vec<ResultRange>,
}

impl ResultRange {
    pub fn intersects(self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

impl ResultTree {
    /// Initialize leaf storage.
    pub fn leaf_init(range: ResultRange) -> ResultRange {
        range
    }

    /// Add a range to a result set.
    pub fn result_set_add(&mut self, range: ResultRange) {
        self.ranges.push(range);
        self.ranges.sort_by_key(|range| (range.start, range.end));
    }

    /// Bulk-load ranges.
    pub fn bulk_load(ranges: Vec<ResultRange>) -> Self {
        let mut tree = Self::create();
        for range in ranges {
            tree.result_set_add(range);
        }
        tree
    }

    /// Create an empty result tree.
    pub fn create() -> Self {
        Self::default()
    }

    /// Recursive search helper.
    pub fn search_recurse(&self, query: ResultRange) -> Vec<ResultRange> {
        self.ranges
            .iter()
            .copied()
            .filter(|range| range.intersects(query))
            .collect()
    }

    /// Search for ranges intersecting `query`.
    pub fn search(&self, query: ResultRange) -> Vec<ResultRange> {
        self.search_recurse(query)
    }

    /// Free search results.
    pub fn free_results(_results: Vec<ResultRange>) {}

    /// Copy a node/range.
    pub fn node_copy(range: ResultRange) -> ResultRange {
        range
    }

    /// Recursive free hook.
    pub fn free_recurse(&mut self) {
        self.ranges.clear();
    }

    /// Free this tree.
    pub fn free(mut self) {
        self.free_recurse();
    }

    /// Copy this tree.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Return whether two leaves intersect.
    pub fn leaves_intersect(lhs: ResultRange, rhs: ResultRange) -> bool {
        lhs.intersects(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::{ResultRange, ResultTree};

    #[test]
    fn result_tree_aliases_find_intersections() {
        let a = ResultTree::leaf_init(ResultRange { start: 0, end: 3 });
        let b = ResultRange { start: 10, end: 12 };
        assert!(ResultTree::leaves_intersect(
            a,
            ResultRange { start: 2, end: 5 }
        ));
        assert!(!ResultTree::leaves_intersect(a, b));

        let mut tree = ResultTree::create();
        tree.result_set_add(a);
        tree.result_set_add(b);
        assert_eq!(tree.search(ResultRange { start: 1, end: 1 }), vec![a]);
        assert_eq!(
            tree.search_recurse(ResultRange { start: 11, end: 11 }),
            vec![b]
        );
        let copy = tree.copy();
        assert_eq!(copy, tree);
        ResultTree::free_results(copy.search(ResultRange { start: 0, end: 20 }));
        ResultTree::bulk_load(vec![a, b]).free();
        assert_eq!(ResultTree::node_copy(a), a);
    }
}
