use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FreeListStats {
    pub regular_bytes: usize,
    pub block_bytes: usize,
    pub array_bytes: usize,
    pub factory_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeListManager {
    regular: Vec<Vec<u8>>,
    blocks: Vec<Vec<u8>>,
    arrays: Vec<Vec<u8>>,
    factories: Vec<Vec<u8>>,
    max_regular: usize,
    max_block: usize,
    max_array: usize,
    max_factory: usize,
}

impl Default for FreeListManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FreeListManager {
    pub fn new() -> Self {
        Self {
            regular: Vec::new(),
            blocks: Vec::new(),
            arrays: Vec::new(),
            factories: Vec::new(),
            max_regular: usize::MAX,
            max_block: usize::MAX,
            max_array: usize::MAX,
            max_factory: usize::MAX,
        }
    }

    pub fn term_package(&mut self) {
        self.garbage_coll();
    }

    pub fn malloc(size: usize) -> Result<Vec<u8>> {
        if size == 0 {
            return Err(Error::InvalidFormat(
                "free-list allocation size is zero".into(),
            ));
        }
        Ok(vec![0; size])
    }

    pub fn reg_malloc(&mut self, size: usize) -> Result<Vec<u8>> {
        self.regular
            .pop()
            .filter(|buf| buf.len() >= size)
            .map_or_else(
                || Self::malloc(size),
                |mut buf| {
                    buf.resize(size, 0);
                    Ok(buf)
                },
            )
    }

    pub fn reg_calloc(&mut self, size: usize) -> Result<Vec<u8>> {
        let mut buf = self.reg_malloc(size)?;
        buf.fill(0);
        Ok(buf)
    }

    pub fn reg_gc_list(&mut self) {
        self.regular.clear();
    }

    pub fn reg_gc(&mut self) {
        self.reg_gc_list();
    }

    pub fn reg_term(&mut self) {
        self.reg_gc();
    }

    pub fn blk_find_list(&mut self, size: usize) -> Option<Vec<u8>> {
        let pos = self.blocks.iter().position(|buf| buf.len() >= size)?;
        Some(self.blocks.remove(pos))
    }

    pub fn blk_create_list(&mut self) {
        self.blocks.clear();
    }

    pub fn blk_init(&mut self) {
        self.blk_create_list();
    }

    pub fn blk_free_block_avail(&self, size: usize) -> bool {
        self.blocks.iter().any(|buf| buf.len() >= size)
    }

    pub fn blk_malloc(&mut self, size: usize) -> Result<Vec<u8>> {
        self.blk_find_list(size).map_or_else(
            || Self::malloc(size),
            |mut buf| {
                buf.resize(size, 0);
                Ok(buf)
            },
        )
    }

    pub fn blk_calloc(&mut self, size: usize) -> Result<Vec<u8>> {
        let mut buf = self.blk_malloc(size)?;
        buf.fill(0);
        Ok(buf)
    }

    pub fn blk_free(&mut self, mut buf: Vec<u8>) {
        if self.blocks.len() < self.max_block {
            buf.fill(0);
            self.blocks.push(buf);
        }
    }

    pub fn blk_realloc(&mut self, mut buf: Vec<u8>, new_size: usize) -> Result<Vec<u8>> {
        if new_size == 0 {
            return Err(Error::InvalidFormat(
                "free-list allocation size is zero".into(),
            ));
        }
        buf.resize(new_size, 0);
        Ok(buf)
    }

    pub fn blk_gc_list(&mut self) {
        self.blocks.clear();
    }

    pub fn blk_gc(&mut self) {
        self.blk_gc_list();
    }

    pub fn blk_term(&mut self) {
        self.blk_gc();
    }

    pub fn arr_init(&mut self) {
        self.arrays.clear();
    }

    pub fn arr_free(&mut self, mut buf: Vec<u8>) {
        if self.arrays.len() < self.max_array {
            buf.fill(0);
            self.arrays.push(buf);
        }
    }

    pub fn arr_malloc(&mut self, count: usize, elem_size: usize) -> Result<Vec<u8>> {
        let size = count
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("free-list array size overflow".into()))?;
        Self::malloc(size)
    }

    pub fn arr_calloc(&mut self, count: usize, elem_size: usize) -> Result<Vec<u8>> {
        self.arr_malloc(count, elem_size)
    }

    pub fn arr_realloc(
        &mut self,
        mut buf: Vec<u8>,
        count: usize,
        elem_size: usize,
    ) -> Result<Vec<u8>> {
        let size = count
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("free-list array size overflow".into()))?;
        buf.resize(size, 0);
        Ok(buf)
    }

    pub fn arr_gc_list(&mut self) {
        self.arrays.clear();
    }

    pub fn arr_gc(&mut self) {
        self.arr_gc_list();
    }

    pub fn arr_term(&mut self) {
        self.arr_gc();
    }

    pub fn seq_free(&mut self, buf: Vec<u8>) {
        self.arr_free(buf);
    }

    pub fn seq_malloc(&mut self, count: usize, elem_size: usize) -> Result<Vec<u8>> {
        self.arr_malloc(count, elem_size)
    }

    pub fn seq_calloc(&mut self, count: usize, elem_size: usize) -> Result<Vec<u8>> {
        self.arr_calloc(count, elem_size)
    }

    pub fn seq_realloc(&mut self, buf: Vec<u8>, count: usize, elem_size: usize) -> Result<Vec<u8>> {
        self.arr_realloc(buf, count, elem_size)
    }

    pub fn fac_init(&mut self) {
        self.factories.clear();
    }

    pub fn fac_free(&mut self, mut buf: Vec<u8>) {
        if self.factories.len() < self.max_factory {
            buf.fill(0);
            self.factories.push(buf);
        }
    }

    pub fn fac_malloc(&mut self, size: usize) -> Result<Vec<u8>> {
        self.factories
            .pop()
            .filter(|buf| buf.len() >= size)
            .map_or_else(
                || Self::malloc(size),
                |mut buf| {
                    buf.resize(size, 0);
                    Ok(buf)
                },
            )
    }

    pub fn fac_calloc(&mut self, size: usize) -> Result<Vec<u8>> {
        let mut buf = self.fac_malloc(size)?;
        buf.fill(0);
        Ok(buf)
    }

    pub fn fac_gc_list(&mut self) {
        self.factories.clear();
    }

    pub fn fac_gc(&mut self) {
        self.fac_gc_list();
    }

    pub fn fac_term(&mut self) {
        self.fac_gc();
    }

    pub fn fac_term_all(&mut self) {
        self.fac_term();
    }

    pub fn garbage_coll(&mut self) {
        self.reg_gc();
        self.blk_gc();
        self.arr_gc();
        self.fac_gc();
    }

    pub fn set_free_list_limits(
        &mut self,
        regular: usize,
        block: usize,
        array: usize,
        factory: usize,
    ) {
        self.max_regular = regular;
        self.max_block = block;
        self.max_array = array;
        self.max_factory = factory;
        self.regular.truncate(regular);
        self.blocks.truncate(block);
        self.arrays.truncate(array);
        self.factories.truncate(factory);
    }

    pub fn get_free_list_sizes(&self) -> FreeListStats {
        FreeListStats {
            regular_bytes: self.regular.iter().map(Vec::len).sum(),
            block_bytes: self.blocks.iter().map(Vec::len).sum(),
            array_bytes: self.arrays.iter().map(Vec::len).sum(),
            factory_bytes: self.factories.iter().map(Vec::len).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_free_list_reuses_buffers() {
        let mut lists = FreeListManager::new();
        let buf = lists.blk_malloc(8).unwrap();
        lists.blk_free(buf);
        assert!(lists.blk_free_block_avail(8));
        let reused = lists.blk_malloc(4).unwrap();
        assert_eq!(reused.len(), 4);
    }
}
