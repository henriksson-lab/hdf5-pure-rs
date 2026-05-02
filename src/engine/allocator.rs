/// Simple bump allocator for file space.
/// Allocates space sequentially from the end of the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileAllocator {
    /// Current end-of-file position.
    next_offset: u64,
}

impl FileAllocator {
    /// Create a new allocator starting at the given offset.
    pub fn new(start_offset: u64) -> Self {
        Self {
            next_offset: start_offset,
        }
    }

    /// Allocate `size` bytes, optionally aligned to `align` bytes.
    /// Returns the starting offset of the allocated block.
    pub fn allocate(&mut self, size: u64, align: u64) -> u64 {
        if align > 1 {
            let remainder = self.next_offset % align;
            if remainder != 0 {
                self.next_offset += align - remainder;
            }
        }

        let offset = self.next_offset;
        self.next_offset += size;
        offset
    }

    /// Get the current end-of-file position.
    pub fn eof(&self) -> u64 {
        self.next_offset
    }
}
