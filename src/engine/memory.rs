use crate::error::{Error, Result};

pub fn memcpy(dst: &mut [u8], src: &[u8]) -> Result<()> {
    if dst.len() < src.len() {
        return Err(Error::InvalidFormat("memcpy destination too small".into()));
    }
    dst[..src.len()].copy_from_slice(src);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5MM_memcpy(dst: &mut [u8], src: &[u8]) -> Result<()> {
    memcpy(dst, src)
}

pub fn realloc(mut buf: Vec<u8>, new_size: usize) -> Vec<u8> {
    buf.resize(new_size, 0);
    buf
}

pub fn xstrdup(value: &str) -> String {
    value.to_string()
}

pub fn strdup(value: &str) -> String {
    value.to_string()
}

pub fn strndup(value: &str, max_len: usize) -> String {
    value.chars().take(max_len).collect()
}

pub fn xfree_const<T>(_value: T) {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedBuffer {
    actual: Vec<u8>,
}

impl WrappedBuffer {
    pub fn wrap(data: impl Into<Vec<u8>>) -> Self {
        Self {
            actual: data.into(),
        }
    }

    pub fn actual(&self) -> &[u8] {
        &self.actual
    }

    pub fn actual_clear(&mut self) {
        self.actual.clear();
    }

    pub fn unwrap(self) -> Vec<u8> {
        self.actual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapped_buffer_roundtrips() {
        let mut wb = WrappedBuffer::wrap(b"abc".to_vec());
        assert_eq!(wb.actual(), b"abc");
        wb.actual_clear();
        assert!(wb.actual().is_empty());
    }
}
