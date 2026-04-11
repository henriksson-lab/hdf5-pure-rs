/// Describes a field in a compound HDF5 type.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub type_class: TypeClass,
}

/// Simple type class for describing H5Type fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeClass {
    Integer { signed: bool },
    Float,
    Compound,
}

/// Trait for types that can be stored in HDF5 datasets/attributes.
///
/// # Safety
/// Implementors must ensure that `type_size()` returns the exact size of the
/// in-memory representation and that the type is safe to transmute from raw bytes.
pub unsafe trait H5Type: Copy + 'static {
    /// Size of one element in bytes.
    fn type_size() -> usize;

    /// For compound types, return the field descriptors.
    /// Returns None for primitive types.
    fn compound_fields() -> Option<Vec<FieldDescriptor>> {
        None
    }

    /// For enum types, return the (name, value) mapping.
    fn enum_members() -> Option<Vec<(String, i64)>> {
        None
    }
}

macro_rules! impl_h5type {
    ($($t:ty),*) => {
        $(
            unsafe impl H5Type for $t {
                fn type_size() -> usize { std::mem::size_of::<$t>() }
            }
        )*
    };
}

impl_h5type!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

/// Reinterpret a byte slice as a slice of `T`, copying to ensure alignment.
pub fn bytes_to_slice<T: H5Type>(bytes: &[u8]) -> crate::Result<&[T]> {
    // For zero-copy, we need alignment to be correct.
    // Since Vec<u8> may not be aligned for T, we use bytes_to_vec instead for safety.
    // This function exists for small reads (attributes) where the data is borrowed.
    let elem_size = T::type_size();
    if elem_size == 0 {
        return Err(crate::Error::Other("zero-size type".into()));
    }
    if bytes.len() % elem_size != 0 {
        return Err(crate::Error::InvalidFormat(format!(
            "byte count {} is not a multiple of element size {}",
            bytes.len(),
            elem_size
        )));
    }
    let align = std::mem::align_of::<T>();
    if (bytes.as_ptr() as usize) % align != 0 {
        // Misaligned -- caller should use bytes_to_vec instead
        return Err(crate::Error::Other(
            "buffer alignment insufficient for type; use read() instead of read_scalar()".into(),
        ));
    }
    let count = bytes.len() / elem_size;
    let ptr = bytes.as_ptr() as *const T;
    // SAFETY: We verified alignment and size.
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
}

/// Reinterpret a byte vec as a vec of `T`.
/// Copies data to a properly aligned buffer if needed.
pub fn bytes_to_vec<T: H5Type>(bytes: Vec<u8>) -> crate::Result<Vec<T>> {
    let elem_size = T::type_size();
    if elem_size == 0 {
        return Err(crate::Error::Other("zero-size type".into()));
    }
    if bytes.len() % elem_size != 0 {
        return Err(crate::Error::InvalidFormat(format!(
            "byte count {} is not a multiple of element size {}",
            bytes.len(),
            elem_size
        )));
    }
    let count = bytes.len() / elem_size;
    let align = std::mem::align_of::<T>();

    if (bytes.as_ptr() as usize) % align == 0 {
        // Already aligned -- zero-copy reinterpret
        let mut bytes = std::mem::ManuallyDrop::new(bytes);
        let ptr = bytes.as_mut_ptr() as *mut T;
        let cap = bytes.capacity() / elem_size;
        // SAFETY: Alignment verified, size checked, T: Copy.
        Ok(unsafe { Vec::from_raw_parts(ptr, count, cap) })
    } else {
        // Misaligned -- copy to aligned buffer
        let mut result = Vec::<T>::with_capacity(count);
        // SAFETY: T: Copy, and we're copying byte-by-byte into aligned storage.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
            result.set_len(count);
        }
        Ok(result)
    }
}
