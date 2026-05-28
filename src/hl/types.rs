/// Describes a field in a compound HDF5 type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDescriptor {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub type_class: TypeClass,
}

/// Simple type class for describing H5Type fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeClass {
    Integer {
        signed: bool,
    },
    Float,
    Enum {
        signed: bool,
        size: usize,
        members: Vec<(String, u64)>,
    },
    Compound {
        size: usize,
        fields: Vec<FieldDescriptor>,
    },
}

/// Primitive Rust storage type for HDF5 dtype inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Integer { signed: bool, size: usize },
    Float { size: usize },
}

/// Trait for types that can be stored in HDF5 datasets/attributes.
///
/// # Safety
/// Implementors must ensure that `type_size()` returns the exact size of the
/// in-memory representation. `validate_bytes()` must reject byte patterns that
/// are not valid values of the implementing Rust type, and `has_padding()` must
/// return true when raw byte views of initialized values could read padding.
///
/// If implemented, metadata methods must describe the same in-memory bytes:
/// `primitive_type()` must match the Rust representation and signedness,
/// enum metadata must include the exact base type and raw encoded members, and
/// compound field descriptors must use valid initialized field byte ranges.
pub unsafe trait H5Type: Copy + 'static {
    /// Size of one element in bytes.
    fn type_size() -> usize;

    /// Whether values of this type contain padding bytes that are not safe to
    /// expose through raw typed writes.
    fn has_padding() -> bool {
        false
    }

    /// Whether raw bytes must be checked before they can be exposed as `Self`.
    fn requires_validation() -> bool {
        false
    }

    /// Return the primitive storage type, if `Self` maps directly to one.
    fn primitive_type() -> Option<PrimitiveType> {
        None
    }

    /// Validate one encoded element before constructing a Rust value from it.
    fn validate_bytes(bytes: &[u8]) -> crate::Result<()> {
        if bytes.len() == Self::type_size() {
            Ok(())
        } else {
            Err(crate::Error::InvalidFormat(format!(
                "byte count {} does not match element size {}",
                bytes.len(),
                Self::type_size()
            )))
        }
    }

    /// Validate a whole element slice.
    fn validate_byte_slice(bytes: &[u8]) -> crate::Result<()> {
        let elem_size = Self::type_size();
        if elem_size == 0 {
            return Err(crate::Error::Other("zero-size type".into()));
        }
        if !bytes.len().is_multiple_of(elem_size) {
            return Err(crate::Error::InvalidFormat(format!(
                "byte count {} is not a multiple of element size {}",
                bytes.len(),
                elem_size
            )));
        }
        for chunk in bytes.chunks_exact(elem_size) {
            Self::validate_bytes(chunk)?;
        }
        Ok(())
    }

    /// Visit compound field descriptors without returning a fresh Vec.
    fn visit_compound_fields<F>(_visitor: F) -> Option<()>
    where
        F: FnMut(FieldDescriptor),
    {
        None
    }

    /// Store compound field descriptors in caller-provided storage.
    fn compound_fields_into(out: &mut Vec<FieldDescriptor>) -> Option<()> {
        let mut next = Vec::new();
        let visited = Self::visit_compound_fields(|field| next.push(field));
        if visited.is_some() {
            *out = next;
        } else {
            out.clear();
        }
        visited
    }

    /// Return compound field descriptors in a fresh vector.
    fn compound_fields() -> Option<Vec<FieldDescriptor>> {
        let mut fields = Vec::new();
        Self::compound_fields_into(&mut fields)?;
        Some(fields)
    }

    /// Visit enum members without returning a fresh Vec.
    fn visit_enum_members<F>(_visitor: F) -> Option<()>
    where
        F: FnMut(&str, u64),
    {
        None
    }

    /// Store enum members in caller-provided storage.
    fn enum_members_into(out: &mut Vec<(String, u64)>) -> Option<()> {
        let mut next = Vec::new();
        let visited = Self::visit_enum_members(|name, value| next.push((name.to_string(), value)));
        if visited.is_some() {
            *out = next;
        } else {
            out.clear();
        }
        visited
    }

    /// Return enum members in a fresh vector.
    fn enum_members() -> Option<Vec<(String, u64)>> {
        let mut members = Vec::new();
        Self::enum_members_into(&mut members)?;
        Some(members)
    }

    /// Return the enum's integer base as `(signed, size_in_bytes)`.
    fn enum_base_type() -> Option<(bool, usize)> {
        None
    }

    /// Store enum members as raw encoded integer values.
    fn enum_members_u64_into(out: &mut Vec<(String, u64)>) -> Option<()> {
        out.clear();
        Self::visit_enum_members(|name, value| out.push((name.to_string(), value)))
    }
}

macro_rules! impl_h5type_integer {
    ($($t:ty => $signed:literal),*) => {
        $(
            unsafe impl H5Type for $t {
                fn type_size() -> usize { std::mem::size_of::<$t>() }
                fn primitive_type() -> Option<PrimitiveType> {
                    Some(PrimitiveType::Integer {
                        signed: $signed,
                        size: std::mem::size_of::<$t>(),
                    })
                }
            }
        )*
    };
}

macro_rules! impl_h5type_float {
    ($($t:ty),*) => {
        $(
            unsafe impl H5Type for $t {
                fn type_size() -> usize { std::mem::size_of::<$t>() }
                fn primitive_type() -> Option<PrimitiveType> {
                    Some(PrimitiveType::Float {
                        size: std::mem::size_of::<$t>(),
                    })
                }
            }
        )*
    };
}

impl_h5type_integer!(
    u8 => false,
    u16 => false,
    u32 => false,
    u64 => false,
    u128 => false,
    i8 => true,
    i16 => true,
    i32 => true,
    i64 => true,
    i128 => true
);
impl_h5type_float!(f32, f64);

/// Return the HDF5 type class for a Rust `H5Type` when it is known.
pub fn type_class_for<T: H5Type>() -> Option<TypeClass> {
    if let Some((signed, size)) = T::enum_base_type() {
        let mut members = Vec::new();
        T::enum_members_u64_into(&mut members)?;
        Some(TypeClass::Enum {
            signed,
            size,
            members,
        })
    } else if let Some(primitive) = T::primitive_type() {
        match primitive {
            PrimitiveType::Integer { signed, .. } => Some(TypeClass::Integer { signed }),
            PrimitiveType::Float { .. } => Some(TypeClass::Float),
        }
    } else {
        T::compound_fields().map(|fields| TypeClass::Compound {
            size: T::type_size(),
            fields,
        })
    }
}

/// Reinterpret a byte slice as a slice of `T`, copying to ensure alignment.
pub fn bytes_to_slice<T: H5Type>(bytes: &[u8]) -> crate::Result<&[T]> {
    // For zero-copy, we need alignment to be correct.
    // Since Vec<u8> may not be aligned for T, we use bytes_to_vec instead for safety.
    // This function exists for small reads (attributes) where the data is borrowed.
    let elem_size = T::type_size();
    if elem_size == 0 {
        return Err(crate::Error::Other("zero-size type".into()));
    }
    if !bytes.len().is_multiple_of(elem_size) {
        return Err(crate::Error::InvalidFormat(format!(
            "byte count {} is not a multiple of element size {}",
            bytes.len(),
            elem_size
        )));
    }
    let align = std::mem::align_of::<T>();
    if !(bytes.as_ptr() as usize).is_multiple_of(align) {
        // Misaligned -- caller should use bytes_to_vec instead
        return Err(crate::Error::Other(
            "buffer alignment insufficient for type; use read() instead of read_scalar()".into(),
        ));
    }
    T::validate_byte_slice(bytes)?;
    let count = bytes.len() / elem_size;
    let ptr = bytes.as_ptr() as *const T;
    // SAFETY: We verified alignment and size.
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
}

/// View a mutable typed slice as raw bytes for caller-buffer I/O.
pub fn slice_as_bytes_mut<T: H5Type>(values: &mut [T]) -> &mut [u8] {
    let len = values
        .len()
        .checked_mul(T::type_size())
        .expect("typed slice byte length overflow");
    let ptr = values.as_mut_ptr() as *mut u8;
    // SAFETY: `values` is a live mutable slice and `T: H5Type` promises a plain
    // byte-addressable representation with exactly `type_size()` bytes.
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// View a typed slice as raw bytes only when doing so cannot expose padding.
pub fn slice_as_bytes_checked<T: H5Type>(values: &[T]) -> crate::Result<&[u8]> {
    if T::has_padding() {
        return Err(crate::Error::Unsupported(
            "typed writes for H5Type values with padding are not supported".into(),
        ));
    }
    Ok(slice_as_bytes(values))
}

/// Return write bytes for typed values without exposing padding bytes.
pub fn values_as_bytes_checked<T: H5Type>(
    values: &[T],
) -> crate::Result<std::borrow::Cow<'_, [u8]>> {
    if !T::has_padding() {
        return Ok(std::borrow::Cow::Borrowed(slice_as_bytes(values)));
    }

    let elem_size = T::type_size();
    let total_len = values
        .len()
        .checked_mul(elem_size)
        .ok_or_else(|| crate::Error::InvalidFormat("typed slice byte length overflow".into()))?;
    let fields = T::compound_fields().ok_or_else(|| {
        crate::Error::Unsupported(
            "typed writes for padded non-compound H5Type values are not supported".into(),
        )
    })?;
    let mut out = vec![0u8; total_len];
    for (index, value) in values.iter().enumerate() {
        let src_base = value as *const T as *const u8;
        let dst_base = index * elem_size;
        copy_initialized_field_bytes(
            &mut out,
            dst_base,
            src_base,
            elem_size,
            &fields,
            "compound field",
        )?;
    }
    Ok(std::borrow::Cow::Owned(out))
}

fn copy_initialized_field_bytes(
    out: &mut [u8],
    dst_base: usize,
    src_base: *const u8,
    elem_size: usize,
    fields: &[FieldDescriptor],
    context: &str,
) -> crate::Result<()> {
    for field in fields {
        let field_end = field
            .offset
            .checked_add(field.size)
            .ok_or_else(|| crate::Error::InvalidFormat(format!("{context} range overflow")))?;
        if field_end > elem_size {
            return Err(crate::Error::InvalidFormat(format!(
                "{context} '{}' ends at byte {field_end}, beyond element size {elem_size}",
                field.name
            )));
        }

        match &field.type_class {
            TypeClass::Compound {
                size,
                fields: nested,
            } => {
                if field.size != *size {
                    return Err(crate::Error::InvalidFormat(format!(
                        "nested compound field '{}' size metadata mismatch: field size {}, type size {}",
                        field.name, field.size, size
                    )));
                }
                copy_initialized_field_bytes(
                    out,
                    dst_base + field.offset,
                    // SAFETY: the range check above proves `field.offset` is
                    // within this element. The recursive call validates nested
                    // ranges before reading bytes.
                    unsafe { src_base.add(field.offset) },
                    *size,
                    nested,
                    "nested compound field",
                )?;
            }
            _ => {
                // SAFETY: compound metadata promises that non-compound field
                // ranges point only to initialized field bytes. Padding bytes
                // in `out` remain zeroed.
                let src =
                    unsafe { std::slice::from_raw_parts(src_base.add(field.offset), field.size) };
                out[dst_base + field.offset..dst_base + field_end].copy_from_slice(src);
            }
        }
    }
    Ok(())
}

/// View a typed slice as raw bytes without allocating.
pub fn slice_as_bytes<T: H5Type>(values: &[T]) -> &[u8] {
    assert!(
        !T::has_padding(),
        "typed writes for H5Type values with padding are not supported"
    );
    let len = values
        .len()
        .checked_mul(T::type_size())
        .expect("typed slice byte length overflow");
    let ptr = values.as_ptr() as *const u8;
    // SAFETY: `values` is a live slice and `T: H5Type` promises a plain
    // byte-addressable representation with exactly `type_size()` bytes.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Reinterpret a byte vec as a vec of `T`.
/// Copies data to a properly aligned buffer if needed.
pub fn bytes_to_vec<T: H5Type>(bytes: Vec<u8>) -> crate::Result<Vec<T>> {
    let elem_size = T::type_size();
    let actual_size = std::mem::size_of::<T>();
    if elem_size == 0 {
        return Err(crate::Error::Other("zero-size type".into()));
    }
    if elem_size != actual_size {
        return Err(crate::Error::Other(format!(
            "H5Type size {} does not match Rust type size {}",
            elem_size, actual_size
        )));
    }
    if !bytes.len().is_multiple_of(elem_size) {
        return Err(crate::Error::InvalidFormat(format!(
            "byte count {} is not a multiple of element size {}",
            bytes.len(),
            elem_size
        )));
    }
    let count = bytes.len() / elem_size;
    let align = std::mem::align_of::<T>();

    if elem_size == 1 && align == 1 {
        T::validate_byte_slice(&bytes)?;
        // `Vec::from_raw_parts` must deallocate with the same allocation
        // layout that created the buffer. A `Vec<u8>` allocation is layout-
        // compatible only with one-byte, one-alignment element types.
        let mut bytes = std::mem::ManuallyDrop::new(bytes);
        let ptr = bytes.as_mut_ptr() as *mut T;
        let cap = bytes.capacity();
        // SAFETY: size and alignment are both one, so the allocation layout is
        // unchanged. `T: H5Type` promises every copied byte pattern is valid.
        Ok(unsafe { Vec::from_raw_parts(ptr, count, cap) })
    } else {
        T::validate_byte_slice(&bytes)?;
        // Copy into storage allocated with `T`'s layout.
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
