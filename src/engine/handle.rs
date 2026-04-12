use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

/// The type used for internal object identifiers, analogous to HDF5's `hid_t`.
pub type Hid = i64;

/// Invalid handle ID sentinel.
pub const INVALID_HID: Hid = -1;

/// Types of internal objects, analogous to HDF5's H5I_type_t.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandleType {
    File,
    Group,
    Dataset,
    Attribute,
    Datatype,
    Dataspace,
    PropertyList,
}

/// Metadata stored for each registered handle.
struct HandleEntry {
    handle_type: HandleType,
    refcount: i32,
    /// Opaque data associated with this handle.
    data: Arc<dyn std::any::Any + Send + Sync>,
}

/// Global registry of internal handles, replacing HDF5's C-level hid_t system.
pub struct HandleRegistry {
    next_id: AtomicI64,
    entries: RwLock<HashMap<Hid, HandleEntry>>,
}

impl HandleRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            next_id: AtomicI64::new(1), // Start at 1, 0 and negative are reserved
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new object and return its handle ID.
    pub fn register<T: Send + Sync + 'static>(
        &self,
        handle_type: HandleType,
        data: T,
    ) -> Hid {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = HandleEntry {
            handle_type,
            refcount: 1,
            data: Arc::new(data),
        };
        self.entries.write().insert(id, entry);
        id
    }

    /// Increment the reference count for a handle.
    pub fn incref(&self, id: Hid) -> Option<i32> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&id) {
            entry.refcount += 1;
            Some(entry.refcount)
        } else {
            None
        }
    }

    /// Decrement the reference count for a handle.
    /// Returns the new refcount, or None if the handle was invalid.
    /// When refcount reaches 0, the entry is removed.
    pub fn decref(&self, id: Hid) -> Option<i32> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&id) {
            entry.refcount -= 1;
            let rc = entry.refcount;
            if rc <= 0 {
                entries.remove(&id);
            }
            Some(rc)
        } else {
            None
        }
    }

    /// Get the current reference count.
    pub fn refcount(&self, id: Hid) -> Option<i32> {
        self.entries.read().get(&id).map(|e| e.refcount)
    }

    /// Get the handle type.
    pub fn handle_type(&self, id: Hid) -> Option<HandleType> {
        self.entries.read().get(&id).map(|e| e.handle_type)
    }

    /// Check if a handle is valid (exists and has refcount > 0).
    pub fn is_valid(&self, id: Hid) -> bool {
        self.entries
            .read()
            .get(&id)
            .map_or(false, |e| e.refcount > 0)
    }

    /// Get the data associated with a handle, downcasted to the expected type.
    pub fn get<T: Send + Sync + 'static>(&self, id: Hid) -> Option<Arc<T>> {
        let entries = self.entries.read();
        entries
            .get(&id)
            .and_then(|e| e.data.clone().downcast::<T>().ok())
    }

    /// Number of currently registered handles.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

impl Default for HandleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global handle registry instance.
static REGISTRY: parking_lot::Mutex<Option<HandleRegistry>> = parking_lot::Mutex::new(None);

/// Get or initialize the global handle registry.
pub fn global_registry() -> &'static parking_lot::Mutex<Option<HandleRegistry>> {
    // Ensure registry is initialized
    {
        let mut guard = REGISTRY.lock();
        if guard.is_none() {
            *guard = Some(HandleRegistry::new());
        }
    }
    &REGISTRY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let reg = HandleRegistry::new();
        let id = reg.register(HandleType::File, "test_data".to_string());

        assert!(reg.is_valid(id));
        assert_eq!(reg.handle_type(id), Some(HandleType::File));
        assert_eq!(reg.refcount(id), Some(1));

        let data = reg.get::<String>(id).unwrap();
        assert_eq!(&*data, "test_data");
    }

    #[test]
    fn test_refcount() {
        let reg = HandleRegistry::new();
        let id = reg.register(HandleType::Group, 42u32);

        assert_eq!(reg.incref(id), Some(2));
        assert_eq!(reg.incref(id), Some(3));
        assert_eq!(reg.decref(id), Some(2));
        assert_eq!(reg.decref(id), Some(1));
        assert!(reg.is_valid(id));

        // Drop to 0 -- entry removed
        assert_eq!(reg.decref(id), Some(0));
        assert!(!reg.is_valid(id));
    }

    #[test]
    fn test_invalid_handle() {
        let reg = HandleRegistry::new();
        assert!(!reg.is_valid(999));
        assert_eq!(reg.refcount(999), None);
        assert_eq!(reg.incref(999), None);
    }
}
