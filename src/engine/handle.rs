use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

/// The type used for internal object identifiers, analogous to HDF5's `hid_t`.
pub type Hid = i64;

/// Invalid handle ID sentinel.
pub const INVALID_HID: Hid = -1;

pub fn invalid_hid() -> Hid {
    INVALID_HID
}

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
    type_refcounts: RwLock<HashMap<HandleType, i32>>,
}

impl HandleRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            next_id: AtomicI64::new(1), // Start at 1, 0 and negative are reserved
            entries: RwLock::new(HashMap::new()),
            type_refcounts: RwLock::new(HashMap::new()),
        }
    }

    /// Invalid handle sentinel.
    pub fn invalid_hid() -> Hid {
        INVALID_HID
    }

    /// Terminate this registry package by clearing all registered handles.
    pub fn term_package(&self) {
        self.entries.write().clear();
        self.type_refcounts.write().clear();
    }

    /// Register a handle type in the local registry metadata.
    pub fn register_type_common(&self, handle_type: HandleType) -> HandleType {
        self.type_refcounts.write().entry(handle_type).or_insert(1);
        handle_type
    }

    /// Register a handle type.
    pub fn register_type(&self, handle_type: HandleType) -> HandleType {
        self.register_type_common(handle_type)
    }

    /// Legacy type-registration alias.
    pub fn register_type_v1(&self, handle_type: HandleType) -> HandleType {
        self.register_type_common(handle_type)
    }

    /// Current type-registration alias.
    pub fn register_type_v2(&self, handle_type: HandleType) -> HandleType {
        self.register_type_common(handle_type)
    }

    /// Return whether a type currently exists in this registry.
    pub fn type_exists(&self, handle_type: HandleType) -> bool {
        self.type_refcounts.read().contains_key(&handle_type)
            || self
                .entries
                .read()
                .values()
                .any(|entry| entry.handle_type == handle_type)
    }

    /// Register a new object and return its handle ID.
    pub fn register<T: Send + Sync + 'static>(&self, handle_type: HandleType, data: T) -> Hid {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = HandleEntry {
            handle_type,
            refcount: 1,
            data: Arc::new(data),
        };
        self.entries.write().insert(id, entry);
        self.register_type_common(handle_type);
        id
    }

    /// Internal register alias.
    pub fn register_internal<T: Send + Sync + 'static>(
        &self,
        handle_type: HandleType,
        data: T,
    ) -> Hid {
        self.register(handle_type, data)
    }

    /// Register an object using an existing id.
    pub fn register_using_existing_id<T: Send + Sync + 'static>(
        &self,
        id: Hid,
        handle_type: HandleType,
        data: T,
    ) -> Option<Hid> {
        if id <= 0 {
            return None;
        }
        let entry = HandleEntry {
            handle_type,
            refcount: 1,
            data: Arc::new(data),
        };
        self.entries.write().insert(id, entry);
        self.register_type_common(handle_type);
        Some(id)
    }

    /// Public existing-id registration alias.
    pub fn register_using_existing_id_api<T: Send + Sync + 'static>(
        &self,
        id: Hid,
        handle_type: HandleType,
        data: T,
    ) -> Option<Hid> {
        self.register_using_existing_id(id, handle_type, data)
    }

    /// Public register alias.
    pub fn register_api<T: Send + Sync + 'static>(&self, handle_type: HandleType, data: T) -> Hid {
        self.register(handle_type, data)
    }

    /// Future register alias. Async execution is not used here.
    pub fn register_future<T: Send + Sync + 'static>(
        &self,
        handle_type: HandleType,
        data: T,
    ) -> Hid {
        self.register(handle_type, data)
    }

    /// Replace the object associated with an id.
    pub fn subst<T: Send + Sync + 'static>(&self, id: Hid, data: T) -> bool {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&id) {
            entry.data = Arc::new(data);
            true
        } else {
            false
        }
    }

    /// Return true if this id is a file object.
    pub fn is_file_object(&self, id: Hid) -> bool {
        self.handle_type(id) == Some(HandleType::File)
    }

    /// Internal unwrap helper returning the id when valid.
    pub fn unwrap_id(&self, id: Hid) -> Option<Hid> {
        self.is_valid(id).then_some(id)
    }

    /// Mark-node helper. This compact registry has no separate mark bit.
    pub fn mark_node(&self, id: Hid) -> bool {
        self.is_valid(id)
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

    /// Public increment-ref alias.
    pub fn inc_ref(&self, id: Hid) -> Option<i32> {
        self.incref(id)
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

    /// Internal decrement-ref alias.
    pub fn dec_ref_internal(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Internal decrement application-ref alias.
    pub fn dec_app_ref_internal(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Public decrement application-ref alias.
    pub fn dec_app_ref(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Async decrement application-ref alias. Async execution is not used here.
    pub fn dec_app_ref_async(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Always-close decrement application-ref alias.
    pub fn dec_app_ref_always_close(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Internal always-close decrement application-ref alias.
    pub fn dec_app_ref_always_close_internal(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Async always-close decrement application-ref alias.
    pub fn dec_app_ref_always_close_async(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Public decrement-ref alias.
    pub fn dec_ref(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Get the current reference count.
    pub fn refcount(&self, id: Hid) -> Option<i32> {
        self.entries.read().get(&id).map(|e| e.refcount)
    }

    /// Public refcount alias.
    pub fn get_ref(&self, id: Hid) -> Option<i32> {
        self.refcount(id)
    }

    /// Get the handle type.
    pub fn handle_type(&self, id: Hid) -> Option<HandleType> {
        self.entries.read().get(&id).map(|e| e.handle_type)
    }

    /// Public handle-type alias.
    pub fn get_type(&self, id: Hid) -> Option<HandleType> {
        self.handle_type(id)
    }

    /// Verify an object by id and expected type.
    pub fn object_verify(&self, id: Hid, expected: HandleType) -> bool {
        self.handle_type(id) == Some(expected)
    }

    /// Check if a handle is valid (exists and has refcount > 0).
    pub fn is_valid(&self, id: Hid) -> bool {
        self.entries
            .read()
            .get(&id)
            .map_or(false, |e| e.refcount > 0)
    }

    /// Public validity alias.
    pub fn is_valid_api(&self, id: Hid) -> bool {
        self.is_valid(id)
    }

    /// Get the data associated with a handle, downcasted to the expected type.
    pub fn get<T: Send + Sync + 'static>(&self, id: Hid) -> Option<Arc<T>> {
        let entries = self.entries.read();
        entries
            .get(&id)
            .and_then(|e| e.data.clone().downcast::<T>().ok())
    }

    /// Remove an id and return whether it existed.
    pub fn remove(&self, id: Hid) -> bool {
        self.entries.write().remove(&id).is_some()
    }

    /// Internal remove-and-verify helper.
    pub fn remove_verify_internal(&self, id: Hid, expected: HandleType) -> bool {
        if self.object_verify(id, expected) {
            self.remove(id)
        } else {
            false
        }
    }

    /// Internal common remove helper.
    pub fn remove_common(&self, id: Hid) -> bool {
        self.remove(id)
    }

    /// Public remove-and-verify helper.
    pub fn remove_verify(&self, id: Hid, expected: HandleType) -> bool {
        self.remove_verify_internal(id, expected)
    }

    /// Clear all handles of a type. Returns the number removed.
    pub fn clear_type(&self, handle_type: HandleType) -> usize {
        let mut entries = self.entries.write();
        let before = entries.len();
        entries.retain(|_, entry| entry.handle_type != handle_type);
        before - entries.len()
    }

    /// Public clear-type alias.
    pub fn clear_type_api(&self, handle_type: HandleType) -> usize {
        self.clear_type(handle_type)
    }

    /// Destroy all handles of a type and remove its type metadata.
    pub fn destroy_type(&self, handle_type: HandleType) -> usize {
        self.type_refcounts.write().remove(&handle_type);
        self.clear_type(handle_type)
    }

    /// Internal destroy-type alias.
    pub fn destroy_type_internal(&self, handle_type: HandleType) -> usize {
        self.destroy_type(handle_type)
    }

    /// Public destroy-type alias.
    pub fn destroy_type_api(&self, handle_type: HandleType) -> usize {
        self.destroy_type(handle_type)
    }

    /// Number of members of a handle type.
    pub fn nmembers(&self, handle_type: HandleType) -> usize {
        self.entries
            .read()
            .values()
            .filter(|entry| entry.handle_type == handle_type)
            .count()
    }

    /// Public number-of-members alias.
    pub fn nmembers_api(&self, handle_type: HandleType) -> usize {
        self.nmembers(handle_type)
    }

    /// Increment a type reference count.
    pub fn inc_type_ref(&self, handle_type: HandleType) -> i32 {
        let mut refs = self.type_refcounts.write();
        let value = refs.entry(handle_type).or_insert(0);
        *value += 1;
        *value
    }

    /// Internal increment type-ref alias.
    pub fn inc_type_ref_internal(&self, handle_type: HandleType) -> i32 {
        self.inc_type_ref(handle_type)
    }

    /// Decrement a type reference count.
    pub fn dec_type_ref(&self, handle_type: HandleType) -> Option<i32> {
        let mut refs = self.type_refcounts.write();
        let value = refs.get_mut(&handle_type)?;
        *value -= 1;
        let out = *value;
        if out <= 0 {
            refs.remove(&handle_type);
        }
        Some(out)
    }

    /// Public no-underscore type-ref decrement alias.
    pub fn dec_type_ref_api(&self, handle_type: HandleType) -> Option<i32> {
        self.dec_type_ref(handle_type)
    }

    /// Get a type reference count.
    pub fn get_type_ref(&self, handle_type: HandleType) -> i32 {
        self.type_refcounts
            .read()
            .get(&handle_type)
            .copied()
            .unwrap_or(0)
    }

    /// Internal get-type-ref alias.
    pub fn get_type_ref_internal(&self, handle_type: HandleType) -> i32 {
        self.get_type_ref(handle_type)
    }

    /// Iterate over ids of a type.
    pub fn iterate<F>(&self, handle_type: HandleType, mut callback: F)
    where
        F: FnMut(Hid),
    {
        let ids: Vec<_> = self
            .entries
            .read()
            .iter()
            .filter_map(|(&id, entry)| (entry.handle_type == handle_type).then_some(id))
            .collect();
        for id in ids {
            callback(id);
        }
    }

    /// Internal iterate callback adapter.
    pub fn iterate_cb<F>(&self, handle_type: HandleType, callback: F)
    where
        F: FnMut(Hid),
    {
        self.iterate(handle_type, callback);
    }

    /// Public iterate callback adapter.
    pub fn iterate_pub_cb<F>(&self, handle_type: HandleType, callback: F)
    where
        F: FnMut(Hid),
    {
        self.iterate(handle_type, callback);
    }

    /// Public iterate alias.
    pub fn iterate_api<F>(&self, handle_type: HandleType, callback: F)
    where
        F: FnMut(Hid),
    {
        self.iterate(handle_type, callback);
    }

    /// Find an id of a type matching a predicate.
    pub fn find_id<F>(&self, handle_type: HandleType, mut predicate: F) -> Option<Hid>
    where
        F: FnMut(Hid) -> bool,
    {
        self.entries.read().iter().find_map(|(&id, entry)| {
            (entry.handle_type == handle_type && predicate(id)).then_some(id)
        })
    }

    /// Internal search callback adapter.
    pub fn search_cb<F>(&self, handle_type: HandleType, predicate: F) -> Option<Hid>
    where
        F: FnMut(Hid) -> bool,
    {
        self.find_id(handle_type, predicate)
    }

    /// Public search alias.
    pub fn search<F>(&self, handle_type: HandleType, predicate: F) -> Option<Hid>
    where
        F: FnMut(Hid) -> bool,
    {
        self.find_id(handle_type, predicate)
    }

    /// Return a file id for file objects.
    pub fn get_file_id(&self, id: Hid) -> Option<Hid> {
        self.is_file_object(id).then_some(id)
    }

    /// Return an implementation-defined object name.
    pub fn get_name(&self, id: Hid) -> Option<String> {
        self.handle_type(id)
            .map(|handle_type| format!("{handle_type:?}:{id}"))
    }

    /// Internal name-test helper.
    pub fn get_name_test(&self, id: Hid) -> Option<String> {
        self.get_name(id)
    }

    /// Dump ids for a type.
    pub fn dump_ids_for_type(&self, handle_type: HandleType) -> Vec<Hid> {
        let mut ids = Vec::new();
        self.iterate(handle_type, |id| ids.push(id));
        ids
    }

    /// Internal dump callback helper.
    pub fn id_dump_cb(&self, handle_type: HandleType) -> Vec<Hid> {
        self.dump_ids_for_type(handle_type)
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

    #[test]
    fn test_type_registry_and_iteration_aliases() {
        let reg = HandleRegistry::new();
        assert_eq!(HandleRegistry::invalid_hid(), INVALID_HID);
        assert_eq!(reg.register_type(HandleType::Dataset), HandleType::Dataset);
        assert!(reg.type_exists(HandleType::Dataset));
        assert_eq!(reg.get_type_ref(HandleType::Dataset), 1);
        assert_eq!(reg.inc_type_ref(HandleType::Dataset), 2);
        assert_eq!(reg.dec_type_ref_api(HandleType::Dataset), Some(1));

        let a = reg.register_api(HandleType::Dataset, "a");
        let b = reg.register_future(HandleType::Dataset, "b");
        assert_eq!(reg.nmembers(HandleType::Dataset), 2);
        assert_eq!(reg.find_id(HandleType::Dataset, |id| id == b), Some(b));
        assert_eq!(reg.search(HandleType::Dataset, |id| id == a), Some(a));

        let mut seen = Vec::new();
        reg.iterate_api(HandleType::Dataset, |id| seen.push(id));
        seen.sort_unstable();
        assert_eq!(seen, vec![a, b]);
        assert_eq!(reg.dump_ids_for_type(HandleType::Dataset).len(), 2);
        assert_eq!(reg.get_name(a), Some(format!("Dataset:{a}")));

        assert!(reg.subst(a, "replacement"));
        assert_eq!(&*reg.get::<&str>(a).unwrap(), &"replacement");
        assert!(reg.remove_verify(a, HandleType::Dataset));
        assert!(!reg.is_valid(a));
        assert_eq!(reg.clear_type_api(HandleType::Dataset), 1);
        assert_eq!(reg.nmembers_api(HandleType::Dataset), 0);
        reg.term_package();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_existing_id_and_ref_aliases() {
        let reg = HandleRegistry::new();
        assert_eq!(
            reg.register_using_existing_id_api(44, HandleType::File, "file"),
            Some(44)
        );
        assert_eq!(reg.unwrap_id(44), Some(44));
        assert!(reg.mark_node(44));
        assert!(reg.is_file_object(44));
        assert_eq!(reg.get_file_id(44), Some(44));
        assert!(reg.object_verify(44, HandleType::File));
        assert_eq!(reg.get_type(44), Some(HandleType::File));
        assert_eq!(reg.inc_ref(44), Some(2));
        assert_eq!(reg.get_ref(44), Some(2));
        assert_eq!(reg.dec_app_ref(44), Some(1));
        assert_eq!(reg.dec_app_ref_always_close(44), Some(0));
        assert!(!reg.is_valid_api(44));
    }
}
