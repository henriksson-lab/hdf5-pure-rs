use std::collections::BTreeMap;

use crate::error::{Error, Result};

#[path = "vol_explicit_wrappers.rs"]
pub mod explicit_vol_wrappers;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VolConnector {
    pub id: u64,
    pub name: String,
    pub value: u64,
    pub refcount: usize,
    pub cap_flags: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VolRegistry {
    pub connectors: BTreeMap<u64, VolConnector>,
    pub by_name: BTreeMap<String, u64>,
    pub default_conn: Option<u64>,
    pub next_id: u64,
    pub optional_ops: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VolObject {
    pub connector_id: u64,
    pub name: String,
    pub payload: Vec<u8>,
    pub wrapped: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VolLibState {
    pub default_conn: Option<u64>,
    pub wrapper_depth: usize,
}

#[allow(non_snake_case)]
pub fn H5VL_init_phase1() -> VolRegistry {
    H5VL__init_package()
}

#[allow(non_snake_case)]
pub fn H5VL_init_phase2(registry: &mut VolRegistry) {
    if registry.next_id == 0 {
        registry.next_id = 1;
    }
}

#[allow(non_snake_case)]
pub fn H5VL__init_package() -> VolRegistry {
    let mut registry = VolRegistry {
        next_id: 1,
        ..VolRegistry::default()
    };
    H5VL__native_register(&mut registry);
    registry
}

#[allow(non_snake_case)]
pub fn H5VL_term_package(registry: &mut VolRegistry) {
    registry.connectors.clear();
    registry.by_name.clear();
    registry.default_conn = None;
    registry.optional_ops.clear();
}

#[allow(non_snake_case)]
pub fn H5VL__free_cls(_connector: VolConnector) {}

#[allow(non_snake_case)]
pub fn H5VL__set_def_conn(registry: &mut VolRegistry, id: u64) -> Result<()> {
    if registry.connectors.contains_key(&id) {
        registry.default_conn = Some(id);
        Ok(())
    } else {
        Err(Error::InvalidFormat(
            "VOL connector id is not registered".into(),
        ))
    }
}

#[allow(non_snake_case)]
pub fn H5VL__wrap_obj(mut object: VolObject) -> VolObject {
    object.wrapped = true;
    object
}

#[allow(non_snake_case)]
pub fn H5VL_new_vol_obj(connector_id: u64, name: impl Into<String>) -> VolObject {
    VolObject {
        connector_id,
        name: name.into(),
        ..VolObject::default()
    }
}

#[allow(non_snake_case)]
pub fn H5VL_conn_prop_copy(connector: &VolConnector) -> VolConnector {
    connector.clone()
}

#[allow(non_snake_case)]
pub fn H5VL_conn_prop_cmp(left: &VolConnector, right: &VolConnector) -> std::cmp::Ordering {
    left.name
        .cmp(&right.name)
        .then_with(|| left.value.cmp(&right.value))
}

#[allow(non_snake_case)]
pub fn H5VL_conn_prop_free(_connector: VolConnector) {}

#[allow(non_snake_case)]
pub fn H5VL_register(registry: &mut VolRegistry, name: &str, value: u64) -> u64 {
    H5VL__register_connector_by_name(registry, name, value)
}

#[allow(non_snake_case)]
pub fn H5VL_register_using_existing_id(
    registry: &mut VolRegistry,
    id: u64,
    name: &str,
    value: u64,
) -> u64 {
    H5VL__register_connector_by_value(registry, id, name, value)
}

#[allow(non_snake_case)]
pub fn H5VL_conn_register(registry: &mut VolRegistry, connector: VolConnector) -> u64 {
    let id = connector.id;
    registry.by_name.insert(connector.name.clone(), id);
    registry.connectors.insert(id, connector);
    id
}

#[allow(non_snake_case)]
pub fn H5VL__conn_find<'a>(registry: &'a VolRegistry, name: &str) -> Option<&'a VolConnector> {
    let id = registry.by_name.get(name)?;
    registry.connectors.get(id)
}

#[allow(non_snake_case)]
pub fn H5VL_conn_inc_rc(registry: &mut VolRegistry, id: u64) -> Result<usize> {
    let connector = registry
        .connectors
        .get_mut(&id)
        .ok_or_else(|| Error::InvalidFormat("VOL connector id is not registered".into()))?;
    connector.refcount = connector.refcount.saturating_add(1);
    Ok(connector.refcount)
}

#[allow(non_snake_case)]
pub fn H5VL_conn_dec_rc(registry: &mut VolRegistry, id: u64) -> Result<usize> {
    let connector = registry
        .connectors
        .get_mut(&id)
        .ok_or_else(|| Error::InvalidFormat("VOL connector id is not registered".into()))?;
    connector.refcount = connector.refcount.saturating_sub(1);
    Ok(connector.refcount)
}

#[allow(non_snake_case)]
pub fn H5VL__conn_free(registry: &mut VolRegistry, id: u64) {
    if let Some(connector) = registry.connectors.remove(&id) {
        registry.by_name.remove(&connector.name);
    }
}

#[allow(non_snake_case)]
pub fn H5VL__conn_free_id(registry: &mut VolRegistry, id: u64) {
    H5VL__conn_free(registry, id);
}

#[allow(non_snake_case)]
pub fn H5VL_file_is_same(left: &VolObject, right: &VolObject) -> bool {
    left.connector_id == right.connector_id && left.name == right.name
}

#[allow(non_snake_case)]
pub fn H5VL__register_connector_by_class(
    registry: &mut VolRegistry,
    connector: VolConnector,
) -> u64 {
    H5VL_conn_register(registry, connector)
}

#[allow(non_snake_case)]
pub fn H5VL__register_connector_by_name(registry: &mut VolRegistry, name: &str, value: u64) -> u64 {
    let id = registry.next_id.max(1);
    registry.next_id = id.saturating_add(1);
    let connector = VolConnector {
        id,
        name: name.to_string(),
        value,
        refcount: 1,
        cap_flags: 0,
    };
    H5VL_conn_register(registry, connector)
}

#[allow(non_snake_case)]
pub fn H5VL__register_connector_by_value(
    registry: &mut VolRegistry,
    id: u64,
    name: &str,
    value: u64,
) -> u64 {
    let connector = VolConnector {
        id,
        name: name.to_string(),
        value,
        refcount: 1,
        cap_flags: 0,
    };
    H5VL_conn_register(registry, connector)
}

#[allow(non_snake_case)]
pub fn H5VL__get_connector_by_name<'a>(
    registry: &'a VolRegistry,
    name: &str,
) -> Option<&'a VolConnector> {
    H5VL__conn_find(registry, name)
}

#[allow(non_snake_case)]
pub fn H5VL__get_connector_by_value<'a>(
    registry: &'a VolRegistry,
    value: u64,
) -> Option<&'a VolConnector> {
    registry
        .connectors
        .values()
        .find(|connector| connector.value == value)
}

#[allow(non_snake_case)]
pub fn H5VL__get_connector_by_id<'a>(
    registry: &'a VolRegistry,
    id: u64,
) -> Option<&'a VolConnector> {
    registry.connectors.get(&id)
}

#[allow(non_snake_case)]
pub fn H5VL_connector_get_name(connector: &VolConnector) -> &str {
    &connector.name
}

#[allow(non_snake_case)]
pub fn H5VL_connector_get_value(connector: &VolConnector) -> u64 {
    connector.value
}

#[allow(non_snake_case)]
pub fn H5VL_connector_get_id(connector: &VolConnector) -> u64 {
    connector.id
}

#[allow(non_snake_case)]
pub fn H5VL_connector_set_cap_flags(connector: &mut VolConnector, flags: u64) {
    connector.cap_flags = flags;
}

#[allow(non_snake_case)]
pub fn H5VL__connector_names(registry: &VolRegistry) -> Vec<String> {
    registry.by_name.keys().cloned().collect()
}

#[allow(non_snake_case)]
pub fn H5VL_vol_object(object: &VolObject) -> &VolObject {
    object
}

#[allow(non_snake_case)]
pub fn H5VL_object_unwrap(mut object: VolObject) -> VolObject {
    object.wrapped = false;
    object
}

#[allow(non_snake_case)]
pub fn H5VL_object(object: &VolObject) -> &VolObject {
    object
}

#[allow(non_snake_case)]
pub fn H5VL_retrieve_lib_state(registry: &VolRegistry) -> VolLibState {
    VolLibState {
        default_conn: registry.default_conn,
        wrapper_depth: 0,
    }
}

#[allow(non_snake_case)]
pub fn H5VL_start_lib_state(state: &mut VolLibState) {
    state.wrapper_depth = state.wrapper_depth.saturating_add(1);
}

#[allow(non_snake_case)]
pub fn H5VL_restore_lib_state(registry: &mut VolRegistry, state: &VolLibState) {
    registry.default_conn = state.default_conn;
}

#[allow(non_snake_case)]
pub fn H5VL_finish_lib_state(state: &mut VolLibState) {
    state.wrapper_depth = state.wrapper_depth.saturating_sub(1);
}

#[allow(non_snake_case)]
pub fn H5VL_free_lib_state(_state: VolLibState) {}

#[allow(non_snake_case)]
pub fn H5VL_set_vol_wrapper(object: &mut VolObject) {
    object.wrapped = true;
}

#[allow(non_snake_case)]
pub fn H5VL_inc_vol_wrapper(state: &mut VolLibState) {
    state.wrapper_depth = state.wrapper_depth.saturating_add(1);
}

#[allow(non_snake_case)]
pub fn H5VL_dec_vol_wrapper(state: &mut VolLibState) {
    state.wrapper_depth = state.wrapper_depth.saturating_sub(1);
}

#[allow(non_snake_case)]
pub fn H5VL_reset_vol_wrapper(state: &mut VolLibState) {
    state.wrapper_depth = 0;
}

#[allow(non_snake_case)]
pub fn H5VL_wrap_register(object: &mut VolObject) {
    object.wrapped = true;
}

#[allow(non_snake_case)]
pub fn H5VL_check_plugin_load(_name: &str) -> Result<()> {
    Err(Error::Unsupported(
        "dynamic VOL plugin loading is unsupported in pure-Rust mode".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5VL__is_default_conn(registry: &VolRegistry, id: u64) -> bool {
    registry.default_conn == Some(id)
}

#[allow(non_snake_case)]
pub fn H5VL_setup_args(args: &[String]) -> Vec<String> {
    args.to_vec()
}

#[allow(non_snake_case)]
pub fn H5VL_setup_loc_args(name: &str) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5VL_setup_acc_args(flags: u64) -> u64 {
    flags
}

#[allow(non_snake_case)]
pub fn H5VL_setup_self_args(object: &VolObject) -> VolObject {
    object.clone()
}

#[allow(non_snake_case)]
pub fn H5VL_setup_name_args(name: &str) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5VL_setup_idx_args(index: usize) -> usize {
    index
}

#[allow(non_snake_case)]
pub fn H5VL_setup_token_args(token: u64) -> u64 {
    token
}

#[allow(non_snake_case)]
pub fn H5VL_conn_prop_get_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

#[allow(non_snake_case)]
pub fn H5VL__passthru_register(registry: &mut VolRegistry) -> u64 {
    H5VL__register_connector_by_name(registry, "pass_through", 0)
}

#[allow(non_snake_case)]
pub fn H5VL__passthru_unregister(registry: &mut VolRegistry) {
    if let Some(id) = registry.by_name.get("pass_through").copied() {
        H5VL__conn_free(registry, id);
    }
}

#[allow(non_snake_case)]
pub fn H5VL__release_dyn_op(registry: &mut VolRegistry, name: &str) {
    registry.optional_ops.remove(name);
}

#[allow(non_snake_case)]
pub fn H5VL__term_opt_operation_cb(_name: &str, _value: u64) {}

#[allow(non_snake_case)]
pub fn H5VL__term_opt_operation(registry: &mut VolRegistry) {
    registry.optional_ops.clear();
}

#[allow(non_snake_case)]
pub fn H5VL__register_opt_operation(registry: &mut VolRegistry, name: &str, value: u64) {
    registry.optional_ops.insert(name.to_string(), value);
}

#[allow(non_snake_case)]
pub fn H5VL__num_opt_operation(registry: &VolRegistry) -> usize {
    registry.optional_ops.len()
}

#[allow(non_snake_case)]
pub fn H5VL__find_opt_operation(registry: &VolRegistry, name: &str) -> Option<u64> {
    registry.optional_ops.get(name).copied()
}

#[allow(non_snake_case)]
pub fn H5VL__opt_operation_names(registry: &VolRegistry) -> Vec<String> {
    registry.optional_ops.keys().cloned().collect()
}

#[allow(non_snake_case)]
pub fn H5VL__unregister_opt_operation(registry: &mut VolRegistry, name: &str) {
    registry.optional_ops.remove(name);
}

#[allow(non_snake_case)]
pub fn H5VL__native_blob_put(object: &mut VolObject, data: &[u8]) {
    object.payload.clear();
    object.payload.extend_from_slice(data);
}

#[allow(non_snake_case)]
pub fn H5VL__native_blob_get(object: &VolObject) -> Vec<u8> {
    object.payload.clone()
}

#[allow(non_snake_case)]
pub fn H5VL__native_blob_specific(object: &VolObject) -> usize {
    object.payload.len()
}

#[allow(non_snake_case)]
pub fn H5VL__native_attr_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

#[allow(non_snake_case)]
pub fn H5VL__native_attr_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_create(parent, name)
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_new_obj(object: VolObject) -> VolObject {
    H5VL__wrap_obj(object)
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_free_obj(_object: VolObject) {}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_init() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_info_copy(connector: &VolConnector) -> VolConnector {
    connector.clone()
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_info_free(_connector: VolConnector) {}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_info_to_str(connector: &VolConnector) -> String {
    connector.name.clone()
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_str_to_info(name: &str) -> VolConnector {
    VolConnector {
        name: name.to_string(),
        ..VolConnector::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_registry_exposes_connector_introspection() {
        let mut registry = H5VL__init_package();
        let id = H5VL_register(&mut registry, "audit_vol", 42);
        H5VL__set_def_conn(&mut registry, id).unwrap();

        let connector = H5VL__get_connector_by_name(&registry, "audit_vol").unwrap();
        assert_eq!(H5VL_connector_get_id(connector), id);
        assert_eq!(H5VL_connector_get_name(connector), "audit_vol");
        assert_eq!(H5VL_connector_get_value(connector), 42);
        assert!(H5VL__connector_names(&registry).contains(&"audit_vol".to_string()));
        assert!(H5VL__is_default_conn(&registry, id));

        let connector = registry.connectors.get_mut(&id).unwrap();
        H5VL_connector_set_cap_flags(connector, 0x1234);
        assert_eq!(H5VL_conn_prop_get_cap_flags(connector), 0x1234);
    }

    #[test]
    fn vol_optional_operation_registry_lists_and_releases_ops() {
        let mut registry = H5VL__init_package();
        H5VL__register_opt_operation(&mut registry, "flush_async", 9);
        H5VL__register_opt_operation(&mut registry, "snapshot", 11);

        assert_eq!(H5VL__num_opt_operation(&registry), 2);
        assert_eq!(H5VL__find_opt_operation(&registry, "snapshot"), Some(11));
        assert!(H5VL__opt_operation_names(&registry).contains(&"flush_async".to_string()));

        H5VL__release_dyn_op(&mut registry, "flush_async");
        assert_eq!(H5VL__find_opt_operation(&registry, "flush_async"), None);
        H5VL__term_opt_operation(&mut registry);
        assert_eq!(H5VL__num_opt_operation(&registry), 0);
    }

    #[test]
    fn vol_registry_registers_and_finds_connectors() {
        let mut registry = VolRegistry::default();
        let id = H5VL_register(&mut registry, "native", 0);
        H5VL__set_def_conn(&mut registry, id).unwrap();
        assert!(H5VL__is_default_conn(&registry, id));
        assert_eq!(
            H5VL__get_connector_by_name(&registry, "native").unwrap().id,
            id
        );
    }
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_get_object(object: &VolObject) -> &VolObject {
    object
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_get_wrap_ctx(object: &VolObject) -> bool {
    object.wrapped
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_wrap_object(object: VolObject) -> VolObject {
    H5VL__wrap_obj(object)
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_unwrap_object(object: VolObject) -> VolObject {
    H5VL_object_unwrap(object)
}

#[allow(non_snake_case)]
pub fn H5VL_pass_through_free_wrap_ctx(_wrapped: bool) {}

#[allow(non_snake_case)]
pub fn H5VL__native_register(registry: &mut VolRegistry) -> u64 {
    H5VL__register_connector_by_value(registry, 0, "native", 0)
}

#[allow(non_snake_case)]
pub fn H5VL__native_unregister(registry: &mut VolRegistry) {
    H5VL__conn_free(registry, 0);
}
