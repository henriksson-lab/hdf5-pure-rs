#![allow(dead_code, non_snake_case)]

use super::{
    H5VL__conn_free, H5VL__get_connector_by_name, H5VL__get_connector_by_value,
    H5VL__native_attr_create, H5VL__native_attr_open, H5VL__native_blob_get, H5VL__native_blob_put,
    H5VL__native_blob_specific, H5VL__register_connector_by_name,
    H5VL__register_connector_by_value, H5VL__set_def_conn, H5VL__wrap_obj, H5VL_conn_dec_rc,
    H5VL_conn_inc_rc, H5VL_new_vol_obj, H5VL_object_unwrap, H5VL_pass_through_get_wrap_ctx,
    H5VL_restore_lib_state, H5VL_retrieve_lib_state, H5VL_wrap_register, VolConnector, VolLibState,
    VolObject, VolRegistry,
};
use crate::error::{Error, Result};

fn unsupported_vol(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires a concrete VOL connector operation that is not implemented"
    ))
}

pub fn H5VL_pass_through_attr_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_create(parent, name)
}

pub fn H5VL_pass_through_attr_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_open(parent, name)
}

pub fn H5VL_pass_through_attr_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VL_pass_through_attr_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_attr_specific"))
}

pub fn H5VL_pass_through_dataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_pass_through_dataset_read(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
}

pub fn H5VL_pass_through_dataset_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VL_pass_through_dataset_get(object: &VolObject) -> usize {
    object.payload.len()
}

pub fn H5VL_pass_through_dataset_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_dataset_specific"))
}

pub fn H5VL_pass_through_dataset_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_dataset_optional"))
}

pub fn H5VL_pass_through_dataset_close(_object: VolObject) {}

pub fn H5VL_pass_through_datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_pass_through_datatype_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_pass_through_datatype_get(object: &VolObject) -> usize {
    object.payload.len()
}

pub fn H5VL_pass_through_datatype_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_datatype_specific"))
}

pub fn H5VL_pass_through_datatype_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_datatype_optional"))
}

pub fn H5VL_pass_through_datatype_close(_object: VolObject) {}

pub fn H5VL_pass_through_file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

pub fn H5VL_pass_through_file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

pub fn H5VL_pass_through_file_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL_pass_through_file_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_file_specific"))
}

pub fn H5VL_pass_through_file_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_file_optional"))
}

pub fn H5VL_pass_through_file_close(_object: VolObject) {}

pub fn H5VL_pass_through_group_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_pass_through_group_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_pass_through_group_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL_pass_through_group_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_group_specific"))
}

pub fn H5VL_pass_through_group_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_group_optional"))
}

pub fn H5VL_pass_through_link_copy(_src: &VolObject, _dst: &mut VolObject) -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_copy"))
}

pub fn H5VL_pass_through_link_move(_src: &mut VolObject, _dst: &mut VolObject) -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_move"))
}

pub fn H5VL_pass_through_link_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL_pass_through_link_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_specific"))
}

pub fn H5VL_pass_through_link_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_optional"))
}

pub fn H5VL_pass_through_object_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_pass_through_object_copy(object: &VolObject) -> VolObject {
    object.clone()
}

pub fn H5VL_pass_through_object_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL_pass_through_object_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_object_specific"))
}

pub fn H5VL_pass_through_object_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_object_optional"))
}

pub fn H5VL_pass_through_introspect_get_conn_cls(connector: &VolConnector) -> VolConnector {
    connector.clone()
}

pub fn H5VL_pass_through_introspect_get_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

pub fn H5VL_pass_through_introspect_opt_query() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_introspect_opt_query"))
}

pub fn H5VL_pass_through_request_wait() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_wait"))
}

pub fn H5VL_pass_through_request_notify() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_notify"))
}

pub fn H5VL_pass_through_request_cancel() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_cancel"))
}

pub fn H5VL_pass_through_request_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_specific"))
}

pub fn H5VL_pass_through_request_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_optional"))
}

pub fn H5VL_pass_through_request_free() {}

pub fn H5VL_pass_through_blob_put(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VL_pass_through_blob_get(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
}

pub fn H5VL_pass_through_blob_specific(object: &VolObject) -> usize {
    H5VL__native_blob_specific(object)
}

pub fn H5VL_pass_through_blob_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_blob_optional"))
}

pub fn H5VL_pass_through_token_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

pub fn H5VL_pass_through_token_to_str(token: u64) -> String {
    token.to_string()
}

pub fn H5VL_pass_through_token_from_str(token: &str) -> Result<u64> {
    token
        .parse()
        .map_err(|_| Error::InvalidFormat("invalid VOL token string".into()))
}

pub fn H5VL_pass_through_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_optional"))
}

pub fn H5VLregister_connector(registry: &mut VolRegistry, name: &str, value: u64) -> u64 {
    H5VL__register_connector_by_name(registry, name, value)
}

pub fn H5VLregister_connector_by_name(registry: &mut VolRegistry, name: &str, value: u64) -> u64 {
    H5VL__register_connector_by_name(registry, name, value)
}

pub fn H5VLregister_connector_by_value(
    registry: &mut VolRegistry,
    id: u64,
    name: &str,
    value: u64,
) -> u64 {
    H5VL__register_connector_by_value(registry, id, name, value)
}

pub fn H5VLget_connector_id_by_name(registry: &VolRegistry, name: &str) -> Option<u64> {
    H5VL__get_connector_by_name(registry, name).map(|connector| connector.id)
}

pub fn H5VLget_connector_id_by_value(registry: &VolRegistry, value: u64) -> Option<u64> {
    H5VL__get_connector_by_value(registry, value).map(|connector| connector.id)
}

pub fn H5VLget_connector_name(registry: &VolRegistry, id: u64) -> Option<String> {
    registry
        .connectors
        .get(&id)
        .map(|connector| connector.name.clone())
}

pub fn H5VLclose(registry: &mut VolRegistry, id: u64) {
    H5VL__conn_free(registry, id);
}

pub fn H5VLunregister_connector(registry: &mut VolRegistry, id: u64) {
    H5VL__conn_free(registry, id);
}

pub fn H5VLcmp_connector_cls(left: &VolConnector, right: &VolConnector) -> std::cmp::Ordering {
    left.name
        .cmp(&right.name)
        .then(left.value.cmp(&right.value))
}

pub fn H5VLwrap_register(object: &mut VolObject) {
    H5VL_wrap_register(object);
}

pub fn H5VLretrieve_lib_state(registry: &VolRegistry) -> VolLibState {
    H5VL_retrieve_lib_state(registry)
}

pub fn H5VLrestore_lib_state(registry: &mut VolRegistry, state: &VolLibState) {
    H5VL_restore_lib_state(registry, state);
}

pub fn H5VLfree_lib_state(_state: VolLibState) {}

pub fn H5VL__native_link_create() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_create"))
}

pub fn H5VL__native_link_copy() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_copy"))
}

pub fn H5VL__native_link_move() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_move"))
}

pub fn H5VL__native_link_get() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_get"))
}

pub fn H5VL__native_link_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_specific"))
}

pub fn H5VL__native_introspect_get_conn_cls(connector: &VolConnector) -> VolConnector {
    connector.clone()
}

pub fn H5VL__native_introspect_get_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

pub fn H5VL_native_get_file_addr_len() -> usize {
    std::mem::size_of::<u64>()
}

pub fn H5VL__native_get_file_addr_len() -> usize {
    H5VL_native_get_file_addr_len()
}

pub fn H5VL__native_object_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_object_copy(object: &VolObject) -> VolObject {
    object.clone()
}

pub fn H5VL__native_object_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL__native_object_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_object_specific"))
}

pub fn H5VL__native_object_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_object_optional"))
}

pub fn H5VL__native_group_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_group_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_group_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL__native_group_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_group_specific"))
}

pub fn H5VL__native_group_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_group_optional"))
}

pub fn H5VL__native_group_close(_object: VolObject) {}

pub fn H5VL__native_token_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

pub fn H5VL__native_token_to_str(token: u64) -> String {
    token.to_string()
}

pub fn H5VL__native_str_to_token(token: &str) -> Result<u64> {
    H5VL_pass_through_token_from_str(token)
}

pub fn H5VLget_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

pub fn H5VLget_value(connector: &VolConnector) -> u64 {
    connector.value
}

pub fn H5VLfree_wrap_ctx(_wrapped: bool) {}

pub fn H5VLattr_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_create(parent, name)
}

pub fn H5VLattr_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_open(parent, name)
}

pub fn H5VLattr_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VLattr_get(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
}

pub fn H5VL__attr_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__attr_specific"))
}

pub fn H5VL_attr_specific() -> Result<()> {
    H5VL__attr_specific()
}

pub fn H5VLattr_specific() -> Result<()> {
    H5VL__attr_specific()
}

pub fn H5VL__attr_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__attr_optional"))
}

pub fn H5VLattr_optional() -> Result<()> {
    H5VL__attr_optional()
}

pub fn H5VLattr_optional_op() -> Result<()> {
    H5VL__attr_optional()
}

pub fn H5VLattr_close(_object: VolObject) {}

pub fn H5VLdataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VLdataset_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VLdataset_read(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
}

pub fn H5VLdataset_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VLdataset_get(object: &VolObject) -> usize {
    object.payload.len()
}

pub fn H5VL__dataset_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__dataset_specific"))
}

pub fn H5VL_dataset_specific() -> Result<()> {
    H5VL__dataset_specific()
}

pub fn H5VLdataset_specific() -> Result<()> {
    H5VL__dataset_specific()
}

pub fn H5VL__dataset_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__dataset_optional"))
}

pub fn H5VL_dataset_optional() -> Result<()> {
    H5VL__dataset_optional()
}

pub fn H5VLdataset_optional() -> Result<()> {
    H5VL__dataset_optional()
}

pub fn H5VLdataset_optional_op() -> Result<()> {
    H5VL__dataset_optional()
}

pub fn H5VLdataset_close(_object: VolObject) {}

pub fn H5VL__datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL_datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL__datatype_commit(parent, name)
}

pub fn H5VLdatatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL__datatype_commit(parent, name)
}

pub fn H5VLdatatype_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__datatype_get(object: &VolObject) -> usize {
    object.payload.len()
}

pub fn H5VLdatatype_get(object: &VolObject) -> usize {
    H5VL__datatype_get(object)
}

pub fn H5VL__datatype_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__datatype_specific"))
}

pub fn H5VL_datatype_specific() -> Result<()> {
    H5VL__datatype_specific()
}

pub fn H5VLdatatype_specific() -> Result<()> {
    H5VL__datatype_specific()
}

pub fn H5VL__datatype_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__datatype_optional"))
}

pub fn H5VL_datatype_optional() -> Result<()> {
    H5VL__datatype_optional()
}

pub fn H5VL_datatype_optional_op() -> Result<()> {
    H5VL__datatype_optional()
}

pub fn H5VLdatatype_optional() -> Result<()> {
    H5VL__datatype_optional()
}

pub fn H5VLdatatype_optional_op() -> Result<()> {
    H5VL__datatype_optional()
}

pub fn H5VL__datatype_close(_object: VolObject) {}

pub fn H5VLdatatype_close(object: VolObject) {
    H5VL__datatype_close(object);
}

pub fn H5VL__file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

pub fn H5VL_file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_create(connector_id, name)
}

pub fn H5VLfile_create(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_create(connector_id, name)
}

pub fn H5VL__file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

pub fn H5VL__file_open_find_connector_cb<'a>(
    registry: &'a VolRegistry,
    name: &str,
) -> Option<&'a VolConnector> {
    H5VL__get_connector_by_name(registry, name)
}

pub fn H5VL_file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_open(connector_id, name)
}

pub fn H5VLfile_open(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_open(connector_id, name)
}

pub fn H5VLfile_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL__file_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__file_specific"))
}

pub fn H5VLfile_specific() -> Result<()> {
    H5VL__file_specific()
}

pub fn H5VLfile_optional() -> Result<()> {
    Err(unsupported_vol("H5VLfile_optional"))
}

pub fn H5VLfile_optional_op() -> Result<()> {
    H5VLfile_optional()
}

pub fn H5VLfile_close(_object: VolObject) {}

pub fn H5VLgroup_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VLgroup_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VLgroup_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL__group_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__group_specific"))
}

pub fn H5VL_group_specific() -> Result<()> {
    H5VL__group_specific()
}

pub fn H5VLgroup_specific() -> Result<()> {
    H5VL__group_specific()
}

pub fn H5VL__group_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__group_optional"))
}

pub fn H5VL_group_optional() -> Result<()> {
    H5VL__group_optional()
}

pub fn H5VLgroup_optional() -> Result<()> {
    H5VL__group_optional()
}

pub fn H5VLgroup_optional_op() -> Result<()> {
    H5VL__group_optional()
}

pub fn H5VLgroup_close(_object: VolObject) {}

pub fn H5VLlink_create() -> Result<()> {
    Err(unsupported_vol("H5VLlink_create"))
}

pub fn H5VLlink_copy() -> Result<()> {
    Err(unsupported_vol("H5VLlink_copy"))
}

pub fn H5VLlink_move() -> Result<()> {
    Err(unsupported_vol("H5VLlink_move"))
}

pub fn H5VLlink_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL__link_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__link_specific"))
}

pub fn H5VL_link_specific() -> Result<()> {
    H5VL__link_specific()
}

pub fn H5VLlink_specific() -> Result<()> {
    H5VL__link_specific()
}

pub fn H5VL__link_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__link_optional"))
}

pub fn H5VL_link_optional() -> Result<()> {
    H5VL__link_optional()
}

pub fn H5VLlink_optional() -> Result<()> {
    H5VL__link_optional()
}

pub fn H5VLlink_optional_op() -> Result<()> {
    H5VL__link_optional()
}

pub fn H5VLobject_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VLobject_copy(object: &VolObject) -> VolObject {
    object.clone()
}

pub fn H5VLobject_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VLobject_specific() -> Result<()> {
    Err(unsupported_vol("H5VLobject_specific"))
}

pub fn H5VL__object_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__object_optional"))
}

pub fn H5VL_object_optional() -> Result<()> {
    H5VL__object_optional()
}

pub fn H5VLobject_optional() -> Result<()> {
    H5VL__object_optional()
}

pub fn H5VLobject_optional_op() -> Result<()> {
    H5VL__object_optional()
}

pub fn H5VLintrospect_opt_query() -> Result<()> {
    Err(unsupported_vol("H5VLintrospect_opt_query"))
}

pub fn H5VLrequest_wait() -> Result<()> {
    Err(unsupported_vol("H5VLrequest_wait"))
}

pub fn H5VLrequest_notify() -> Result<()> {
    Err(unsupported_vol("H5VLrequest_notify"))
}

pub fn H5VLrequest_cancel() -> Result<()> {
    Err(unsupported_vol("H5VLrequest_cancel"))
}

pub fn H5VL__request_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__request_specific"))
}

pub fn H5VL_request_specific() -> Result<()> {
    H5VL__request_specific()
}

pub fn H5VLrequest_specific() -> Result<()> {
    H5VL__request_specific()
}

pub fn H5VL__request_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__request_optional"))
}

pub fn H5VL_request_optional() -> Result<()> {
    H5VL__request_optional()
}

pub fn H5VLrequest_optional() -> Result<()> {
    H5VL__request_optional()
}

pub fn H5VLrequest_optional_op() -> Result<()> {
    H5VL__request_optional()
}

pub fn H5VLrequest_free() {}

pub fn H5VLblob_put(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VLblob_get(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
}

pub fn H5VL__blob_specific(object: &VolObject) -> usize {
    H5VL__native_blob_specific(object)
}

pub fn H5VL_blob_specific(object: &VolObject) -> usize {
    H5VL__blob_specific(object)
}

pub fn H5VLblob_specific(object: &VolObject) -> usize {
    H5VL__blob_specific(object)
}

pub fn H5VL__blob_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__blob_optional"))
}

pub fn H5VL_blob_optional() -> Result<()> {
    H5VL__blob_optional()
}

pub fn H5VLblob_optional() -> Result<()> {
    H5VL__blob_optional()
}

pub fn H5VLtoken_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

pub fn H5VLtoken_to_str(token: u64) -> String {
    token.to_string()
}

pub fn H5VL__native_file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

pub fn H5VL__native_file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

pub fn H5VL__native_file_get(object: &VolObject) -> &str {
    &object.name
}

pub fn H5VL__native_file_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_file_specific"))
}

pub fn H5VL__native_file_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_file_optional"))
}

pub fn H5VL__native_file_close(_object: VolObject) {}

pub fn H5VL__reparse_def_vol_conn_variable_test(registry: &mut VolRegistry, id: u64) -> Result<()> {
    H5VL__set_def_conn(registry, id)
}

pub fn H5VL__is_native_connector_test(connector: &VolConnector) -> bool {
    connector.name == "native" || connector.value == 0
}

pub fn H5VL__register_using_vol_id_test(
    registry: &mut VolRegistry,
    id: u64,
    name: &str,
    value: u64,
) -> u64 {
    H5VL__register_connector_by_value(registry, id, name, value)
}

pub fn H5VL_obj_get_rc(registry: &VolRegistry, id: u64) -> Option<usize> {
    registry
        .connectors
        .get(&id)
        .map(|connector| connector.refcount)
}

pub fn H5VL_obj_get_connector<'a>(
    registry: &'a VolRegistry,
    object: &VolObject,
) -> Option<&'a VolConnector> {
    registry.connectors.get(&object.connector_id)
}

pub fn H5VL_obj_get_data(object: &VolObject) -> &[u8] {
    &object.payload
}

pub fn H5VL_obj_reset_data(object: &mut VolObject) {
    object.payload.clear();
}

pub fn H5VL__native_datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_datatype_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_datatype_get(object: &VolObject) -> usize {
    object.payload.len()
}

pub fn H5VL__native_datatype_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_datatype_specific"))
}

pub fn H5VL__native_datatype_close(_object: VolObject) {}

pub fn H5VL__native_introspect_opt_query() -> Result<()> {
    Err(unsupported_vol("H5VL__native_introspect_opt_query"))
}

pub fn H5VL__native_dataset_io_setup(object: &VolObject) -> VolObject {
    object.clone()
}

pub fn H5VL__native_dataset_io_cleanup(_object: VolObject) {}

pub fn H5VL__native_dataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_dataset_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

pub fn H5VL__native_dataset_read(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
}

pub fn H5VL__native_dataset_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

pub fn H5VL__native_dataset_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_dataset_optional"))
}

pub fn H5VL__native_dataset_close(_object: VolObject) {}

pub fn H5VL_conn_inc_rc_public(registry: &mut VolRegistry, id: u64) -> Result<usize> {
    H5VL_conn_inc_rc(registry, id)
}

pub fn H5VL_conn_dec_rc_public(registry: &mut VolRegistry, id: u64) -> Result<usize> {
    H5VL_conn_dec_rc(registry, id)
}

pub fn H5VL_object_wrap_state(object: VolObject) -> (VolObject, bool) {
    let wrapped = H5VL_pass_through_get_wrap_ctx(&object);
    (H5VL__wrap_obj(object), wrapped)
}

pub fn H5VL_object_unwrap_public(object: VolObject) -> VolObject {
    H5VL_object_unwrap(object)
}
