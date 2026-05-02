use std::collections::BTreeMap;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectMessage {
    pub msg_type: u16,
    pub flags: u8,
    pub creation_index: u16,
    pub data: Vec<u8>,
    pub shared: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectHeaderState {
    pub addr: u64,
    pub messages: Vec<ObjectMessage>,
    pub refcount: u32,
    pub comment: Option<String>,
    pub flush_disabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedMessageTable {
    pub refs: BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FsInfoMessage {
    pub version: u8,
    pub free_space_strategy: u8,
    pub persist: bool,
    pub threshold: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolTableMessage {
    pub btree_addr: u64,
    pub heap_addr: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayoutMessage {
    pub version: u8,
    pub raw: Vec<u8>,
}

#[allow(non_snake_case)]
pub fn H5O__shared_link_adj(table: &mut SharedMessageTable, addr: u64, delta: isize) {
    let entry = table.refs.entry(addr).or_default();
    if delta.is_negative() {
        *entry = entry.saturating_sub(delta.unsigned_abs());
    } else {
        *entry = entry.saturating_add(delta as usize);
    }
    if *entry == 0 {
        table.refs.remove(&addr);
    }
}

#[allow(non_snake_case)]
pub fn H5O_set_shared(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

#[allow(non_snake_case)]
pub fn H5O__shared_delete(table: &mut SharedMessageTable, addr: u64) {
    table.refs.remove(&addr);
}

#[allow(non_snake_case)]
pub fn H5O__shared_copy_file(table: &SharedMessageTable) -> SharedMessageTable {
    table.clone()
}

#[allow(non_snake_case)]
pub fn H5O__shared_debug(table: &SharedMessageTable) -> String {
    format!("shared_messages={}", table.refs.len())
}

#[allow(non_snake_case)]
pub fn H5O__group_isa(header: &ObjectHeaderState) -> bool {
    header
        .messages
        .iter()
        .any(|msg| matches!(msg.msg_type, 0x0011 | 0x000A | 0x000B))
}

#[allow(non_snake_case)]
pub fn H5O__group_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__group_bh_info(header: &ObjectHeaderState) -> usize {
    header.messages.len()
}

#[allow(non_snake_case)]
pub fn H5O_msg_append_oh(header: &mut ObjectHeaderState, message: ObjectMessage) {
    H5O__msg_append_real(header, message);
}

#[allow(non_snake_case)]
pub fn H5O__msg_append_real(header: &mut ObjectHeaderState, message: ObjectMessage) {
    header.messages.push(message);
}

#[allow(non_snake_case)]
pub fn H5O__msg_write_real(
    header: &mut ObjectHeaderState,
    index: usize,
    data: Vec<u8>,
) -> Result<()> {
    let msg = header
        .messages
        .get_mut(index)
        .ok_or_else(|| Error::InvalidFormat("object message index out of range".into()))?;
    msg.data = data;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5O_msg_reset(message: &mut ObjectMessage) {
    H5O__msg_reset_real(message);
}

#[allow(non_snake_case)]
pub fn H5O__msg_reset_real(message: &mut ObjectMessage) {
    message.data.clear();
    message.flags = 0;
    message.shared = false;
}

#[allow(non_snake_case)]
pub fn H5O_msg_free(_message: ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__msg_free_mesg(message: &mut ObjectMessage) {
    message.data.clear();
}

#[allow(non_snake_case)]
pub fn H5O_msg_free_real(_message: ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O_msg_copy(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O_msg_exists(header: &ObjectHeaderState, msg_type: u16) -> bool {
    header.messages.iter().any(|msg| msg.msg_type == msg_type)
}

#[allow(non_snake_case)]
pub fn H5O_msg_exists_oh(header: &ObjectHeaderState, msg_type: u16) -> bool {
    H5O_msg_exists(header, msg_type)
}

#[allow(non_snake_case)]
pub fn H5O_msg_remove(header: &mut ObjectHeaderState, msg_type: u16) -> Option<ObjectMessage> {
    H5O__msg_remove_real(header, msg_type)
}

#[allow(non_snake_case)]
pub fn H5O_msg_remove_op(message: &ObjectMessage, msg_type: u16) -> bool {
    message.msg_type == msg_type
}

#[allow(non_snake_case)]
pub fn H5O__msg_remove_cb(message: &ObjectMessage, msg_type: u16) -> bool {
    H5O_msg_remove_op(message, msg_type)
}

#[allow(non_snake_case)]
pub fn H5O__msg_remove_real(
    header: &mut ObjectHeaderState,
    msg_type: u16,
) -> Option<ObjectMessage> {
    let pos = header
        .messages
        .iter()
        .position(|message| message.msg_type == msg_type)?;
    Some(header.messages.remove(pos))
}

#[allow(non_snake_case)]
pub fn H5O_msg_iterate(header: &ObjectHeaderState) -> impl Iterator<Item = &ObjectMessage> {
    H5O__msg_iterate_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__msg_iterate_real(header: &ObjectHeaderState) -> impl Iterator<Item = &ObjectMessage> {
    header.messages.iter()
}

#[allow(non_snake_case)]
pub fn H5O_msg_raw_size(message: &ObjectMessage) -> usize {
    message.data.len()
}

#[allow(non_snake_case)]
pub fn H5O_msg_size_f(message: &ObjectMessage) -> usize {
    H5O_msg_raw_size(message)
}

#[allow(non_snake_case)]
pub fn H5O_msg_size_oh(header: &ObjectHeaderState) -> usize {
    header.messages.iter().map(H5O_msg_raw_size).sum()
}

#[allow(non_snake_case)]
pub fn H5O_msg_can_share(message: &ObjectMessage) -> bool {
    !message.data.is_empty()
}

#[allow(non_snake_case)]
pub fn H5O_msg_can_share_in_ohdr(message: &ObjectMessage) -> bool {
    H5O_msg_can_share(message)
}

#[allow(non_snake_case)]
pub fn H5O_msg_is_shared(message: &ObjectMessage) -> bool {
    message.shared
}

#[allow(non_snake_case)]
pub fn H5O_msg_set_share(message: &mut ObjectMessage) {
    message.shared = true;
}

#[allow(non_snake_case)]
pub fn H5O_msg_reset_share(message: &mut ObjectMessage) {
    message.shared = false;
}

#[allow(non_snake_case)]
pub fn H5O_msg_get_crt_index(message: &ObjectMessage) -> u16 {
    message.creation_index
}

#[allow(non_snake_case)]
pub fn H5O_msg_encode(message: &ObjectMessage) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + message.data.len());
    out.extend_from_slice(&message.msg_type.to_le_bytes());
    out.push(message.flags);
    out.extend_from_slice(&message.creation_index.to_le_bytes());
    out.extend_from_slice(&message.data);
    out
}

#[allow(non_snake_case)]
pub fn H5O_msg_decode(bytes: &[u8]) -> Result<ObjectMessage> {
    if bytes.len() < 5 {
        return Err(Error::InvalidFormat(
            "object message image is too short".into(),
        ));
    }
    Ok(ObjectMessage {
        msg_type: u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
        flags: bytes[2],
        creation_index: u16::from_le_bytes(bytes[3..5].try_into().unwrap()),
        data: bytes[5..].to_vec(),
        shared: false,
    })
}

#[allow(non_snake_case)]
pub fn H5O__msg_copy_file(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__msg_alloc(msg_type: u16, data: Vec<u8>) -> ObjectMessage {
    ObjectMessage {
        msg_type,
        data,
        ..ObjectMessage::default()
    }
}

#[allow(non_snake_case)]
pub fn H5O__copy_mesg(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O_msg_delete(message: &mut ObjectMessage) {
    message.data.clear();
}

#[allow(non_snake_case)]
pub fn H5O_msg_flush(_message: &ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__flush_msgs(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_msg_get_flags(message: &ObjectMessage) -> u8 {
    message.flags
}

#[allow(non_snake_case)]
pub fn H5O__cache_verify_chksum(image: &[u8], checksum: u32) -> bool {
    image
        .iter()
        .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
        == checksum
}

#[allow(non_snake_case)]
pub fn H5O__cache_serialize(header: &ObjectHeaderState) -> Vec<u8> {
    header.messages.iter().flat_map(H5O_msg_encode).collect()
}

#[allow(non_snake_case)]
pub fn H5O__cache_notify(_header: &ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__cache_free_icr(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_get_initial_load_size(size: usize) -> usize {
    size.min(512)
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_verify_chksum(image: &[u8], checksum: u32) -> bool {
    H5O__cache_verify_chksum(image, checksum)
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_deserialize(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_image_len(image: &[u8]) -> usize {
    image.len()
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_serialize(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_notify(_image: &[u8]) {}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_free_icr(_image: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__add_cont_msg(header: &mut ObjectHeaderState, addr: u64, size: u64) {
    let mut data = Vec::with_capacity(16);
    data.extend_from_slice(&addr.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    H5O_msg_append_oh(header, H5O__msg_alloc(0x0010, data));
}

#[allow(non_snake_case)]
pub fn H5O__prefix_deserialize(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__chunk_deserialize(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__bogus_encode(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__bogus_size(data: &[u8]) -> usize {
    data.len()
}

#[allow(non_snake_case)]
pub fn H5O__bogus_debug(data: &[u8]) -> String {
    format!("bogus({} bytes)", data.len())
}

#[allow(non_snake_case)]
pub fn H5O__layout_decode(bytes: &[u8]) -> LayoutMessage {
    LayoutMessage {
        version: bytes.first().copied().unwrap_or_default(),
        raw: bytes.to_vec(),
    }
}

#[allow(non_snake_case)]
pub fn H5O__layout_encode(layout: &LayoutMessage) -> Vec<u8> {
    layout.raw.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_copy(layout: &LayoutMessage) -> LayoutMessage {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_size(layout: &LayoutMessage) -> usize {
    layout.raw.len()
}

#[allow(non_snake_case)]
pub fn H5O__layout_reset(layout: &mut LayoutMessage) {
    layout.raw.clear();
}

#[allow(non_snake_case)]
pub fn H5O__layout_free(_layout: LayoutMessage) {}

#[allow(non_snake_case)]
pub fn H5O__layout_delete(layout: &mut LayoutMessage) {
    layout.raw.clear();
}

#[allow(non_snake_case)]
pub fn H5O__layout_pre_copy_file(layout: &LayoutMessage) -> LayoutMessage {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_copy_file(layout: &LayoutMessage) -> LayoutMessage {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_debug(layout: &LayoutMessage) -> String {
    format!(
        "layout(version={}, bytes={})",
        layout.version,
        layout.raw.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O__refcount_encode(refcount: u32) -> Vec<u8> {
    refcount.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__refcount_copy(refcount: u32) -> u32 {
    refcount
}

#[allow(non_snake_case)]
pub fn H5O__refcount_size(_refcount: u32) -> usize {
    4
}

#[allow(non_snake_case)]
pub fn H5O__refcount_free(_refcount: u32) {}

#[allow(non_snake_case)]
pub fn H5O__refcount_pre_copy_file(refcount: u32) -> u32 {
    refcount
}

#[allow(non_snake_case)]
pub fn H5O__refcount_debug(refcount: u32) -> String {
    format!("refcount={refcount}")
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_decode(bytes: &[u8]) -> FsInfoMessage {
    FsInfoMessage {
        version: bytes.first().copied().unwrap_or_default(),
        free_space_strategy: bytes.get(1).copied().unwrap_or_default(),
        persist: bytes.get(2).copied().unwrap_or_default() != 0,
        threshold: bytes
            .get(3..11)
            .and_then(|raw| raw.try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or_default(),
    }
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_encode(info: &FsInfoMessage) -> Vec<u8> {
    let mut out = vec![
        info.version,
        info.free_space_strategy,
        u8::from(info.persist),
    ];
    out.extend_from_slice(&info.threshold.to_le_bytes());
    out
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_copy(info: &FsInfoMessage) -> FsInfoMessage {
    info.clone()
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_size(_info: &FsInfoMessage) -> usize {
    11
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_free(_info: FsInfoMessage) {}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_debug(info: &FsInfoMessage) -> String {
    format!(
        "fsinfo(version={}, threshold={})",
        info.version, info.threshold
    )
}

#[allow(non_snake_case)]
pub fn H5O_fsinfo_set_version(info: &mut FsInfoMessage, version: u8) {
    info.version = version;
}

#[allow(non_snake_case)]
pub fn H5O_fsinfo_check_version(info: &FsInfoMessage) -> bool {
    info.version <= 1
}

#[allow(non_snake_case)]
pub fn H5O__stab_encode(stab: &SymbolTableMessage) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&stab.btree_addr.to_le_bytes());
    out.extend_from_slice(&stab.heap_addr.to_le_bytes());
    out
}

#[allow(non_snake_case)]
pub fn H5O__stab_copy(stab: &SymbolTableMessage) -> SymbolTableMessage {
    stab.clone()
}

#[allow(non_snake_case)]
pub fn H5O__stab_size(_stab: &SymbolTableMessage) -> usize {
    16
}

#[allow(non_snake_case)]
pub fn H5O__stab_free(_stab: SymbolTableMessage) {}

#[allow(non_snake_case)]
pub fn H5O__stab_delete(stab: &mut SymbolTableMessage) {
    *stab = SymbolTableMessage::default();
}

#[allow(non_snake_case)]
pub fn H5O__stab_copy_file(stab: &SymbolTableMessage) -> SymbolTableMessage {
    stab.clone()
}

#[allow(non_snake_case)]
pub fn H5O__stab_debug(stab: &SymbolTableMessage) -> String {
    format!("stab(btree={}, heap={})", stab.btree_addr, stab.heap_addr)
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_decode(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|raw| u64::from_le_bytes(raw.try_into().unwrap()))
        .collect()
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_copy(space: &[u64]) -> Vec<u64> {
    space.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_reset(space: &mut Vec<u64>) {
    space.clear();
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_free(_space: Vec<u64>) {}

#[allow(non_snake_case)]
pub fn H5O__sdspace_pre_copy_file(space: &[u64]) -> Vec<u64> {
    space.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_debug(space: &[u64]) -> String {
    format!("sdspace{:?}", space)
}

#[allow(non_snake_case)]
pub fn H5Olink(header: &mut ObjectHeaderState, delta: i32) {
    if delta.is_negative() {
        header.refcount = header.refcount.saturating_sub(delta.unsigned_abs());
    } else {
        header.refcount = header.refcount.saturating_add(delta as u32);
    }
}

#[allow(non_snake_case)]
pub fn H5Oincr_refcount(header: &mut ObjectHeaderState) {
    H5Olink(header, 1);
}

#[allow(non_snake_case)]
pub fn H5Odecr_refcount(header: &mut ObjectHeaderState) {
    H5Olink(header, -1);
}

#[allow(non_snake_case)]
pub fn H5Oexists_by_name(objects: &BTreeMap<String, ObjectHeaderState>, name: &str) -> bool {
    objects.contains_key(name)
}

#[allow(non_snake_case)]
pub fn H5Oset_comment(header: &mut ObjectHeaderState, comment: impl Into<String>) {
    header.comment = Some(comment.into());
}

#[allow(non_snake_case)]
pub fn H5Oset_comment_by_name(
    objects: &mut BTreeMap<String, ObjectHeaderState>,
    name: &str,
    comment: impl Into<String>,
) -> Result<()> {
    let header = objects
        .get_mut(name)
        .ok_or_else(|| Error::InvalidFormat(format!("object '{name}' not found")))?;
    H5Oset_comment(header, comment);
    Ok(())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectLocation {
    pub file_name: Option<String>,
    pub addr: u64,
    pub held: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectInfo {
    pub addr: u64,
    pub refcount: u32,
    pub msg_count: usize,
    pub has_checksum: bool,
}

fn bytes_decode(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

fn bytes_encode(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

fn bytes_size(bytes: &[u8]) -> usize {
    bytes.len()
}

fn bytes_debug(label: &str, bytes: &[u8]) -> String {
    format!("{label}({} bytes)", bytes.len())
}

#[allow(non_snake_case)]
pub fn H5Ovisit3(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    objects.keys().cloned().collect()
}

#[allow(non_snake_case)]
pub fn H5O__are_mdc_flushes_disabled(header: &ObjectHeaderState) -> bool {
    header.flush_disabled
}

#[allow(non_snake_case)]
pub fn H5Oare_mdc_flushes_disabled(header: &ObjectHeaderState) -> bool {
    H5O__are_mdc_flushes_disabled(header)
}

#[allow(non_snake_case)]
pub fn H5Otoken_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

#[allow(non_snake_case)]
pub fn H5Otoken_to_str(token: u64) -> String {
    format!("{token:#x}")
}

#[allow(non_snake_case)]
pub fn H5Otoken_from_str(token: &str) -> Result<u64> {
    let trimmed = token.strip_prefix("0x").unwrap_or(token);
    u64::from_str_radix(trimmed, 16)
        .map_err(|_| Error::InvalidFormat("invalid object token".into()))
}

#[allow(non_snake_case)]
pub fn H5O__print_time_field(timestamp: u64) -> String {
    timestamp.to_string()
}

#[allow(non_snake_case)]
pub fn H5O__assert(condition: bool) -> Result<()> {
    condition
        .then_some(())
        .ok_or_else(|| Error::InvalidFormat("object assertion failed".into()))
}

#[allow(non_snake_case)]
pub fn H5O_debug_id(addr: u64) -> String {
    format!("object@{addr:#x}")
}

#[allow(non_snake_case)]
pub fn H5O__debug_real(header: &ObjectHeaderState) -> String {
    format!(
        "object(addr={:#x}, messages={})",
        header.addr,
        header.messages.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O_debug(header: &ObjectHeaderState) -> String {
    H5O__debug_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__mdci_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__mdci_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__mdci_size(bytes: &[u8]) -> usize {
    bytes_size(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__mdci_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__mdci_delete(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__mdci_debug(bytes: &[u8]) -> String {
    bytes_debug("mdci", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__attr_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__attr_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__attr_size(bytes: &[u8]) -> usize {
    bytes_size(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__attr_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__attr_pre_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__attr_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__attr_post_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__attr_debug(bytes: &[u8]) -> String {
    bytes_debug("attr", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__chunk_add(header: &mut ObjectHeaderState, message: ObjectMessage) {
    H5O_msg_append_oh(header, message);
}

#[allow(non_snake_case)]
pub fn H5O__chunk_unprotect(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__chunk_update_idx(header: &mut ObjectHeaderState) {
    for (idx, msg) in header.messages.iter_mut().enumerate() {
        msg.creation_index = idx.min(u16::MAX as usize) as u16;
    }
}

#[allow(non_snake_case)]
pub fn H5O__add_gap(header: &mut ObjectHeaderState, size: usize) {
    H5O_msg_append_oh(header, H5O__msg_alloc(0, vec![0; size]));
}

#[allow(non_snake_case)]
pub fn H5O__eliminate_gap(header: &mut ObjectHeaderState) {
    header
        .messages
        .retain(|msg| msg.msg_type != 0 || msg.data.iter().any(|b| *b != 0));
}

#[allow(non_snake_case)]
pub fn H5O__alloc_null(size: usize) -> ObjectMessage {
    H5O__msg_alloc(0, vec![0; size])
}

#[allow(non_snake_case)]
pub fn H5O__alloc_msgs(header: &mut ObjectHeaderState, count: usize) {
    header.messages.reserve(count);
}

#[allow(non_snake_case)]
pub fn H5O__alloc_extend_chunk(header: &mut ObjectHeaderState, size: usize) {
    H5O__add_gap(header, size);
}

#[allow(non_snake_case)]
pub fn H5O__alloc_new_chunk(size: usize) -> Vec<u8> {
    vec![0; size]
}

#[allow(non_snake_case)]
pub fn H5O__alloc_find_best_null(header: &ObjectHeaderState) -> Option<usize> {
    header.messages.iter().position(|msg| msg.msg_type == 0)
}

#[allow(non_snake_case)]
pub fn H5O__alloc(header: &mut ObjectHeaderState, message: ObjectMessage) {
    H5O_msg_append_oh(header, message);
}

#[allow(non_snake_case)]
pub fn H5O__release_mesg(_message: &mut ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__move_cont(header: &mut ObjectHeaderState, from: usize, to: usize) -> Result<()> {
    if from >= header.messages.len() || to > header.messages.len() {
        return Err(Error::InvalidFormat(
            "object message move index out of range".into(),
        ));
    }
    let msg = header.messages.remove(from);
    header.messages.insert(to.min(header.messages.len()), msg);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5O__move_msgs_forward(header: &mut ObjectHeaderState) {
    header.messages.sort_by_key(|msg| msg.creation_index);
}

#[allow(non_snake_case)]
pub fn H5O__merge_null(header: &mut ObjectHeaderState) {
    let total: usize = header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0)
        .map(|msg| msg.data.len())
        .sum();
    header.messages.retain(|msg| msg.msg_type != 0);
    if total > 0 {
        header.messages.push(H5O__alloc_null(total));
    }
}

#[allow(non_snake_case)]
pub fn H5O__remove_empty_chunks(header: &mut ObjectHeaderState) {
    header
        .messages
        .retain(|msg| !msg.data.is_empty() || msg.msg_type == 0);
}

#[allow(non_snake_case)]
pub fn H5O__condense_header(header: &mut ObjectHeaderState) {
    H5O__merge_null(header);
    H5O__remove_empty_chunks(header);
}

#[allow(non_snake_case)]
pub fn H5O__alloc_shrink_chunk(header: &mut ObjectHeaderState) {
    H5O__condense_header(header);
}

#[allow(non_snake_case)]
pub fn H5O__mtime_new_decode(bytes: &[u8]) -> u64 {
    H5O__mtime_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__mtime_decode(bytes: &[u8]) -> u64 {
    bytes
        .get(..8)
        .and_then(|raw| raw.try_into().ok())
        .map(u64::from_le_bytes)
        .unwrap_or_default()
}

#[allow(non_snake_case)]
pub fn H5O__mtime_new_encode(timestamp: u64) -> Vec<u8> {
    H5O__mtime_encode(timestamp)
}

#[allow(non_snake_case)]
pub fn H5O__mtime_encode(timestamp: u64) -> Vec<u8> {
    timestamp.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__mtime_copy(timestamp: u64) -> u64 {
    timestamp
}

#[allow(non_snake_case)]
pub fn H5O__mtime_new_size(_timestamp: u64) -> usize {
    8
}

#[allow(non_snake_case)]
pub fn H5O__mtime_size(_timestamp: u64) -> usize {
    8
}

#[allow(non_snake_case)]
pub fn H5O__mtime_free(_timestamp: u64) {}

#[allow(non_snake_case)]
pub fn H5O__mtime_debug(timestamp: u64) -> String {
    format!("mtime={timestamp}")
}

#[allow(non_snake_case)]
pub fn H5O__copy_header_real(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_free_addrmap_cb(_addr: u64) {}

#[allow(non_snake_case)]
pub fn H5O__copy_header(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_obj(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_free_comm_dt_cb(_addr: u64) {}

#[allow(non_snake_case)]
pub fn H5O__copy_comm_dt_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_attr_cb(_message: &ObjectMessage) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_check(_header: &ObjectHeaderState) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_cb(_header: &ObjectHeaderState) -> Option<u64> {
    None
}

#[allow(non_snake_case)]
pub fn H5O__copy_insert_comm_dt(_addr: u64) {}

#[allow(non_snake_case)]
pub fn H5O_flush(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_flush_common(header: &mut ObjectHeaderState) {
    H5O_flush(header);
}

#[allow(non_snake_case)]
pub fn H5O__oh_tag(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O_refresh_metadata(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__refresh_metadata_close(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_refresh_metadata_reopen(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_debug(bytes: &[u8]) -> String {
    bytes_debug("shmesg", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__pline_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__pline_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__pline_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__pline_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__pline_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__pline_pre_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__pline_debug(bytes: &[u8]) -> String {
    bytes_debug("pline", bytes)
}

#[allow(non_snake_case)]
pub fn H5O_pline_set_version(bytes: &mut Vec<u8>, version: u8) {
    if bytes.is_empty() {
        bytes.push(version);
    } else {
        bytes[0] = version;
    }
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_debug(bytes: &[u8]) -> String {
    bytes_debug("drvinfo", bytes)
}

#[allow(non_snake_case)]
pub fn H5O_init() -> bool {
    H5O__init_package()
}

#[allow(non_snake_case)]
pub fn H5O__init_package() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5O__set_version(layout: &mut LayoutMessage, version: u8) {
    layout.version = version;
}

#[allow(non_snake_case)]
pub fn H5O_create_ohdr(addr: u64) -> ObjectHeaderState {
    ObjectHeaderState {
        addr,
        refcount: 1,
        ..ObjectHeaderState::default()
    }
}

#[allow(non_snake_case)]
pub fn H5O_apply_ohdr(header: &mut ObjectHeaderState, f: impl FnOnce(&mut ObjectHeaderState)) {
    f(header);
}

#[allow(non_snake_case)]
pub fn H5O_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O_open_name(
    objects: &BTreeMap<String, ObjectHeaderState>,
    name: &str,
) -> Option<ObjectHeaderState> {
    objects.get(name).cloned()
}

#[allow(non_snake_case)]
pub fn H5O__open_by_idx(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectHeaderState> {
    objects.values().nth(idx).cloned()
}

#[allow(non_snake_case)]
pub fn H5O__open_by_addr(
    objects: &BTreeMap<String, ObjectHeaderState>,
    addr: u64,
) -> Option<ObjectHeaderState> {
    objects.values().find(|header| header.addr == addr).cloned()
}

#[allow(non_snake_case)]
pub fn H5O_open_by_loc(
    location: &ObjectLocation,
    objects: &BTreeMap<String, ObjectHeaderState>,
) -> Option<ObjectHeaderState> {
    H5O__open_by_addr(objects, location.addr)
}

#[allow(non_snake_case)]
pub fn H5O_close(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__link_oh(header: &mut ObjectHeaderState, delta: i32) {
    H5Olink(header, delta);
}

#[allow(non_snake_case)]
pub fn H5O_link(header: &mut ObjectHeaderState, delta: i32) {
    H5Olink(header, delta);
}

#[allow(non_snake_case)]
pub fn H5O_pin(location: &mut ObjectLocation) {
    location.held = true;
}

#[allow(non_snake_case)]
pub fn H5O_unpin(location: &mut ObjectLocation) {
    location.held = false;
}

#[allow(non_snake_case)]
pub fn H5O_unprotect(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_touch_oh(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_touch(header: &mut ObjectHeaderState) {
    H5O_touch_oh(header);
}

#[allow(non_snake_case)]
pub fn H5O_bogus_oh(header: &ObjectHeaderState) -> bool {
    header.addr == u64::MAX
}

#[allow(non_snake_case)]
pub fn H5O_delete(header: &mut ObjectHeaderState) {
    H5O__delete_oh(header);
}

#[allow(non_snake_case)]
pub fn H5O__delete_oh(header: &mut ObjectHeaderState) {
    header.messages.clear();
    header.refcount = 0;
}

#[allow(non_snake_case)]
pub fn H5O_obj_type(header: &ObjectHeaderState) -> &'static str {
    H5O__obj_type_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__obj_type_real(header: &ObjectHeaderState) -> &'static str {
    if H5O__group_isa(header) {
        "group"
    } else if header.messages.iter().any(|msg| msg.msg_type == 0x0001) {
        "dataset"
    } else {
        "unknown"
    }
}

#[allow(non_snake_case)]
pub fn H5O__obj_class(header: &ObjectHeaderState) -> &'static str {
    H5O_obj_type(header)
}

#[allow(non_snake_case)]
pub fn H5O_get_loc(header: &ObjectHeaderState) -> ObjectLocation {
    ObjectLocation {
        addr: header.addr,
        ..ObjectLocation::default()
    }
}

#[allow(non_snake_case)]
pub fn H5O_loc_reset(location: &mut ObjectLocation) {
    *location = ObjectLocation::default();
}

#[allow(non_snake_case)]
pub fn H5O_loc_copy(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

#[allow(non_snake_case)]
pub fn H5O_loc_copy_shallow(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

#[allow(non_snake_case)]
pub fn H5O_loc_copy_deep(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

#[allow(non_snake_case)]
pub fn H5O_loc_hold_file(location: &mut ObjectLocation) {
    location.held = true;
}

#[allow(non_snake_case)]
pub fn H5O_loc_free(_location: ObjectLocation) {}

#[allow(non_snake_case)]
pub fn H5O_get_hdr_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O__get_hdr_info_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__get_hdr_info_real(header: &ObjectHeaderState) -> ObjectInfo {
    ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count: header.messages.len(),
        has_checksum: H5O_has_chksum(header),
    }
}

#[allow(non_snake_case)]
pub fn H5O_get_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_hdr_info(header)
}

#[allow(non_snake_case)]
pub fn H5O_get_native_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_hdr_info(header)
}

#[allow(non_snake_case)]
pub fn H5O_get_create_plist(_header: &ObjectHeaderState) -> BTreeMap<String, String> {
    BTreeMap::new()
}

#[allow(non_snake_case)]
pub fn H5O_get_nlinks(header: &ObjectHeaderState) -> u32 {
    header.refcount
}

#[allow(non_snake_case)]
pub fn H5O_obj_create(addr: u64) -> ObjectHeaderState {
    H5O_create_ohdr(addr)
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_addr(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_flags(header: &ObjectHeaderState) -> u8 {
    header.messages.iter().fold(0, |acc, msg| acc | msg.flags)
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_mtime(_header: &ObjectHeaderState) -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_version(_header: &ObjectHeaderState) -> u8 {
    2
}

#[allow(non_snake_case)]
pub fn H5O_get_rc_and_type(header: &ObjectHeaderState) -> (u32, &'static str) {
    (header.refcount, H5O_obj_type(header))
}

#[allow(non_snake_case)]
pub fn H5O__visit_cb(name: &str, _header: &ObjectHeaderState) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5O__visit(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    H5Ovisit3(objects)
}

#[allow(non_snake_case)]
pub fn H5O__inc_rc(header: &mut ObjectHeaderState) {
    H5Oincr_refcount(header);
}

#[allow(non_snake_case)]
pub fn H5O__dec_rc(header: &mut ObjectHeaderState) {
    H5Odecr_refcount(header);
}

#[allow(non_snake_case)]
pub fn H5O_get_proxy(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__free(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__reset_info2(info: &mut ObjectInfo) {
    *info = ObjectInfo::default();
}

#[allow(non_snake_case)]
pub fn H5O_has_chksum(header: &ObjectHeaderState) -> bool {
    !header.messages.is_empty()
}

#[allow(non_snake_case)]
pub fn H5O_get_version_bound(_header: &ObjectHeaderState) -> (u8, u8) {
    (0, 4)
}

#[allow(non_snake_case)]
pub fn H5O__copy_obj_by_ref(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_object1(token: u64) -> u64 {
    token
}

#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_region1(region: &[u8]) -> Vec<u8> {
    region.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_object2(token: u64) -> u64 {
    token
}

#[allow(non_snake_case)]
pub fn H5O_copy_expand_ref(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__cont_decode(bytes: &[u8]) -> (u64, u64) {
    let addr = bytes
        .get(0..8)
        .and_then(|raw| raw.try_into().ok())
        .map(u64::from_le_bytes)
        .unwrap_or_default();
    let size = bytes
        .get(8..16)
        .and_then(|raw| raw.try_into().ok())
        .map(u64::from_le_bytes)
        .unwrap_or_default();
    (addr, size)
}

#[allow(non_snake_case)]
pub fn H5O__cont_encode(addr: u64, size: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&addr.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out
}

#[allow(non_snake_case)]
pub fn H5O__cont_size(_addr: u64, _size: u64) -> usize {
    16
}

#[allow(non_snake_case)]
pub fn H5O__cont_free(_cont: (u64, u64)) {}

#[allow(non_snake_case)]
pub fn H5O__cont_delete(cont: &mut (u64, u64)) {
    *cont = (0, 0);
}

#[allow(non_snake_case)]
pub fn H5O__cont_debug(cont: (u64, u64)) -> String {
    format!("cont(addr={}, size={})", cont.0, cont.1)
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__ginfo_debug(bytes: &[u8]) -> String {
    bytes_debug("ginfo", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__attr_create(header: &mut ObjectHeaderState, name: &str, value: &[u8]) {
    let mut data = name.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(value);
    H5O_msg_append_oh(header, H5O__msg_alloc(0x000c, data));
}

#[allow(non_snake_case)]
pub fn H5O__attr_open_by_name(header: &ObjectHeaderState, name: &str) -> Option<ObjectMessage> {
    let prefix = name.as_bytes();
    header
        .messages
        .iter()
        .find(|msg| msg.msg_type == 0x000c && msg.data.starts_with(prefix))
        .cloned()
}

#[allow(non_snake_case)]
pub fn H5O__attr_open_by_idx_cb(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__attr_open_by_idx(header: &ObjectHeaderState, index: usize) -> Option<ObjectMessage> {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c)
        .nth(index)
        .cloned()
}

#[allow(non_snake_case)]
pub fn H5O__attr_find_opened_attr(header: &ObjectHeaderState, name: &str) -> bool {
    H5O__attr_open_by_name(header, name).is_some()
}

#[allow(non_snake_case)]
pub fn H5O__attr_update_shared(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

#[allow(non_snake_case)]
pub fn H5O__attr_write_cb(message: &mut ObjectMessage, data: &[u8]) {
    message.data.clear();
    message.data.extend_from_slice(data);
}

#[allow(non_snake_case)]
pub fn H5O__attr_write(header: &mut ObjectHeaderState, name: &str, value: &[u8]) {
    if let Some(pos) = header
        .messages
        .iter()
        .position(|msg| msg.msg_type == 0x000c && msg.data.starts_with(name.as_bytes()))
    {
        H5O__attr_write_cb(&mut header.messages[pos], value);
    } else {
        H5O__attr_create(header, name, value);
    }
}

#[allow(non_snake_case)]
pub fn H5O__attr_rename(header: &mut ObjectHeaderState, old_name: &str, new_name: &str) -> bool {
    if let Some(msg) = header
        .messages
        .iter_mut()
        .find(|msg| msg.msg_type == 0x000c && msg.data.starts_with(old_name.as_bytes()))
    {
        let value = msg
            .data
            .splitn(2, |byte| *byte == 0)
            .nth(1)
            .unwrap_or(&[])
            .to_vec();
        msg.data = new_name.as_bytes().to_vec();
        msg.data.push(0);
        msg.data.extend_from_slice(&value);
        true
    } else {
        false
    }
}

#[allow(non_snake_case)]
pub fn H5O_attr_iterate_real(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c)
        .cloned()
        .collect()
}

#[allow(non_snake_case)]
pub fn H5O__attr_iterate(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    H5O_attr_iterate_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__attr_remove_update(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__attr_remove(header: &mut ObjectHeaderState, name: &str) -> bool {
    if let Some(pos) = header
        .messages
        .iter()
        .position(|msg| msg.msg_type == 0x000c && msg.data.starts_with(name.as_bytes()))
    {
        header.messages.remove(pos);
        true
    } else {
        false
    }
}

#[allow(non_snake_case)]
pub fn H5O__attr_remove_by_idx(
    header: &mut ObjectHeaderState,
    index: usize,
) -> Option<ObjectMessage> {
    let pos = header
        .messages
        .iter()
        .enumerate()
        .filter(|(_, msg)| msg.msg_type == 0x000c)
        .nth(index)
        .map(|(pos, _)| pos)?;
    Some(header.messages.remove(pos))
}

#[allow(non_snake_case)]
pub fn H5O__attr_count_real(header: &ObjectHeaderState) -> usize {
    H5O_attr_iterate_real(header).len()
}

#[allow(non_snake_case)]
pub fn H5O__attr_exists(header: &ObjectHeaderState, name: &str) -> bool {
    H5O__attr_find_opened_attr(header, name)
}

#[allow(non_snake_case)]
pub fn H5O__attr_bh_info(header: &ObjectHeaderState) -> usize {
    H5O__attr_count_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__fill_new_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__fill_old_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__fill_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__fill_new_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__fill_old_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O_fill_reset_dyn(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__fill_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__fill_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__fill_pre_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O_fill_set_version(bytes: &mut Vec<u8>, version: u8) {
    if bytes.is_empty() {
        bytes.push(version);
    } else {
        bytes[0] = version;
    }
}

#[allow(non_snake_case)]
pub fn H5O__reset_info1(info: &mut ObjectInfo) {
    *info = ObjectInfo::default();
}

#[allow(non_snake_case)]
pub fn H5O__iterate1_adapter(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    header.messages.clone()
}

#[allow(non_snake_case)]
pub fn H5O__get_info_old(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_info(header)
}

#[allow(non_snake_case)]
pub fn H5Oopen_by_addr(
    objects: &BTreeMap<String, ObjectHeaderState>,
    addr: u64,
) -> Option<ObjectHeaderState> {
    H5O__open_by_addr(objects, addr)
}

#[allow(non_snake_case)]
pub fn H5Ovisit1(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    H5Ovisit3(objects)
}

#[allow(non_snake_case)]
pub fn H5Ovisit_by_name2(
    objects: &BTreeMap<String, ObjectHeaderState>,
    prefix: &str,
) -> Vec<String> {
    objects
        .keys()
        .filter(|name| name.starts_with(prefix))
        .cloned()
        .collect()
}

#[allow(non_snake_case)]
pub fn H5O__btreek_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__btreek_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__btreek_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__btreek_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__btreek_debug(bytes: &[u8]) -> String {
    bytes_debug("btreek", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__unknown_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__link_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__link_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__link_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__link_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__link_post_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__link_debug(bytes: &[u8]) -> String {
    bytes_debug("link", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__linfo_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__linfo_delete(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__linfo_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_post_copy_file_cb(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_post_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_debug(bytes: &[u8]) -> String {
    bytes_debug("linfo", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__efl_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__efl_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__efl_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__efl_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__efl_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O_efl_total_size(files: &[Vec<u8>]) -> usize {
    files.iter().map(Vec::len).sum()
}

#[allow(non_snake_case)]
pub fn H5O__efl_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__efl_debug(bytes: &[u8]) -> String {
    bytes_debug("efl", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__ainfo_delete(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_pre_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_post_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_debug(bytes: &[u8]) -> String {
    bytes_debug("ainfo", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__dset_get_copy_file_udata(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dset_free_copy_file_udata(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__dset_isa(header: &ObjectHeaderState) -> bool {
    header
        .messages
        .iter()
        .any(|msg| msg.msg_type == 0x0001 || msg.msg_type == 0x0003)
}

#[allow(non_snake_case)]
pub fn H5O__dset_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dset_create(addr: u64) -> ObjectHeaderState {
    H5O_create_ohdr(addr)
}

#[allow(non_snake_case)]
pub fn H5O__dset_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__dset_bh_info(header: &ObjectHeaderState) -> usize {
    header.messages.len()
}

#[allow(non_snake_case)]
pub fn H5O__dset_flush(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__dtype_isa(header: &ObjectHeaderState) -> bool {
    header.messages.iter().any(|msg| msg.msg_type == 0x0003)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_create(addr: u64) -> ObjectHeaderState {
    H5O_create_ohdr(addr)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__is_attr_dense_test(header: &ObjectHeaderState) -> bool {
    H5O__attr_count_real(header) > 8
}

#[allow(non_snake_case)]
pub fn H5O__is_attr_empty_test(header: &ObjectHeaderState) -> bool {
    H5O__attr_count_real(header) == 0
}

#[allow(non_snake_case)]
pub fn H5O__num_attrs_test(header: &ObjectHeaderState) -> usize {
    H5O__attr_count_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__attr_dense_info_test(header: &ObjectHeaderState) -> usize {
    H5O__attr_count_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__check_msg_marked_test(message: &ObjectMessage) -> bool {
    message.flags != 0
}

#[allow(non_snake_case)]
pub fn H5O__expunge_chunks_test(header: &mut ObjectHeaderState) {
    header.messages.clear();
}

#[allow(non_snake_case)]
pub fn H5O__get_rc_test(header: &ObjectHeaderState) -> u32 {
    header.refcount
}

#[allow(non_snake_case)]
pub fn H5O__msg_get_chunkno_test(_message: &ObjectMessage) -> usize {
    0
}

#[allow(non_snake_case)]
pub fn H5O__msg_move_to_new_chunk_test(header: &mut ObjectHeaderState, idx: usize) -> Result<()> {
    H5O__move_cont(header, idx, header.messages.len())
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_DECODE(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_ENCODE(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_SIZE(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_DELETE(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_LINK(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_COPY_FILE(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_POST_COPY_FILE(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_DEBUG(bytes: &[u8]) -> String {
    bytes_debug("shared", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_decode_helper(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_encode_helper(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__dtype_can_share(bytes: &[u8]) -> bool {
    !bytes.is_empty()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_pre_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_copy_file(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_debug(bytes: &[u8]) -> String {
    bytes_debug("dtype", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__name_decode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_end_matches('\0')
        .to_string()
}

#[allow(non_snake_case)]
pub fn H5O__name_encode(name: &str) -> Vec<u8> {
    name.as_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__name_copy(name: &str) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5O__name_size(name: &str) -> usize {
    name.len()
}

#[allow(non_snake_case)]
pub fn H5O__name_reset(name: &mut String) {
    name.clear();
}

#[allow(non_snake_case)]
pub fn H5O__name_debug(name: &str) -> String {
    format!("name={name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_messages_roundtrip_and_remove() {
        let msg = H5O__msg_alloc(42, b"abc".to_vec());
        let decoded = H5O_msg_decode(&H5O_msg_encode(&msg)).unwrap();
        assert_eq!(decoded.msg_type, 42);
        assert_eq!(decoded.data, b"abc");

        let mut header = ObjectHeaderState::default();
        H5O_msg_append_oh(&mut header, decoded);
        assert!(H5O_msg_exists(&header, 42));
        assert_eq!(H5O_msg_remove(&mut header, 42).unwrap().data, b"abc");
    }
}
