use std::collections::BTreeMap;

/// Shared object-header message payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessage {
    pub msg_type: u8,
    pub heap_addr: u64,
    pub data: Vec<u8>,
    pub refcount: u32,
}

/// Shared-message table/list state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedMessageStore {
    messages: BTreeMap<u64, SharedMessage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessageInfo {
    pub count: usize,
    pub total_bytes: usize,
}

impl SharedMessage {
    pub fn new(msg_type: u8, heap_addr: u64, data: Vec<u8>) -> Self {
        Self {
            msg_type,
            heap_addr,
            data,
            refcount: 1,
        }
    }

    pub fn encoded_len(&self) -> usize {
        1 + 8 + 4 + self.data.len()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.push(self.msg_type);
        out.extend_from_slice(&self.heap_addr.to_le_bytes());
        out.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }
}

impl SharedMessageStore {
    pub fn cache_table_get_initial_load_size(count: usize) -> usize {
        count.saturating_mul(16)
    }

    pub fn cache_table_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
        crc32fast::hash(bytes) == checksum
    }

    pub fn cache_table_image_len(&self) -> usize {
        self.messages.values().map(SharedMessage::encoded_len).sum()
    }

    pub fn cache_table_serialize(&self) -> Vec<u8> {
        self.messages
            .values()
            .flat_map(SharedMessage::encode)
            .collect()
    }

    pub fn cache_table_free_icr(_bytes: Vec<u8>) {}

    pub fn cache_list_get_initial_load_size(count: usize) -> usize {
        count.saturating_mul(12)
    }

    pub fn cache_list_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
        Self::cache_table_verify_chksum(bytes, checksum)
    }

    pub fn cache_list_deserialize(bytes: &[u8]) -> Vec<u8> {
        bytes.to_vec()
    }

    pub fn cache_list_image_len(&self) -> usize {
        self.cache_table_image_len()
    }

    pub fn cache_list_serialize(&self) -> Vec<u8> {
        self.cache_table_serialize()
    }

    pub fn cache_list_free_icr(_bytes: Vec<u8>) {}

    pub fn get_mesg_count_test(&self) -> usize {
        self.messages.len()
    }

    pub fn init() -> Self {
        Self::default()
    }

    pub fn type_to_flag(msg_type: u8) -> u32 {
        1u32.checked_shl(msg_type as u32).unwrap_or(0)
    }

    pub fn type_shared(msg_type: u8, mask: u32) -> bool {
        mask & Self::type_to_flag(msg_type) != 0
    }

    pub fn get_fheap_addr(&self, key: u64) -> Option<u64> {
        self.messages.get(&key).map(|msg| msg.heap_addr)
    }

    pub fn create_index(&mut self) {}

    pub fn delete_index(&mut self) {
        self.messages.clear();
    }

    pub fn create_list(&mut self) {}

    pub fn bt2_convert_to_list_op(&self) -> Vec<SharedMessage> {
        self.messages.values().cloned().collect()
    }

    pub fn can_share_common(msg: &SharedMessage) -> bool {
        !msg.data.is_empty()
    }

    pub fn can_share(msg: &SharedMessage) -> bool {
        Self::can_share_common(msg)
    }

    pub fn try_share(&mut self, key: u64, msg: SharedMessage) -> bool {
        if Self::can_share(&msg) {
            self.messages.insert(key, msg);
            true
        } else {
            false
        }
    }

    pub fn incr_ref(&mut self, key: u64) -> Option<u32> {
        let msg = self.messages.get_mut(&key)?;
        msg.refcount = msg.refcount.saturating_add(1);
        Some(msg.refcount)
    }

    pub fn write_mesg(&mut self, key: u64, msg: SharedMessage) {
        self.messages.insert(key, msg);
    }

    pub fn delete(&mut self, key: u64) -> Option<SharedMessage> {
        self.messages.remove(&key)
    }

    pub fn find_in_list(&self, key: u64) -> Option<&SharedMessage> {
        self.messages.get(&key)
    }

    pub fn decr_ref(&mut self, key: u64) -> Option<u32> {
        let msg = self.messages.get_mut(&key)?;
        msg.refcount = msg.refcount.saturating_sub(1);
        Some(msg.refcount)
    }

    pub fn delete_from_index(&mut self, key: u64) -> Option<SharedMessage> {
        self.delete(key)
    }

    pub fn get_info(&self) -> SharedMessageInfo {
        SharedMessageInfo {
            count: self.messages.len(),
            total_bytes: self.cache_table_image_len(),
        }
    }

    pub fn reconstitute(messages: Vec<(u64, SharedMessage)>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
        }
    }

    pub fn get_refcount_bt2_cb(&self, key: u64) -> Option<u32> {
        self.get_refcount(key)
    }

    pub fn get_refcount(&self, key: u64) -> Option<u32> {
        self.messages.get(&key).map(|msg| msg.refcount)
    }

    pub fn read_mesg(&self, key: u64) -> Option<&[u8]> {
        self.messages.get(&key).map(|msg| msg.data.as_slice())
    }

    pub fn table_free(&mut self) {
        self.messages.clear();
    }

    pub fn list_free(&mut self) {
        self.messages.clear();
    }

    pub fn table_debug(&self) -> String {
        format!("{:?}", self.messages)
    }

    pub fn list_debug(&self) -> String {
        self.table_debug()
    }

    pub fn ih_size(&self) -> usize {
        self.messages.len()
    }

    pub fn compare_cb(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        Self::message_compare(lhs, rhs)
    }

    pub fn compare_iter_op(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        Self::message_compare(lhs, rhs)
    }

    pub fn message_compare(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        lhs.msg_type
            .cmp(&rhs.msg_type)
            .then_with(|| lhs.data.cmp(&rhs.data))
    }

    pub fn message_encode(msg: &SharedMessage) -> Vec<u8> {
        msg.encode()
    }

    pub fn bt2_crt_context() -> Self {
        Self::default()
    }

    pub fn bt2_dst_context(self) {}

    pub fn bt2_store(&mut self, key: u64, msg: SharedMessage) {
        self.write_mesg(key, msg);
    }

    pub fn bt2_debug(&self) -> String {
        self.table_debug()
    }
}

#[cfg(test)]
mod tests {
    use super::{SharedMessage, SharedMessageStore};

    #[test]
    fn shared_message_store_aliases_roundtrip() {
        let msg = SharedMessage::new(2, 99, vec![1, 2, 3]);
        assert!(SharedMessageStore::can_share(&msg));
        assert!(SharedMessageStore::type_shared(
            2,
            SharedMessageStore::type_to_flag(2)
        ));

        let mut store = SharedMessageStore::init();
        store.create_index();
        store.create_list();
        assert!(store.try_share(10, msg.clone()));
        assert_eq!(store.get_mesg_count_test(), 1);
        assert_eq!(store.get_fheap_addr(10), Some(99));
        assert_eq!(store.incr_ref(10), Some(2));
        assert_eq!(store.decr_ref(10), Some(1));
        assert_eq!(store.get_refcount_bt2_cb(10), Some(1));
        assert_eq!(store.read_mesg(10), Some([1, 2, 3].as_slice()));
        assert!(store.cache_table_image_len() >= msg.encoded_len());
        assert_eq!(SharedMessageStore::message_encode(&msg), msg.encode());
        assert_eq!(store.get_info().count, 1);
        assert!(store.table_debug().contains("SharedMessage"));

        let bytes = store.cache_list_serialize();
        assert_eq!(SharedMessageStore::cache_list_deserialize(&bytes), bytes);
        SharedMessageStore::cache_table_free_icr(bytes.clone());
        SharedMessageStore::cache_list_free_icr(bytes);
        assert_eq!(store.bt2_convert_to_list_op().len(), 1);

        let mut rebuilt = SharedMessageStore::reconstitute(vec![(1, msg.clone())]);
        rebuilt.bt2_store(2, msg);
        assert_eq!(rebuilt.ih_size(), 2);
        rebuilt.delete_from_index(1);
        rebuilt.table_free();
        assert_eq!(rebuilt.ih_size(), 0);
    }
}
