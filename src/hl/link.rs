use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::format::messages::link::LinkMessage;
use crate::hl::group::{Group, LinkInfo, LinkValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkClass {
    pub id: u8,
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinkClassRegistry {
    classes: BTreeMap<u8, LinkClass>,
    external_registered: bool,
}

impl LinkClassRegistry {
    pub fn init() -> Self {
        let mut registry = Self::default();
        registry.register(LinkClass {
            id: 0,
            name: "hard".to_string(),
        });
        registry.register(LinkClass {
            id: 1,
            name: "soft".to_string(),
        });
        registry.register_external();
        registry
    }

    pub fn init_package() -> Self {
        Self::init()
    }

    pub fn term_package(&mut self) {
        self.classes.clear();
        self.external_registered = false;
    }

    pub fn register_external(&mut self) {
        self.external_registered = true;
        self.register(LinkClass {
            id: 64,
            name: "external".to_string(),
        });
    }

    pub fn find_class_idx(&self, id: u8) -> Option<usize> {
        self.classes.keys().position(|class_id| *class_id == id)
    }

    pub fn find_class(&self, id: u8) -> Option<&LinkClass> {
        self.classes.get(&id)
    }

    pub fn register(&mut self, class: LinkClass) {
        self.classes.insert(class.id, class);
    }

    pub fn unregister(&mut self, id: u8) -> Option<LinkClass> {
        if id == 64 {
            self.external_registered = false;
        }
        self.classes.remove(&id)
    }

    pub fn is_registered(&self, id: u8) -> bool {
        self.classes.contains_key(&id)
    }

    pub fn register_api(&mut self, class: LinkClass) {
        self.register(class);
    }

    pub fn unregister_api(&mut self, id: u8) -> Option<LinkClass> {
        self.unregister(id)
    }

    pub fn is_registered_api(&self, id: u8) -> bool {
        self.is_registered(id)
    }
}

pub fn extern_traverse(group: &Group, name: &str) -> Result<Option<(String, String)>> {
    Ok(group.find_link_by_name(name)?.external_link)
}

pub fn link(group: &Group, name: &str) -> Result<LinkMessage> {
    group.find_link_by_name(name)
}

pub fn link_object(group: &Group, name: &str) -> Result<u64> {
    group
        .find_link_by_name(name)?
        .hard_link_addr
        .ok_or_else(|| {
            Error::InvalidFormat(format!("link '{name}' does not reference an object header"))
        })
}

pub fn create_soft_api_common(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_soft(name, target)
}

pub fn create_hard_api_common(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_hard(name, target)
}

pub fn create_real(group: &Group, name: &str) -> Result<u64> {
    link_object(group, name)
}

pub fn create_soft(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_soft(name, target)
}

pub fn create_soft_async(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    create_soft(writer, name, target)
}

pub fn create_hard(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_hard(name, target)
}

pub fn create_hard_async(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    create_hard(writer, name, target)
}

pub fn create_external(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    filename: &str,
    object_path: &str,
) -> Result<()> {
    writer.link_external(name, filename, object_path)
}

pub fn create_ud(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    filename: &str,
    object_path: &str,
) -> Result<()> {
    writer.link_external(name, filename, object_path)
}

pub fn create_ud_api(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    filename: &str,
    object_path: &str,
) -> Result<()> {
    create_ud(writer, name, filename, object_path)
}

pub fn get_val_cb(link: &LinkMessage) -> Option<LinkValue> {
    if let Some(target) = &link.soft_link_target {
        return Some(LinkValue::Soft(target.clone()));
    }
    link.external_link
        .as_ref()
        .map(|(filename, object_path)| LinkValue::External {
            filename: filename.clone(),
            object_path: object_path.clone(),
        })
}

pub fn get_val(group: &Group, name: &str) -> Result<Option<LinkValue>> {
    Ok(get_val_cb(&group.find_link_by_name(name)?))
}

pub fn get_val_by_idx_cb(link: &LinkMessage) -> Option<LinkValue> {
    get_val_cb(link)
}

pub fn get_val_by_idx(group: &Group, index: usize) -> Result<Option<LinkValue>> {
    group.link_value_by_idx(index)
}

pub fn exists_final_cb(group: &Group, name: &str) -> Result<bool> {
    group.link_exists(name)
}

pub fn exists_inter_cb(group: &Group, name: &str) -> Result<bool> {
    group.link_exists(name)
}

pub fn exists_tolerant(group: &Group, name: &str) -> bool {
    group.link_exists(name).unwrap_or(false)
}

pub fn exists(group: &Group, name: &str) -> Result<bool> {
    group.link_exists(name)
}

pub fn exists_api_common(group: &Group, name: &str) -> Result<bool> {
    exists(group, name)
}

pub fn get_info_by_idx_cb(link: &LinkMessage) -> Result<LinkInfo> {
    super::group::link_info_from_message(link)
}

pub fn get_info_by_idx(group: &Group, index: usize) -> Result<LinkInfo> {
    group.link_info_by_idx(index)
}

pub fn get_name_by_idx(group: &Group, index: usize) -> Result<String> {
    group.link_name_by_idx(index)
}

pub fn link_copy_file(link: &LinkMessage) -> LinkMessage {
    link.clone()
}

pub fn copy(link: &LinkMessage) -> LinkMessage {
    link_copy_file(link)
}

pub fn iterate(group: &Group) -> Result<Vec<LinkMessage>> {
    group.links()
}

pub fn iterate_by_name2(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn visit2(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn visit_by_name2(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn iterate1(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn iterate_by_name1(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn visit1(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn visit_by_name1(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn get_ocrt_info(link: &LinkMessage) -> Option<u64> {
    link.creation_order
}

pub fn iterate_api_common(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

pub fn iterate2_shim(group: &Group) -> Result<Vec<LinkMessage>> {
    iterate(group)
}

#[derive(Debug, Clone, Default)]
pub struct LinkTable {
    links: BTreeMap<String, LinkMessage>,
}

impl LinkTable {
    pub fn from_links(links: impl IntoIterator<Item = LinkMessage>) -> Self {
        let links = links
            .into_iter()
            .map(|link| (link.name.clone(), link))
            .collect();
        Self { links }
    }

    pub fn insert(&mut self, link: LinkMessage) -> Option<LinkMessage> {
        self.links.insert(link.name.clone(), link)
    }

    pub fn delete(&mut self, name: &str) -> Result<LinkMessage> {
        self.links
            .remove(name)
            .ok_or_else(|| Error::InvalidFormat(format!("link '{name}' not found")))
    }

    pub fn delete_by_idx(&mut self, index: usize) -> Result<LinkMessage> {
        let name = self
            .links
            .keys()
            .nth(index)
            .cloned()
            .ok_or_else(|| Error::InvalidFormat(format!("link index {index} out of range")))?;
        self.delete(&name)
    }

    pub fn move_link(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        if self.links.contains_key(new_name) {
            return Err(Error::InvalidFormat(format!(
                "destination link '{new_name}' already exists"
            )));
        }
        let mut link = self.delete(old_name)?;
        link.name = new_name.to_string();
        self.insert(link);
        Ok(())
    }
}

#[allow(non_snake_case)]
pub fn H5L__delete_cb(table: &mut LinkTable, name: &str) -> Result<LinkMessage> {
    table.delete(name)
}

#[allow(non_snake_case)]
pub fn H5L__delete(table: &mut LinkTable, name: &str) -> Result<LinkMessage> {
    H5L__delete_cb(table, name)
}

#[allow(non_snake_case)]
pub fn H5L__delete_by_idx_cb(table: &mut LinkTable, index: usize) -> Result<LinkMessage> {
    table.delete_by_idx(index)
}

#[allow(non_snake_case)]
pub fn H5L__delete_by_idx(table: &mut LinkTable, index: usize) -> Result<LinkMessage> {
    H5L__delete_by_idx_cb(table, index)
}

#[allow(non_snake_case)]
pub fn H5L__delete_by_idx_api_common(table: &mut LinkTable, index: usize) -> Result<LinkMessage> {
    H5L__delete_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5L__move(table: &mut LinkTable, old_name: &str, new_name: &str) -> Result<()> {
    table.move_link(old_name, new_name)
}

#[allow(non_snake_case)]
pub fn H5Lmove(table: &mut LinkTable, old_name: &str, new_name: &str) -> Result<()> {
    H5L__move(table, old_name, new_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_registry_tracks_builtin_classes() {
        let mut registry = LinkClassRegistry::init();
        assert!(registry.is_registered(0));
        assert!(registry.is_registered(1));
        assert!(registry.is_registered(64));
        assert_eq!(registry.find_class_idx(64), Some(2));
        registry.unregister(64);
        assert!(!registry.is_registered(64));
    }

    #[test]
    fn link_table_deletes_and_moves_links() {
        let mut table = LinkTable::default();
        table.insert(LinkMessage {
            link_type: crate::format::messages::link::LinkType::Soft,
            creation_order: None,
            char_encoding: 0,
            name: "old".into(),
            hard_link_addr: None,
            soft_link_target: Some("/target".into()),
            external_link: None,
        });
        H5Lmove(&mut table, "old", "new").unwrap();
        assert!(H5L__delete(&mut table, "new").is_ok());
        assert!(H5L__delete(&mut table, "new").is_err());
    }
}
