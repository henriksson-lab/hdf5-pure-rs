use std::collections::BTreeMap;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct H5Map {
    key_type: String,
    value_type: String,
    create_plist: BTreeMap<String, Vec<u8>>,
    access_plist: BTreeMap<String, Vec<u8>>,
    entries: BTreeMap<Vec<u8>, Vec<u8>>,
    open: bool,
}

impl H5Map {
    pub fn new(key_type: impl Into<String>, value_type: impl Into<String>) -> Self {
        Self {
            key_type: key_type.into(),
            value_type: value_type.into(),
            create_plist: BTreeMap::new(),
            access_plist: BTreeMap::new(),
            entries: BTreeMap::new(),
            open: true,
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.open {
            Ok(())
        } else {
            Err(Error::InvalidFormat("map handle is closed".into()))
        }
    }
}

#[allow(non_snake_case)]
pub fn H5M_init() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5M__init_package() -> bool {
    H5M_init()
}

#[allow(non_snake_case)]
pub fn H5M_top_term_package() {}

#[allow(non_snake_case)]
pub fn H5M_term_package() {}

#[allow(non_snake_case)]
pub fn H5M__close_cb(map: &mut H5Map) {
    map.open = false;
}

#[allow(non_snake_case)]
pub fn H5M__create_api_common(key_type: impl Into<String>, value_type: impl Into<String>) -> H5Map {
    H5Map::new(key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5Mcreate_anon(key_type: impl Into<String>, value_type: impl Into<String>) -> H5Map {
    H5M__create_api_common(key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5M__open_api_common(map: &H5Map) -> Result<H5Map> {
    map.ensure_open()?;
    Ok(map.clone())
}

#[allow(non_snake_case)]
pub fn H5Mclose(map: &mut H5Map) {
    H5M__close_cb(map);
}

#[allow(non_snake_case)]
pub fn H5Mget_key_type(map: &H5Map) -> Result<&str> {
    map.ensure_open()?;
    Ok(&map.key_type)
}

#[allow(non_snake_case)]
pub fn H5Mget_val_type(map: &H5Map) -> Result<&str> {
    map.ensure_open()?;
    Ok(&map.value_type)
}

#[allow(non_snake_case)]
pub fn H5Mget_create_plist(map: &H5Map) -> Result<BTreeMap<String, Vec<u8>>> {
    map.ensure_open()?;
    Ok(map.create_plist.clone())
}

#[allow(non_snake_case)]
pub fn H5Mget_access_plist(map: &H5Map) -> Result<BTreeMap<String, Vec<u8>>> {
    map.ensure_open()?;
    Ok(map.access_plist.clone())
}

#[allow(non_snake_case)]
pub fn H5Mget_count(map: &H5Map) -> Result<usize> {
    map.ensure_open()?;
    Ok(map.entries.len())
}

#[allow(non_snake_case)]
pub fn H5M__put_api_common(map: &mut H5Map, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    map.ensure_open()?;
    map.entries.insert(key, value);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5Mput(map: &mut H5Map, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    H5M__put_api_common(map, key, value)
}

#[allow(non_snake_case)]
pub fn H5Mput_async(map: &mut H5Map, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    H5Mput(map, key, value)
}

#[allow(non_snake_case)]
pub fn H5M__get_api_common(map: &H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    map.ensure_open()?;
    Ok(map.entries.get(key).cloned())
}

#[allow(non_snake_case)]
pub fn H5Mget(map: &H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    H5M__get_api_common(map, key)
}

#[allow(non_snake_case)]
pub fn H5Mget_async(map: &H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    H5Mget(map, key)
}

#[allow(non_snake_case)]
pub fn H5Mexists(map: &H5Map, key: &[u8]) -> Result<bool> {
    map.ensure_open()?;
    Ok(map.entries.contains_key(key))
}

#[allow(non_snake_case)]
pub fn H5Miterate(map: &H5Map) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    map.ensure_open()?;
    Ok(map
        .entries
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

#[allow(non_snake_case)]
pub fn H5Miterate_by_name(map: &H5Map) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    H5Miterate(map)
}

#[allow(non_snake_case)]
pub fn H5Mdelete(map: &mut H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    map.ensure_open()?;
    Ok(map.entries.remove(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_api_put_get_iterate_delete() {
        let mut map = H5Mcreate_anon("u8", "bytes");
        H5Mput(&mut map, b"k".to_vec(), b"v".to_vec()).unwrap();
        assert_eq!(H5Mget_count(&map).unwrap(), 1);
        assert!(H5Mexists(&map, b"k").unwrap());
        assert_eq!(H5Mget(&map, b"k").unwrap(), Some(b"v".to_vec()));
        assert_eq!(
            H5Miterate(&map).unwrap(),
            vec![(b"k".to_vec(), b"v".to_vec())]
        );
        assert_eq!(H5Mdelete(&mut map, b"k").unwrap(), Some(b"v".to_vec()));
        H5Mclose(&mut map);
        assert!(H5Mget_count(&map).is_err());
    }
}
