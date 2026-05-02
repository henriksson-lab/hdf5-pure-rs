use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginCache {
    plugins: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginPathTable {
    paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginRegistry {
    cache: PluginCache,
    paths: PluginPathTable,
    control_mask: u64,
    loading_enabled: bool,
    open_plugins: BTreeMap<String, usize>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self {
            cache: PluginCache::default(),
            paths: PluginPathTable::default(),
            control_mask: u64::MAX,
            loading_enabled: true,
            open_plugins: BTreeMap::new(),
        }
    }
}

#[allow(non_snake_case)]
pub fn H5PL__create_plugin_cache() -> PluginCache {
    PluginCache::default()
}

#[allow(non_snake_case)]
pub fn H5PL__close_plugin_cache(cache: &mut PluginCache) {
    cache.plugins.clear();
}

#[allow(non_snake_case)]
pub fn H5PL__expand_cache(cache: &mut PluginCache, additional: usize) {
    cache.plugins.reserve(additional);
}

#[allow(non_snake_case)]
pub fn H5PL__add_plugin(cache: &mut PluginCache, name: impl Into<String>) {
    let name = name.into();
    if !cache.plugins.contains(&name) {
        cache.plugins.push(name);
    }
}

#[allow(non_snake_case)]
pub fn H5PL__get_plugin_control_mask(registry: &PluginRegistry) -> u64 {
    registry.control_mask
}

#[allow(non_snake_case)]
pub fn H5PL__set_plugin_control_mask(registry: &mut PluginRegistry, mask: u64) {
    registry.control_mask = mask;
}

#[allow(non_snake_case)]
pub fn H5PL__init_package() -> PluginRegistry {
    PluginRegistry::default()
}

#[allow(non_snake_case)]
pub fn H5PL_term_package(registry: &mut PluginRegistry) {
    H5PL__close_plugin_cache(&mut registry.cache);
    H5PL__close_path_table(&mut registry.paths);
    registry.open_plugins.clear();
}

#[allow(non_snake_case)]
pub fn H5PL_load(registry: &mut PluginRegistry, name: &str) -> Result<()> {
    if !registry.loading_enabled {
        return Err(Error::Unsupported("plugin loading is disabled".into()));
    }
    if registry.cache.plugins.iter().any(|plugin| plugin == name) {
        registry
            .open_plugins
            .entry(name.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "dynamic plugin loading is not supported for '{name}'"
        )))
    }
}

#[allow(non_snake_case)]
pub fn H5PL__open(registry: &mut PluginRegistry, name: &str) -> Result<()> {
    H5PL_load(registry, name)
}

#[allow(non_snake_case)]
pub fn H5PL__close(registry: &mut PluginRegistry, name: &str) -> Result<()> {
    let Some(count) = registry.open_plugins.get_mut(name) else {
        return Err(Error::InvalidFormat(format!("plugin '{name}' is not open")));
    };
    *count = count.saturating_sub(1);
    if *count == 0 {
        registry.open_plugins.remove(name);
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL_iterate(registry: &PluginRegistry) -> Vec<String> {
    registry.cache.plugins.clone()
}

#[allow(non_snake_case)]
pub fn H5PL__insert_at(
    cache: &mut PluginCache,
    index: usize,
    name: impl Into<String>,
) -> Result<()> {
    H5PL__make_space_at(cache, index)?;
    cache.plugins[index] = name.into();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__make_space_at(cache: &mut PluginCache, index: usize) -> Result<()> {
    if index > cache.plugins.len() {
        return Err(Error::InvalidFormat(format!(
            "plugin cache index {index} out of range"
        )));
    }
    cache.plugins.insert(index, String::new());
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__replace_at(
    cache: &mut PluginCache,
    index: usize,
    name: impl Into<String>,
) -> Result<()> {
    let Some(slot) = cache.plugins.get_mut(index) else {
        return Err(Error::InvalidFormat(format!(
            "plugin cache index {index} out of range"
        )));
    };
    *slot = name.into();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__create_path_table() -> PluginPathTable {
    PluginPathTable::default()
}

#[allow(non_snake_case)]
pub fn H5PL__close_path_table(table: &mut PluginPathTable) {
    table.paths.clear();
}

#[allow(non_snake_case)]
pub fn H5PL__get_num_paths(table: &PluginPathTable) -> usize {
    table.paths.len()
}

#[allow(non_snake_case)]
pub fn H5PL__expand_path_table(table: &mut PluginPathTable, additional: usize) {
    table.paths.reserve(additional);
}

#[allow(non_snake_case)]
pub fn H5PL__append_path(table: &mut PluginPathTable, path: impl Into<PathBuf>) {
    table.paths.push(path.into());
}

#[allow(non_snake_case)]
pub fn H5PL__prepend_path(table: &mut PluginPathTable, path: impl Into<PathBuf>) {
    table.paths.insert(0, path.into());
}

#[allow(non_snake_case)]
pub fn H5PL__replace_path(
    table: &mut PluginPathTable,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    let Some(slot) = table.paths.get_mut(index) else {
        return Err(Error::InvalidFormat(format!(
            "plugin path index {index} out of range"
        )));
    };
    *slot = path.into();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__insert_path(
    table: &mut PluginPathTable,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    if index > table.paths.len() {
        return Err(Error::InvalidFormat(format!(
            "plugin path index {index} out of range"
        )));
    }
    table.paths.insert(index, path.into());
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__remove_path(table: &mut PluginPathTable, index: usize) -> Result<PathBuf> {
    if index >= table.paths.len() {
        return Err(Error::InvalidFormat(format!(
            "plugin path index {index} out of range"
        )));
    }
    Ok(table.paths.remove(index))
}

#[allow(non_snake_case)]
pub fn H5PL__get_path(table: &PluginPathTable, index: usize) -> Result<&Path> {
    table
        .paths
        .get(index)
        .map(PathBuf::as_path)
        .ok_or_else(|| Error::InvalidFormat(format!("plugin path index {index} out of range")))
}

#[allow(non_snake_case)]
pub fn H5PL__path_table_iterate(table: &PluginPathTable) -> Vec<PathBuf> {
    table.paths.clone()
}

#[allow(non_snake_case)]
pub fn H5PL__path_table_iterate_process_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[allow(non_snake_case)]
pub fn H5PL__find_plugin_in_path_table(table: &PluginPathTable, name: &str) -> Option<PathBuf> {
    table
        .paths
        .iter()
        .map(|path| path.join(name))
        .find(|candidate| candidate.exists())
}

#[allow(non_snake_case)]
pub fn H5PL__find_plugin_in_path(path: &Path, name: &str) -> Option<PathBuf> {
    let candidate = path.join(name);
    candidate.exists().then_some(candidate)
}

#[allow(non_snake_case)]
pub fn H5PLset_loading_state(registry: &mut PluginRegistry, enabled: bool) {
    registry.loading_enabled = enabled;
}

#[allow(non_snake_case)]
pub fn H5PLappend(registry: &mut PluginRegistry, path: impl Into<PathBuf>) {
    H5PL__append_path(&mut registry.paths, path);
}

#[allow(non_snake_case)]
pub fn H5PLprepend(registry: &mut PluginRegistry, path: impl Into<PathBuf>) {
    H5PL__prepend_path(&mut registry.paths, path);
}

#[allow(non_snake_case)]
pub fn H5PLreplace(
    registry: &mut PluginRegistry,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    H5PL__replace_path(&mut registry.paths, index, path)
}

#[allow(non_snake_case)]
pub fn H5PLinsert(
    registry: &mut PluginRegistry,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    H5PL__insert_path(&mut registry.paths, index, path)
}

#[allow(non_snake_case)]
pub fn H5PLremove(registry: &mut PluginRegistry, index: usize) -> Result<PathBuf> {
    H5PL__remove_path(&mut registry.paths, index)
}

#[allow(non_snake_case)]
pub fn H5PLget(registry: &PluginRegistry, index: usize) -> Result<&Path> {
    H5PL__get_path(&registry.paths, index)
}

#[allow(non_snake_case)]
pub fn H5PLsize(registry: &PluginRegistry) -> usize {
    H5PL__get_num_paths(&registry.paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_path_table_mutates_order() {
        let mut registry = H5PL__init_package();
        H5PLappend(&mut registry, "/b");
        H5PLprepend(&mut registry, "/a");
        H5PLinsert(&mut registry, 1, "/mid").unwrap();
        assert_eq!(H5PLsize(&registry), 3);
        assert_eq!(H5PLget(&registry, 1).unwrap(), Path::new("/mid"));
        assert_eq!(H5PLremove(&mut registry, 1).unwrap(), PathBuf::from("/mid"));
    }

    #[test]
    fn plugin_load_is_explicitly_unsupported_without_cached_plugin() {
        let mut registry = H5PL__init_package();
        assert!(H5PL_load(&mut registry, "missing").is_err());
        H5PL__add_plugin(&mut registry.cache, "known");
        H5PL_load(&mut registry, "known").unwrap();
        H5PL__close(&mut registry, "known").unwrap();
    }
}
