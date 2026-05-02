use std::collections::HashMap;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VirtualMapping {
    pub source_file: String,
    pub source_dataset: String,
    pub min_dims: Vec<u64>,
    pub max_dims: Vec<Option<u64>>,
    pub open: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VirtualLayout {
    pub mappings: Vec<VirtualMapping>,
    pub unlimited: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SingleChunkIndex {
    pub open: bool,
    pub space_allocated: bool,
    pub chunk_addr: Option<u64>,
    pub metadata_loaded: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactStorage {
    pub data: Vec<u8>,
    pub space_allocated: bool,
    pub dirty: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetApi {
    pub name: Option<String>,
    pub extent: Vec<u64>,
    pub raw: Vec<u8>,
    pub virtual_layout: Option<VirtualLayout>,
}

#[allow(non_snake_case)]
pub fn H5D_virtual_check_min_dims(mapping: &VirtualMapping, dims: &[u64]) -> bool {
    mapping
        .min_dims
        .iter()
        .zip(dims)
        .all(|(minimum, actual)| actual >= minimum)
}

#[allow(non_snake_case)]
pub fn H5D__virtual_store_layout(dataset: &mut DatasetApi, layout: VirtualLayout) {
    dataset.virtual_layout = Some(layout);
}

#[allow(non_snake_case)]
pub fn H5D__virtual_load_layout(dataset: &DatasetApi) -> Option<VirtualLayout> {
    dataset.virtual_layout.clone()
}

#[allow(non_snake_case)]
pub fn H5D__virtual_copy_layout(layout: &VirtualLayout) -> VirtualLayout {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5D__virtual_free_layout_mappings(layout: &mut VirtualLayout) {
    layout.mappings.clear();
}

#[allow(non_snake_case)]
pub fn H5D__virtual_reset_layout(layout: &mut VirtualLayout) {
    layout.mappings.clear();
    layout.unlimited = false;
}

#[allow(non_snake_case)]
pub fn H5D__virtual_open_source_dset(mapping: &mut VirtualMapping) -> Result<()> {
    if mapping.source_file.is_empty() || mapping.source_dataset.is_empty() {
        return Err(Error::InvalidFormat(
            "virtual source mapping is incomplete".into(),
        ));
    }
    mapping.open = true;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5D__virtual_set_extent_unlim(layout: &mut VirtualLayout) {
    layout.unlimited = true;
}

#[allow(non_snake_case)]
pub fn H5D__virtual_init_all(layout: &mut VirtualLayout) -> Result<()> {
    for mapping in &mut layout.mappings {
        H5D__virtual_open_source_dset(mapping)?;
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5D__virtual_pre_io_process_mapping(mapping: &VirtualMapping, dims: &[u64]) -> bool {
    mapping.open && H5D_virtual_check_min_dims(mapping, dims)
}

#[allow(non_snake_case)]
pub fn H5D__virtual_flush(_layout: &mut VirtualLayout) {}

#[allow(non_snake_case)]
pub fn H5D__virtual_refresh_source_dset(mapping: &mut VirtualMapping) -> Result<()> {
    H5D__virtual_open_source_dset(mapping)
}

#[allow(non_snake_case)]
pub fn H5D__virtual_refresh_source_dsets(layout: &mut VirtualLayout) -> Result<()> {
    H5D__virtual_init_all(layout)
}

#[allow(non_snake_case)]
pub fn H5D__virtual_release_source_dset_files(layout: &mut VirtualLayout) {
    for mapping in &mut layout.mappings {
        mapping.open = false;
    }
}

#[allow(non_snake_case)]
pub fn H5D__mappings_to_leaves(layout: &VirtualLayout) -> Vec<VirtualMapping> {
    layout.mappings.clone()
}

#[allow(non_snake_case)]
pub fn H5D__virtual_not_in_tree_grow(layout: &mut VirtualLayout, mapping: VirtualMapping) {
    if !layout.mappings.contains(&mapping) {
        layout.mappings.push(mapping);
    }
}

#[allow(non_snake_case)]
pub fn H5D__should_build_tree(layout: &VirtualLayout) -> bool {
    layout.mappings.len() > 1
}

#[allow(non_snake_case)]
pub fn H5D__virtual_close_mapping(mapping: &mut VirtualMapping) {
    mapping.open = false;
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_init() -> SingleChunkIndex {
    SingleChunkIndex::default()
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_create(index: &mut SingleChunkIndex, addr: u64) {
    index.open = true;
    index.space_allocated = true;
    index.chunk_addr = Some(addr);
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_close(index: &mut SingleChunkIndex) {
    index.open = false;
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_is_open(index: &SingleChunkIndex) -> bool {
    index.open
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_is_space_alloc(index: &SingleChunkIndex) -> bool {
    index.space_allocated
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_insert(index: &mut SingleChunkIndex, addr: u64) {
    H5D__single_idx_create(index, addr);
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_get_addr(index: &SingleChunkIndex) -> Option<u64> {
    index.chunk_addr
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_load_metadata(index: &mut SingleChunkIndex) {
    index.metadata_loaded = true;
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_iterate(index: &SingleChunkIndex) -> impl Iterator<Item = u64> + '_ {
    index.chunk_addr.into_iter()
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_remove(index: &mut SingleChunkIndex) -> Option<u64> {
    index.space_allocated = false;
    index.chunk_addr.take()
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_delete(index: &mut SingleChunkIndex) {
    *index = SingleChunkIndex::default();
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_copy_setup(index: &SingleChunkIndex) -> SingleChunkIndex {
    index.clone()
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_reset(index: &mut SingleChunkIndex) {
    H5D__single_idx_delete(index);
}

#[allow(non_snake_case)]
pub fn H5D__single_idx_dump(index: &SingleChunkIndex) -> String {
    format!(
        "single_idx(open={}, allocated={}, addr={:?})",
        index.open, index.space_allocated, index.chunk_addr
    )
}

#[allow(non_snake_case)]
pub fn H5D__create_api_common(name: Option<String>, extent: Vec<u64>) -> DatasetApi {
    DatasetApi {
        name,
        extent,
        raw: Vec::new(),
        virtual_layout: None,
    }
}

#[allow(non_snake_case)]
pub fn H5Dcreate_anon(extent: Vec<u64>) -> DatasetApi {
    H5D__create_api_common(None, extent)
}

#[allow(non_snake_case)]
pub fn H5D__open_api_common(dataset: &DatasetApi) -> DatasetApi {
    dataset.clone()
}

#[allow(non_snake_case)]
pub fn H5Dclose(_dataset: DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5D__get_space_api_common(dataset: &DatasetApi) -> &[u64] {
    &dataset.extent
}

#[allow(non_snake_case)]
pub fn H5Dread_multi(datasets: &[DatasetApi]) -> Vec<Vec<u8>> {
    datasets.iter().map(|dataset| dataset.raw.clone()).collect()
}

#[allow(non_snake_case)]
pub fn H5Dread_multi_async(_datasets: &[DatasetApi]) -> Result<()> {
    Err(Error::Unsupported(
        "async dataset reads require event-set infrastructure".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5Dread_chunk2(dataset: &DatasetApi, offset: usize, len: usize) -> Result<Vec<u8>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("dataset chunk read overflow".into()))?;
    Ok(dataset
        .raw
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat("dataset chunk read out of bounds".into()))?
        .to_vec())
}

#[allow(non_snake_case)]
pub fn H5D__write_api_common(dataset: &mut DatasetApi, data: &[u8]) {
    dataset.raw.clear();
    dataset.raw.extend_from_slice(data);
}

#[allow(non_snake_case)]
pub fn H5Dwrite_multi(datasets: &mut [DatasetApi], payloads: &[Vec<u8>]) {
    for (dataset, payload) in datasets.iter_mut().zip(payloads) {
        H5D__write_api_common(dataset, payload);
    }
}

#[allow(non_snake_case)]
pub fn H5Dwrite_multi_async(_datasets: &mut [DatasetApi], _payloads: &[Vec<u8>]) -> Result<()> {
    Err(Error::Unsupported(
        "async dataset writes require event-set infrastructure".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5Dflush(_dataset: &mut DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5Drefresh(_dataset: &mut DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5Dformat_convert(dataset: &mut DatasetApi) {
    dataset.raw.shrink_to_fit();
}

#[allow(non_snake_case)]
pub fn H5Dchunk_iter(dataset: &DatasetApi, chunk_size: usize) -> impl Iterator<Item = &[u8]> {
    dataset.raw.chunks(chunk_size.max(1))
}

#[allow(non_snake_case)]
pub fn H5D__compact_construct(data: Vec<u8>) -> CompactStorage {
    CompactStorage {
        space_allocated: true,
        dirty: false,
        data,
    }
}

#[allow(non_snake_case)]
pub fn H5D__compact_is_space_alloc(storage: &CompactStorage) -> bool {
    storage.space_allocated
}

#[allow(non_snake_case)]
pub fn H5D__compact_io_init(storage: &mut CompactStorage) {
    storage.space_allocated = true;
}

#[allow(non_snake_case)]
pub fn H5D__compact_iovv_memmanage_cb(storage: &CompactStorage) -> usize {
    storage.data.len()
}

#[allow(non_snake_case)]
pub fn H5D__compact_readvv(storage: &CompactStorage, offset: usize, len: usize) -> Result<Vec<u8>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("compact read overflow".into()))?;
    Ok(storage
        .data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat("compact read out of bounds".into()))?
        .to_vec())
}

#[allow(non_snake_case)]
pub fn H5D__compact_writevv(
    storage: &mut CompactStorage,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| Error::InvalidFormat("compact write overflow".into()))?;
    if storage.data.len() < end {
        storage.data.resize(end, 0);
    }
    storage.data[offset..end].copy_from_slice(data);
    storage.space_allocated = true;
    storage.dirty = true;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5D__compact_flush(storage: &mut CompactStorage) {
    storage.dirty = false;
}

#[allow(non_snake_case)]
pub fn H5D__compact_dest(storage: &mut CompactStorage) {
    storage.data.clear();
    storage.space_allocated = false;
    storage.dirty = false;
}

#[allow(non_snake_case)]
pub fn H5D__compact_copy(storage: &CompactStorage) -> CompactStorage {
    storage.clone()
}

#[allow(non_snake_case)]
pub fn H5D__get_chunk_storage_size(
    chunks: &HashMap<Vec<u64>, Vec<u8>>,
    coord: &[u64],
) -> Option<usize> {
    chunks.get(coord).map(Vec::len)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkInfo {
    pub coord: Vec<u64>,
    pub addr: u64,
    pub size: usize,
    pub filter_mask: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkTable {
    pub chunk_dims: Vec<u64>,
    pub chunks: HashMap<Vec<u64>, ChunkInfo>,
    pub data: HashMap<Vec<u64>, Vec<u8>>,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub locked: bool,
}

#[allow(non_snake_case)]
pub fn H5D__chunk_set_info_real(info: &mut ChunkInfo, addr: u64, size: usize, filter_mask: u32) {
    info.addr = addr;
    info.size = size;
    info.filter_mask = filter_mask;
}

#[allow(non_snake_case)]
pub fn H5D__chunk_set_info(info: &mut ChunkInfo, addr: u64, size: usize) {
    H5D__chunk_set_info_real(info, addr, size, info.filter_mask);
}

#[allow(non_snake_case)]
pub fn H5D__chunk_set_sizes(table: &mut ChunkTable, chunk_dims: Vec<u64>) {
    table.chunk_dims = chunk_dims;
}

#[allow(non_snake_case)]
pub fn H5D__chunk_construct(chunk_dims: Vec<u64>) -> ChunkTable {
    ChunkTable {
        chunk_dims,
        ..ChunkTable::default()
    }
}

#[allow(non_snake_case)]
pub fn H5D__chunk_io_init(_table: &mut ChunkTable) {}

#[allow(non_snake_case)]
pub fn H5D__chunk_io_init_selections(coords: &[Vec<u64>]) -> Vec<Vec<u64>> {
    coords.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D__chunk_mem_alloc(size: usize) -> Vec<u8> {
    vec![0; size]
}

#[allow(non_snake_case)]
pub fn H5D__chunk_mem_realloc(mut buf: Vec<u8>, size: usize) -> Vec<u8> {
    buf.resize(size, 0);
    buf
}

#[allow(non_snake_case)]
pub fn H5D__free_piece_info(info: &mut ChunkInfo) {
    *info = ChunkInfo::default();
}

#[allow(non_snake_case)]
pub fn H5D__create_piece_map_single(coord: Vec<u64>, data: Vec<u8>) -> HashMap<Vec<u64>, Vec<u8>> {
    HashMap::from([(coord, data)])
}

#[allow(non_snake_case)]
pub fn H5D__create_piece_file_map_all(table: &ChunkTable) -> HashMap<Vec<u64>, u64> {
    table
        .chunks
        .iter()
        .map(|(coord, info)| (coord.clone(), info.addr))
        .collect()
}

#[allow(non_snake_case)]
pub fn H5D__create_piece_file_map_hyper(
    table: &ChunkTable,
    coords: &[Vec<u64>],
) -> HashMap<Vec<u64>, u64> {
    coords
        .iter()
        .filter_map(|coord| {
            table
                .chunks
                .get(coord)
                .map(|info| (coord.clone(), info.addr))
        })
        .collect()
}

#[allow(non_snake_case)]
pub fn H5D__create_piece_mem_map_hyper(
    table: &ChunkTable,
    coords: &[Vec<u64>],
) -> HashMap<Vec<u64>, Vec<u8>> {
    coords
        .iter()
        .filter_map(|coord| {
            table
                .data
                .get(coord)
                .map(|data| (coord.clone(), data.clone()))
        })
        .collect()
}

#[allow(non_snake_case)]
pub fn H5D__piece_file_cb(info: &ChunkInfo) -> u64 {
    info.addr
}

#[allow(non_snake_case)]
pub fn H5D__piece_mem_cb(data: &[u8]) -> usize {
    data.len()
}

#[allow(non_snake_case)]
pub fn H5D__chunk_mdio_init(_table: &mut ChunkTable) {}

#[allow(non_snake_case)]
pub fn H5D__chunk_may_use_select_io(coords: &[Vec<u64>]) -> bool {
    coords.len() > 1
}

#[allow(non_snake_case)]
pub fn H5D__chunk_read(table: &mut ChunkTable, coord: &[u64]) -> Result<Vec<u8>> {
    if let Some(data) = table.data.get(coord) {
        table.cache_hits += 1;
        return Ok(data.clone());
    }
    table.cache_misses += 1;
    Err(Error::InvalidFormat("chunk not found".into()))
}

#[allow(non_snake_case)]
pub fn H5D__chunk_write(table: &mut ChunkTable, coord: Vec<u64>, data: Vec<u8>) {
    let info = ChunkInfo {
        coord: coord.clone(),
        addr: table.chunks.len() as u64,
        size: data.len(),
        filter_mask: 0,
    };
    table.chunks.insert(coord.clone(), info);
    table.data.insert(coord, data);
}

#[allow(non_snake_case)]
pub fn H5D__chunk_flush(_table: &mut ChunkTable) {}

#[allow(non_snake_case)]
pub fn H5D__chunk_io_term(_table: &mut ChunkTable) {}

#[allow(non_snake_case)]
pub fn H5D__chunk_dest(table: &mut ChunkTable) {
    table.chunks.clear();
    table.data.clear();
}

#[allow(non_snake_case)]
pub fn H5D_chunk_idx_reset(table: &mut ChunkTable) {
    H5D__chunk_dest(table);
}

#[allow(non_snake_case)]
pub fn H5D__chunk_cinfo_cache_reset(table: &mut ChunkTable) {
    table.cache_hits = 0;
    table.cache_misses = 0;
}

#[allow(non_snake_case)]
pub fn H5D__chunk_cinfo_cache_update(table: &mut ChunkTable, hit: bool) {
    if hit {
        table.cache_hits += 1;
    } else {
        table.cache_misses += 1;
    }
}

#[allow(non_snake_case)]
pub fn H5D__chunk_cinfo_cache_found(table: &mut ChunkTable) -> bool {
    let found = table.cache_hits > 0;
    H5D__chunk_cinfo_cache_update(table, found);
    found
}

#[allow(non_snake_case)]
pub fn H5D__chunk_create(chunk_dims: Vec<u64>) -> ChunkTable {
    H5D__chunk_construct(chunk_dims)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_hash_val(coord: &[u64]) -> u64 {
    coord.iter().fold(1469598103934665603, |hash, value| {
        (hash ^ value).wrapping_mul(1099511628211)
    })
}

#[allow(non_snake_case)]
pub fn H5D__chunk_lookup<'a>(table: &'a ChunkTable, coord: &[u64]) -> Option<&'a ChunkInfo> {
    table.chunks.get(coord)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_flush_entry(_table: &mut ChunkTable, _coord: &[u64]) {}

#[allow(non_snake_case)]
pub fn H5D__chunk_cache_evict(table: &mut ChunkTable, coord: &[u64]) -> Option<Vec<u8>> {
    table.data.remove(coord)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_cache_prune(table: &mut ChunkTable, max_chunks: usize) {
    while table.data.len() > max_chunks {
        if let Some(coord) = table.data.keys().next().cloned() {
            table.data.remove(&coord);
        }
    }
}

#[allow(non_snake_case)]
pub fn H5D__chunk_lock(table: &mut ChunkTable) {
    table.locked = true;
}

#[allow(non_snake_case)]
pub fn H5D__chunk_unlock(table: &mut ChunkTable) {
    table.locked = false;
}

#[allow(non_snake_case)]
pub fn H5D__chunk_allocated(table: &ChunkTable, coord: &[u64]) -> bool {
    table.chunks.contains_key(coord)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_allocate(table: &mut ChunkTable, coord: Vec<u64>, size: usize) {
    H5D__chunk_write(table, coord, vec![0; size]);
}

#[allow(non_snake_case)]
pub fn H5D__chunk_update_old_edge_chunks(_table: &mut ChunkTable) {}

#[allow(non_snake_case)]
pub fn H5D__chunk_cmp_coll_fill_info(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    left.coord
        .cmp(&right.coord)
        .then_with(|| left.addr.cmp(&right.addr))
}

#[allow(non_snake_case)]
pub fn H5D__chunk_prune_fill(table: &mut ChunkTable) {
    table
        .data
        .retain(|_, data| data.iter().any(|byte| *byte != 0));
}

#[allow(non_snake_case)]
pub fn H5D__chunk_prune_by_extent(table: &mut ChunkTable, extent: &[u64]) {
    table
        .data
        .retain(|coord, _| coord.iter().zip(extent).all(|(c, e)| c < e));
    table
        .chunks
        .retain(|coord, _| coord.iter().zip(extent).all(|(c, e)| c < e));
}

#[allow(non_snake_case)]
pub fn H5D__chunk_addrmap_cb(info: &ChunkInfo) -> (Vec<u64>, u64) {
    (info.coord.clone(), info.addr)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_addrmap(table: &ChunkTable) -> HashMap<Vec<u64>, u64> {
    H5D__create_piece_file_map_all(table)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_delete(table: &mut ChunkTable, coord: &[u64]) {
    table.chunks.remove(coord);
    table.data.remove(coord);
}

#[allow(non_snake_case)]
pub fn H5D__chunk_update_cache(table: &mut ChunkTable, coord: Vec<u64>, data: Vec<u8>) {
    table.data.insert(coord, data);
}

#[allow(non_snake_case)]
pub fn H5D__chunk_copy_cb(info: &ChunkInfo) -> ChunkInfo {
    info.clone()
}

#[allow(non_snake_case)]
pub fn H5D__chunk_copy(table: &ChunkTable) -> ChunkTable {
    table.clone()
}

#[allow(non_snake_case)]
pub fn H5D__chunk_stats(table: &ChunkTable) -> (usize, u64, u64) {
    (table.chunks.len(), table.cache_hits, table.cache_misses)
}

#[allow(non_snake_case)]
pub fn H5D__nonexistent_readvv_cb(len: usize) -> Vec<u8> {
    vec![0; len]
}

#[allow(non_snake_case)]
pub fn H5D__nonexistent_readvv(len: usize) -> Vec<u8> {
    H5D__nonexistent_readvv_cb(len)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_file_alloc(table: &mut ChunkTable, coord: Vec<u64>, size: usize) -> u64 {
    let addr = table.chunks.len() as u64;
    let info = ChunkInfo {
        coord: coord.clone(),
        addr,
        size,
        filter_mask: 0,
    };
    table.chunks.insert(coord, info);
    addr
}

#[allow(non_snake_case)]
pub fn H5D__chunk_format_convert_cb(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D__chunk_format_convert(table: &mut ChunkTable) {
    for data in table.data.values_mut() {
        data.shrink_to_fit();
    }
}

#[allow(non_snake_case)]
pub fn H5D__chunk_index_empty_cb(table: &ChunkTable) -> bool {
    table.chunks.is_empty()
}

#[allow(non_snake_case)]
pub fn H5D__get_num_chunks_cb(table: &ChunkTable) -> usize {
    table.chunks.len()
}

#[allow(non_snake_case)]
pub fn H5D__get_num_chunks(table: &ChunkTable) -> usize {
    H5D__get_num_chunks_cb(table)
}

#[allow(non_snake_case)]
pub fn H5D__get_chunk_info_cb<'a>(table: &'a ChunkTable, coord: &[u64]) -> Option<&'a ChunkInfo> {
    table.chunks.get(coord)
}

#[allow(non_snake_case)]
pub fn H5D__get_chunk_info<'a>(table: &'a ChunkTable, coord: &[u64]) -> Option<&'a ChunkInfo> {
    H5D__get_chunk_info_cb(table, coord)
}

#[allow(non_snake_case)]
pub fn H5D__get_chunk_info_by_coord<'a>(
    table: &'a ChunkTable,
    coord: &[u64],
) -> Option<&'a ChunkInfo> {
    H5D__get_chunk_info(table, coord)
}

#[allow(non_snake_case)]
pub fn H5D__chunk_get_offset_copy(coord: &[u64]) -> Vec<u64> {
    coord.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D__read(dataset: &DatasetApi) -> &[u8] {
    &dataset.raw
}

#[allow(non_snake_case)]
pub fn H5D__write(dataset: &mut DatasetApi, data: &[u8]) {
    H5D__write_api_common(dataset, data);
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkIndexState {
    pub open: bool,
    pub space_allocated: bool,
    pub entries: HashMap<Vec<u64>, ChunkInfo>,
    pub metadata_loaded: bool,
}

fn chunk_index_insert(index: &mut ChunkIndexState, coord: Vec<u64>, addr: u64, size: usize) {
    index.open = true;
    index.space_allocated = true;
    index.entries.insert(
        coord.clone(),
        ChunkInfo {
            coord,
            addr,
            size,
            filter_mask: 0,
        },
    );
}

fn chunk_index_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
    index.entries.get(coord).map(|info| info.addr)
}

fn chunk_index_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
    let removed = index.entries.remove(coord);
    index.space_allocated = !index.entries.is_empty();
    removed
}

fn chunk_index_dump(kind: &str, index: &ChunkIndexState) -> String {
    format!(
        "{kind}(open={}, allocated={}, entries={})",
        index.open,
        index.space_allocated,
        index.entries.len()
    )
}

macro_rules! chunk_index_family {
    (
        $crt_context:ident,
        $dst_context:ident,
        $fill:ident,
        $encode:ident,
        $decode:ident,
        $debug:ident,
        $idx_depend:ident,
        $idx_init:ident,
        $idx_create:ident,
        $idx_open:ident,
        $idx_close:ident,
        $idx_is_open:ident,
        $idx_is_space_alloc:ident,
        $idx_insert:ident,
        $idx_get_addr:ident,
        $idx_load_metadata:ident,
        $idx_iterate_cb:ident,
        $idx_iterate:ident,
        $idx_remove:ident,
        $idx_delete_cb:ident,
        $idx_delete:ident,
        $idx_copy_setup:ident,
        $idx_size:ident,
        $idx_reset:ident,
        $idx_dump:ident,
        $idx_dest:ident,
        $kind:literal
    ) => {
        #[allow(non_snake_case)]
        pub fn $crt_context() -> ChunkIndexState {
            ChunkIndexState::default()
        }

        #[allow(non_snake_case)]
        pub fn $dst_context(index: &mut ChunkIndexState) {
            index.entries.clear();
        }

        #[allow(non_snake_case)]
        pub fn $fill(size: usize) -> Vec<u8> {
            vec![0; size]
        }

        #[allow(non_snake_case)]
        pub fn $encode(index: &ChunkIndexState) -> Vec<u8> {
            (index.entries.len() as u64).to_le_bytes().to_vec()
        }

        #[allow(non_snake_case)]
        pub fn $decode(bytes: &[u8]) -> ChunkIndexState {
            let mut index = ChunkIndexState::default();
            index.metadata_loaded = bytes.len() >= 8;
            index
        }

        #[allow(non_snake_case)]
        pub fn $debug(index: &ChunkIndexState) -> String {
            chunk_index_dump($kind, index)
        }

        #[allow(non_snake_case)]
        pub fn $idx_depend(index: &ChunkIndexState) -> usize {
            index.entries.len()
        }

        #[allow(non_snake_case)]
        pub fn $idx_init() -> ChunkIndexState {
            ChunkIndexState::default()
        }

        #[allow(non_snake_case)]
        pub fn $idx_create(index: &mut ChunkIndexState) {
            index.open = true;
        }

        #[allow(non_snake_case)]
        pub fn $idx_open(index: &mut ChunkIndexState) {
            index.open = true;
        }

        #[allow(non_snake_case)]
        pub fn $idx_close(index: &mut ChunkIndexState) {
            index.open = false;
        }

        #[allow(non_snake_case)]
        pub fn $idx_is_open(index: &ChunkIndexState) -> bool {
            index.open
        }

        #[allow(non_snake_case)]
        pub fn $idx_is_space_alloc(index: &ChunkIndexState) -> bool {
            index.space_allocated
        }

        #[allow(non_snake_case)]
        pub fn $idx_insert(index: &mut ChunkIndexState, coord: Vec<u64>, addr: u64, size: usize) {
            chunk_index_insert(index, coord, addr, size);
        }

        #[allow(non_snake_case)]
        pub fn $idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
            chunk_index_get_addr(index, coord)
        }

        #[allow(non_snake_case)]
        pub fn $idx_load_metadata(index: &mut ChunkIndexState) {
            index.metadata_loaded = true;
        }

        #[allow(non_snake_case)]
        pub fn $idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
            info.clone()
        }

        #[allow(non_snake_case)]
        pub fn $idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
            index.entries.values().cloned().collect()
        }

        #[allow(non_snake_case)]
        pub fn $idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
            chunk_index_remove(index, coord)
        }

        #[allow(non_snake_case)]
        pub fn $idx_delete_cb(info: &ChunkInfo) -> ChunkInfo {
            info.clone()
        }

        #[allow(non_snake_case)]
        pub fn $idx_delete(index: &mut ChunkIndexState) {
            index.entries.clear();
            index.space_allocated = false;
        }

        #[allow(non_snake_case)]
        pub fn $idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
            index.clone()
        }

        #[allow(non_snake_case)]
        pub fn $idx_size(index: &ChunkIndexState) -> usize {
            index.entries.len()
        }

        #[allow(non_snake_case)]
        pub fn $idx_reset(index: &mut ChunkIndexState) {
            *index = ChunkIndexState::default();
        }

        #[allow(non_snake_case)]
        pub fn $idx_dump(index: &ChunkIndexState) -> String {
            chunk_index_dump($kind, index)
        }

        #[allow(non_snake_case)]
        pub fn $idx_dest(index: &mut ChunkIndexState) {
            index.entries.clear();
            index.open = false;
            index.space_allocated = false;
        }
    };
}

chunk_index_family!(
    H5D__earray_crt_context,
    H5D__earray_dst_context,
    H5D__earray_fill,
    H5D__earray_encode,
    H5D__earray_filt_decode,
    H5D__earray_debug,
    H5D__earray_idx_depend,
    H5D__earray_idx_init,
    H5D__earray_idx_create,
    H5D__earray_idx_open,
    H5D__earray_idx_close,
    H5D__earray_idx_is_open,
    H5D__earray_idx_is_space_alloc,
    H5D__earray_idx_insert,
    H5D__earray_idx_get_addr,
    H5D__earray_idx_load_metadata,
    H5D__earray_idx_iterate_cb,
    H5D__earray_idx_iterate,
    H5D__earray_idx_remove,
    H5D__earray_idx_delete_cb,
    H5D__earray_idx_delete,
    H5D__earray_idx_copy_setup,
    H5D__earray_idx_resize,
    H5D__earray_idx_reset,
    H5D__earray_idx_dump,
    H5D__earray_idx_dest,
    "earray"
);

#[allow(non_snake_case)]
pub fn H5D__earray_filt_fill(size: usize) -> Vec<u8> {
    H5D__earray_fill(size)
}

#[allow(non_snake_case)]
pub fn H5D__earray_filt_encode(index: &ChunkIndexState) -> Vec<u8> {
    H5D__earray_encode(index)
}

#[allow(non_snake_case)]
pub fn H5D__earray_filt_debug(index: &ChunkIndexState) -> String {
    H5D__earray_debug(index)
}

#[allow(non_snake_case)]
pub fn H5D__earray_crt_dbg_context() -> ChunkIndexState {
    H5D__earray_crt_context()
}

#[allow(non_snake_case)]
pub fn H5D__earray_filt_crt_dbg_context() -> ChunkIndexState {
    H5D__earray_crt_context()
}

#[allow(non_snake_case)]
pub fn H5D__earray_dst_dbg_context(index: &mut ChunkIndexState) {
    H5D__earray_dst_context(index);
}

chunk_index_family!(
    H5D__farray_crt_context,
    H5D__farray_dst_context,
    H5D__farray_fill,
    H5D__farray_encode,
    H5D__farray_decode,
    H5D__farray_debug,
    H5D__farray_idx_depend,
    H5D__farray_idx_init,
    H5D__farray_idx_create,
    H5D__farray_idx_open,
    H5D__farray_idx_close,
    H5D__farray_idx_is_open,
    H5D__farray_idx_is_space_alloc,
    H5D__farray_idx_insert,
    H5D__farray_idx_get_addr,
    H5D__farray_idx_load_metadata,
    H5D__farray_idx_iterate_cb,
    H5D__farray_idx_iterate,
    H5D__farray_idx_remove,
    H5D__farray_idx_delete_cb,
    H5D__farray_idx_delete,
    H5D__farray_idx_copy_setup,
    H5D__farray_idx_size,
    H5D__farray_idx_reset,
    H5D__farray_idx_dump,
    H5D__farray_idx_dest,
    "farray"
);

#[allow(non_snake_case)]
pub fn H5D__farray_crt_dbg_context() -> ChunkIndexState {
    H5D__farray_crt_context()
}

#[allow(non_snake_case)]
pub fn H5D__farray_dst_dbg_context(index: &mut ChunkIndexState) {
    H5D__farray_dst_context(index);
}

#[allow(non_snake_case)]
pub fn H5D__farray_filt_fill(size: usize) -> Vec<u8> {
    H5D__farray_fill(size)
}

#[allow(non_snake_case)]
pub fn H5D__farray_filt_encode(index: &ChunkIndexState) -> Vec<u8> {
    H5D__farray_encode(index)
}

#[allow(non_snake_case)]
pub fn H5D__farray_filt_decode(bytes: &[u8]) -> ChunkIndexState {
    H5D__farray_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5D__farray_filt_debug(index: &ChunkIndexState) -> String {
    H5D__farray_debug(index)
}

#[allow(non_snake_case)]
pub fn H5D__farray_filt_crt_dbg_context() -> ChunkIndexState {
    H5D__farray_crt_context()
}

chunk_index_family!(
    H5D__bt2_crt_context,
    H5D__bt2_dst_context,
    H5D__bt2_store,
    H5D__bt2_unfilt_encode,
    H5D__bt2_unfilt_decode,
    H5D__bt2_unfilt_debug,
    H5D__btree2_idx_depend,
    H5D__bt2_idx_init,
    H5D__bt2_idx_create,
    H5D__bt2_idx_open,
    H5D__bt2_idx_close,
    H5D__bt2_idx_is_open,
    H5D__bt2_idx_is_space_alloc,
    H5D__bt2_idx_insert,
    H5D__bt2_idx_get_addr,
    H5D__bt2_idx_load_metadata,
    H5D__bt2_idx_iterate_cb,
    H5D__bt2_idx_iterate,
    H5D__bt2_idx_remove,
    H5D__bt2_remove_cb,
    H5D__bt2_idx_delete,
    H5D__bt2_idx_copy_setup,
    H5D__bt2_idx_size,
    H5D__bt2_idx_reset,
    H5D__bt2_idx_dump,
    H5D__bt2_idx_dest,
    "bt2"
);

#[allow(non_snake_case)]
pub fn H5D__bt2_compare(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    H5D__chunk_cmp_coll_fill_info(left, right)
}

#[allow(non_snake_case)]
pub fn H5D__bt2_filt_encode(index: &ChunkIndexState) -> Vec<u8> {
    H5D__bt2_unfilt_encode(index)
}

#[allow(non_snake_case)]
pub fn H5D__bt2_filt_decode(bytes: &[u8]) -> ChunkIndexState {
    H5D__bt2_unfilt_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5D__bt2_filt_debug(index: &ChunkIndexState) -> String {
    H5D__bt2_unfilt_debug(index)
}

#[allow(non_snake_case)]
pub fn H5D__bt2_mod_cb(info: &mut ChunkInfo, addr: u64, size: usize) {
    H5D__chunk_set_info(info, addr, size);
}

#[allow(non_snake_case)]
pub fn H5D__bt2_found_cb(info: &ChunkInfo) -> ChunkInfo {
    info.clone()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetIoInfo {
    pub element_size: usize,
    pub file_selection: Vec<u64>,
    pub memory_selection: Vec<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetTypeInfo {
    pub src_size: usize,
    pub dst_size: usize,
    pub conversion_needed: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContiguousStorage {
    pub addr: Option<u64>,
    pub data: Vec<u8>,
    pub cached: bool,
    pub allocated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalFileList {
    pub files: Vec<String>,
    pub allocated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FillState {
    pub value: Vec<u8>,
    pub initialized: bool,
}

#[allow(non_snake_case)]
pub fn H5D__ioinfo_init(element_size: usize) -> DatasetIoInfo {
    DatasetIoInfo {
        element_size,
        file_selection: Vec::new(),
        memory_selection: Vec::new(),
    }
}

#[allow(non_snake_case)]
pub fn H5D__dset_ioinfo_init(dataset: &DatasetApi, element_size: usize) -> DatasetIoInfo {
    DatasetIoInfo {
        element_size,
        file_selection: dataset.extent.clone(),
        memory_selection: dataset.extent.clone(),
    }
}

#[allow(non_snake_case)]
pub fn H5D__typeinfo_init(src_size: usize, dst_size: usize) -> DatasetTypeInfo {
    DatasetTypeInfo {
        src_size,
        dst_size,
        conversion_needed: src_size != dst_size,
    }
}

#[allow(non_snake_case)]
pub fn H5D__typeinfo_init_phase2(info: &mut DatasetTypeInfo) {
    info.conversion_needed = info.src_size != info.dst_size;
}

#[allow(non_snake_case)]
pub fn H5D__ioinfo_adjust(info: &mut DatasetIoInfo, extent: &[u64]) {
    info.file_selection = extent.to_vec();
    info.memory_selection = extent.to_vec();
}

#[allow(non_snake_case)]
pub fn H5D__typeinfo_init_phase3(info: &mut DatasetTypeInfo) {
    H5D__typeinfo_init_phase2(info);
}

#[allow(non_snake_case)]
pub fn H5D__typeinfo_term(info: &mut DatasetTypeInfo) {
    *info = DatasetTypeInfo::default();
}

#[allow(non_snake_case)]
pub fn H5D__layout_version_test(version: u8) -> bool {
    version <= 4
}

#[allow(non_snake_case)]
pub fn H5D__layout_contig_size_test(size: u64) -> bool {
    size > 0
}

#[allow(non_snake_case)]
pub fn H5Ddebug(dataset: &DatasetApi) -> String {
    format!(
        "dataset(name={:?}, extent={:?}, bytes={})",
        dataset.name,
        dataset.extent,
        dataset.raw.len()
    )
}

#[allow(non_snake_case)]
pub fn H5D__mpio_debug_init() -> Result<()> {
    Err(Error::Unsupported(
        "MPI dataset I/O is intentionally unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_opt_possible() -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5D__mpio_select_write() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective dataset writes are intentionally unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_get_sum_chunk(chunks: &[ChunkInfo]) -> usize {
    chunks.iter().map(|chunk| chunk.size).sum()
}

#[allow(non_snake_case)]
pub fn H5D__mpio_get_sum_chunk_dset(table: &ChunkTable) -> usize {
    table.chunks.values().map(|chunk| chunk.size).sum()
}

#[allow(non_snake_case)]
pub fn H5D__piece_io() -> Result<()> {
    Err(Error::Unsupported(
        "piece I/O is only used by MPI collective dataset paths".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__link_chunk_filtered_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk I/O is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__multi_chunk_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective chunk I/O is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__multi_chunk_filtered_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk I/O is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__inter_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI inter-collective dataset I/O is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__final_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective dataset I/O is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__cmp_piece_addr(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    left.addr.cmp(&right.addr)
}

#[allow(non_snake_case)]
pub fn H5D__cmp_filtered_collective_io_info_entry(
    left: &ChunkInfo,
    right: &ChunkInfo,
) -> std::cmp::Ordering {
    H5D__chunk_cmp_coll_fill_info(left, right)
}

#[allow(non_snake_case)]
pub fn H5D__cmp_chunk_redistribute_info(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    left.coord.cmp(&right.coord)
}

#[allow(non_snake_case)]
pub fn H5D__cmp_chunk_redistribute_info_orig_owner(
    left: &ChunkInfo,
    right: &ChunkInfo,
) -> std::cmp::Ordering {
    left.addr
        .cmp(&right.addr)
        .then_with(|| left.coord.cmp(&right.coord))
}

#[allow(non_snake_case)]
pub fn H5D__obtain_mpio_mode() -> Result<()> {
    Err(Error::Unsupported(
        "MPI dataset transfer mode is unavailable".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_io_setup() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk I/O is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_redistribute_shared_chunks_int() -> Result<()> {
    Err(Error::Unsupported(
        "MPI shared chunk redistribution is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_share_chunk_modification_data() -> Result<()> {
    Err(Error::Unsupported(
        "MPI chunk modification sharing is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_read() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk read is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_update() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk update is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_reallocate() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk reallocate is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_reinsert() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk reinsert is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_get_chunk_redistribute_info_types() -> Result<()> {
    Err(Error::Unsupported(
        "MPI datatype construction is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_get_chunk_alloc_info_types() -> Result<()> {
    Err(Error::Unsupported(
        "MPI datatype construction is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_get_chunk_insert_info_types() -> Result<()> {
    Err(Error::Unsupported(
        "MPI datatype construction is unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__mpio_dump_collective_filtered_chunk_list(chunks: &[ChunkInfo]) -> String {
    format!("{} collective chunks", chunks.len())
}

#[allow(non_snake_case)]
pub fn H5D__scatter_file(src: &[u8], spans: &[(usize, usize)]) -> Vec<Vec<u8>> {
    spans
        .iter()
        .filter_map(|&(offset, len)| {
            src.get(offset..offset.saturating_add(len))
                .map(<[u8]>::to_vec)
        })
        .collect()
}

#[allow(non_snake_case)]
pub fn H5D__gather_file(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

#[allow(non_snake_case)]
pub fn H5D__scatter_mem(src: &[u8], spans: &[(usize, usize)]) -> Vec<Vec<u8>> {
    H5D__scatter_file(src, spans)
}

#[allow(non_snake_case)]
pub fn H5D__gather_mem(parts: &[Vec<u8>]) -> Vec<u8> {
    H5D__gather_file(parts)
}

#[allow(non_snake_case)]
pub fn H5D__scatgath_read_select(src: &[u8], spans: &[(usize, usize)]) -> Vec<Vec<u8>> {
    H5D__scatter_file(src, spans)
}

#[allow(non_snake_case)]
pub fn H5D__scatgath_write_select(parts: &[Vec<u8>]) -> Vec<u8> {
    H5D__gather_file(parts)
}

#[allow(non_snake_case)]
pub fn H5D__compound_opt_read(src: &[u8]) -> Vec<u8> {
    src.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D__efl_construct(files: Vec<String>) -> ExternalFileList {
    ExternalFileList {
        files,
        allocated: true,
    }
}

#[allow(non_snake_case)]
pub fn H5D__efl_init(efl: &mut ExternalFileList) {
    efl.allocated = true;
}

#[allow(non_snake_case)]
pub fn H5D__efl_is_space_alloc(efl: &ExternalFileList) -> bool {
    efl.allocated
}

#[allow(non_snake_case)]
pub fn H5D__efl_io_init(_efl: &mut ExternalFileList) {}

#[allow(non_snake_case)]
pub fn H5D__efl_read(_efl: &ExternalFileList, _offset: u64, _buf: &mut [u8]) -> Result<()> {
    Err(Error::Unsupported(
        "external raw dataset file I/O is handled by high-level dataset storage".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__efl_write(_efl: &ExternalFileList, _offset: u64, _data: &[u8]) -> Result<()> {
    Err(Error::Unsupported(
        "external raw dataset writes are handled by high-level dataset storage".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5D__efl_readvv_cb(len: usize) -> Vec<u8> {
    vec![0; len]
}

#[allow(non_snake_case)]
pub fn H5D__efl_readvv(efl: &ExternalFileList, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0; len];
    H5D__efl_read(efl, offset, &mut buf)?;
    Ok(buf)
}

#[allow(non_snake_case)]
pub fn H5D__efl_writevv_cb(data: &[u8]) -> usize {
    data.len()
}

#[allow(non_snake_case)]
pub fn H5D__efl_writevv(efl: &ExternalFileList, offset: u64, data: &[u8]) -> Result<()> {
    H5D__efl_write(efl, offset, data)
}

#[allow(non_snake_case)]
pub fn H5D__efl_bh_info(efl: &ExternalFileList) -> usize {
    efl.files.len()
}

#[allow(non_snake_case)]
pub fn H5D__layout_set_io_ops(_dataset: &mut DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5D__layout_meta_size(dataset: &DatasetApi) -> usize {
    dataset.extent.len() * std::mem::size_of::<u64>()
}

#[allow(non_snake_case)]
pub fn H5D__layout_oh_create(dataset: &DatasetApi) -> Vec<u8> {
    dataset
        .extent
        .iter()
        .flat_map(|dim| dim.to_le_bytes())
        .collect()
}

#[allow(non_snake_case)]
pub fn H5D__layout_oh_write(dataset: &DatasetApi) -> Vec<u8> {
    H5D__layout_oh_create(dataset)
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_create() -> ChunkIndexState {
    ChunkIndexState::default()
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_close(_index: &mut ChunkIndexState) {}

#[allow(non_snake_case)]
pub fn H5D__none_idx_is_open(_index: &ChunkIndexState) -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_is_space_alloc(_index: &ChunkIndexState) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_get_addr(_index: &ChunkIndexState, _coord: &[u64]) -> Option<u64> {
    None
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_load_metadata(_index: &mut ChunkIndexState) {}

#[allow(non_snake_case)]
pub fn H5D__none_idx_iterate(_index: &ChunkIndexState) -> Vec<ChunkInfo> {
    Vec::new()
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_remove(_index: &mut ChunkIndexState, _coord: &[u64]) -> Option<ChunkInfo> {
    None
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_delete(_index: &mut ChunkIndexState) {}

#[allow(non_snake_case)]
pub fn H5D__none_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
    index.clone()
}

#[allow(non_snake_case)]
pub fn H5D__none_idx_reset(_index: &mut ChunkIndexState) {}

#[allow(non_snake_case)]
pub fn H5D__none_idx_dump(_index: &ChunkIndexState) -> String {
    "none_idx".into()
}

#[allow(non_snake_case)]
pub fn H5D_init() -> bool {
    H5D__init_package()
}

#[allow(non_snake_case)]
pub fn H5D__init_package() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5D_top_term_package() {}

#[allow(non_snake_case)]
pub fn H5D_term_package() {}

#[allow(non_snake_case)]
pub fn H5D__close_cb(_dataset: DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5D__get_space_status(dataset: &DatasetApi) -> bool {
    !dataset.raw.is_empty()
}

#[allow(non_snake_case)]
pub fn H5D__new(name: Option<String>, extent: Vec<u64>) -> DatasetApi {
    H5D__create_api_common(name, extent)
}

#[allow(non_snake_case)]
pub fn H5D__init_type(element_size: usize) -> DatasetTypeInfo {
    H5D__typeinfo_init(element_size, element_size)
}

#[allow(non_snake_case)]
pub fn H5D__cache_dataspace_info(dataset: &DatasetApi) -> Vec<u64> {
    dataset.extent.clone()
}

#[allow(non_snake_case)]
pub fn H5D__init_space(dataset: &mut DatasetApi, extent: Vec<u64>) {
    dataset.extent = extent;
}

#[allow(non_snake_case)]
pub fn H5D__use_minimized_dset_headers(dataset: &DatasetApi) -> bool {
    dataset.raw.len() < 64 * 1024
}

#[allow(non_snake_case)]
pub fn H5D__calculate_minimum_header_size(dataset: &DatasetApi) -> usize {
    H5D__layout_meta_size(dataset)
}

#[allow(non_snake_case)]
pub fn H5D__prepare_minimized_oh(dataset: &DatasetApi) -> Vec<u8> {
    H5D__layout_oh_create(dataset)
}

#[allow(non_snake_case)]
pub fn H5D__update_oh_info(dataset: &mut DatasetApi, extent: Vec<u64>) {
    dataset.extent = extent;
}

#[allow(non_snake_case)]
pub fn H5D__build_file_prefix(path: &str) -> String {
    path.rsplit_once('/')
        .map_or(String::new(), |(prefix, _)| prefix.to_string())
}

#[allow(non_snake_case)]
pub fn H5D__create(name: Option<String>, extent: Vec<u64>) -> DatasetApi {
    H5D__create_api_common(name, extent)
}

#[allow(non_snake_case)]
pub fn H5D_open(dataset: &DatasetApi) -> DatasetApi {
    dataset.clone()
}

#[allow(non_snake_case)]
pub fn H5D__append_flush_setup(_dataset: &mut DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5D__open_oid(dataset: &DatasetApi) -> DatasetApi {
    dataset.clone()
}

#[allow(non_snake_case)]
pub fn H5D_close(dataset: DatasetApi) {
    H5Dclose(dataset);
}

#[allow(non_snake_case)]
pub fn H5D_mult_refresh_close(_datasets: &mut [DatasetApi]) {}

#[allow(non_snake_case)]
pub fn H5D_mult_refresh_reopen(datasets: &[DatasetApi]) -> Vec<DatasetApi> {
    datasets.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D_oloc(dataset: &DatasetApi) -> Option<&str> {
    dataset.name.as_deref()
}

#[allow(non_snake_case)]
pub fn H5D_nameof(dataset: &DatasetApi) -> Option<&str> {
    dataset.name.as_deref()
}

#[allow(non_snake_case)]
pub fn H5D__alloc_storage(dataset: &mut DatasetApi, size: usize) {
    dataset.raw.resize(size, 0);
}

#[allow(non_snake_case)]
pub fn H5D__init_storage(dataset: &mut DatasetApi) {
    dataset.raw.clear();
}

#[allow(non_snake_case)]
pub fn H5D__get_storage_size(dataset: &DatasetApi) -> usize {
    dataset.raw.len()
}

#[allow(non_snake_case)]
pub fn H5D__get_offset(_dataset: &DatasetApi) -> Option<u64> {
    Some(0)
}

#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_alloc(size: usize) -> usize {
    size
}

#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_cb(value: &[u8]) -> usize {
    value.len()
}

#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size(values: &[Vec<u8>]) -> usize {
    values.iter().map(Vec::len).sum()
}

#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_gen_cb(value: &[u8]) -> usize {
    H5D__vlen_get_buf_size_cb(value)
}

#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_gen(values: &[Vec<u8>]) -> usize {
    H5D__vlen_get_buf_size(values)
}

#[allow(non_snake_case)]
pub fn H5D__flush_sieve_buf(_dataset: &mut DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5D__flush_real(_dataset: &mut DatasetApi) {}

#[allow(non_snake_case)]
pub fn H5D__flush(dataset: &mut DatasetApi) {
    H5D__flush_real(dataset);
}

#[allow(non_snake_case)]
pub fn H5D__format_convert(dataset: &mut DatasetApi) {
    H5Dformat_convert(dataset);
}

#[allow(non_snake_case)]
pub fn H5D__mark(dataset: &mut DatasetApi, marked: bool) {
    if marked {
        dataset.raw.shrink_to_fit();
    }
}

#[allow(non_snake_case)]
pub fn H5D__flush_all_cb(dataset: &mut DatasetApi) {
    H5D__flush(dataset);
}

#[allow(non_snake_case)]
pub fn H5D_flush_all(datasets: &mut [DatasetApi]) {
    for dataset in datasets {
        H5D__flush_all_cb(dataset);
    }
}

#[allow(non_snake_case)]
pub fn H5D_get_create_plist(_dataset: &DatasetApi) -> HashMap<String, String> {
    HashMap::new()
}

#[allow(non_snake_case)]
pub fn H5D_get_access_plist(_dataset: &DatasetApi) -> HashMap<String, String> {
    HashMap::new()
}

#[allow(non_snake_case)]
pub fn H5D__get_space(dataset: &DatasetApi) -> &[u64] {
    &dataset.extent
}

#[allow(non_snake_case)]
pub fn H5D__get_type(element_size: usize) -> DatasetTypeInfo {
    H5D__init_type(element_size)
}

#[allow(non_snake_case)]
pub fn H5D__contig_alloc(storage: &mut ContiguousStorage, size: usize) {
    storage.data.resize(size, 0);
    storage.allocated = true;
}

#[allow(non_snake_case)]
pub fn H5D__contig_delete(storage: &mut ContiguousStorage) {
    storage.data.clear();
    storage.allocated = false;
}

#[allow(non_snake_case)]
pub fn H5D__contig_check(storage: &ContiguousStorage) -> bool {
    storage.allocated
}

#[allow(non_snake_case)]
pub fn H5D__contig_construct(data: Vec<u8>) -> ContiguousStorage {
    ContiguousStorage {
        addr: Some(0),
        data,
        cached: false,
        allocated: true,
    }
}

#[allow(non_snake_case)]
pub fn H5D__contig_init(storage: &mut ContiguousStorage) {
    storage.allocated = true;
}

#[allow(non_snake_case)]
pub fn H5D__contig_is_space_alloc(storage: &ContiguousStorage) -> bool {
    storage.allocated
}

#[allow(non_snake_case)]
pub fn H5D__contig_is_data_cached(storage: &ContiguousStorage) -> bool {
    storage.cached
}

#[allow(non_snake_case)]
pub fn H5D__contig_io_init(storage: &mut ContiguousStorage) {
    storage.cached = true;
}

#[allow(non_snake_case)]
pub fn H5D__contig_mdio_init(_storage: &mut ContiguousStorage) {}

#[allow(non_snake_case)]
pub fn H5D__contig_may_use_select_io(spans: &[(usize, usize)]) -> bool {
    spans.len() > 1
}

#[allow(non_snake_case)]
pub fn H5D__contig_write_one(
    storage: &mut ContiguousStorage,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    H5D__contig_writevv(storage, &[(offset, data.to_vec())])
}

#[allow(non_snake_case)]
pub fn H5D__contig_readvv_sieve_cb(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D__contig_readvv_cb(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5D__contig_readvv(
    storage: &ContiguousStorage,
    spans: &[(usize, usize)],
) -> Result<Vec<Vec<u8>>> {
    Ok(spans
        .iter()
        .map(|&(offset, len)| {
            storage
                .data
                .get(offset..offset.saturating_add(len))
                .unwrap_or(&[])
                .to_vec()
        })
        .collect())
}

#[allow(non_snake_case)]
pub fn H5D__contig_writevv_sieve_cb(data: &[u8]) -> usize {
    data.len()
}

#[allow(non_snake_case)]
pub fn H5D__contig_writevv_cb(data: &[u8]) -> usize {
    data.len()
}

#[allow(non_snake_case)]
pub fn H5D__contig_writevv(
    storage: &mut ContiguousStorage,
    spans: &[(usize, Vec<u8>)],
) -> Result<()> {
    for (offset, data) in spans {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::InvalidFormat("contiguous write overflow".into()))?;
        if storage.data.len() < end {
            storage.data.resize(end, 0);
        }
        storage.data[*offset..end].copy_from_slice(data);
    }
    storage.allocated = true;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5D__contig_flush(_storage: &mut ContiguousStorage) {}

#[allow(non_snake_case)]
pub fn H5D__contig_io_term(storage: &mut ContiguousStorage) {
    storage.cached = false;
}

#[allow(non_snake_case)]
pub fn H5D__contig_copy(storage: &ContiguousStorage) -> ContiguousStorage {
    storage.clone()
}

#[allow(non_snake_case)]
pub fn H5Dcreate1(extent: Vec<u64>) -> DatasetApi {
    H5Dcreate_anon(extent)
}

#[allow(non_snake_case)]
pub fn H5Dopen1(dataset: &DatasetApi) -> DatasetApi {
    H5D_open(dataset)
}

#[allow(non_snake_case)]
pub fn H5Dextend(dataset: &mut DatasetApi, extent: Vec<u64>) {
    dataset.extent = extent;
}

#[allow(non_snake_case)]
pub fn H5Dvlen_reclaim(values: &mut Vec<Vec<u8>>) {
    values.clear();
}

#[allow(non_snake_case)]
pub fn H5Dread_chunk1(dataset: &DatasetApi, offset: usize, len: usize) -> Result<Vec<u8>> {
    H5Dread_chunk2(dataset, offset, len)
}

#[allow(non_snake_case)]
pub fn H5D__fill(dst: &mut [u8], fill: &[u8]) {
    if fill.is_empty() {
        return;
    }
    for chunk in dst.chunks_mut(fill.len()) {
        let n = chunk.len();
        chunk.copy_from_slice(&fill[..n]);
    }
}

#[allow(non_snake_case)]
pub fn H5D__fill_init(value: Vec<u8>) -> FillState {
    FillState {
        value,
        initialized: true,
    }
}

#[allow(non_snake_case)]
pub fn H5D__fill_refill_vl(state: &FillState, dst: &mut Vec<u8>) {
    dst.clear();
    dst.extend_from_slice(&state.value);
}

#[allow(non_snake_case)]
pub fn H5D__fill_release(state: &mut FillState) {
    state.value.clear();
    state.initialized = false;
}

#[allow(non_snake_case)]
pub fn H5D__fill_term(state: &mut FillState) {
    H5D__fill_release(state);
}

pub mod explicit_index_wrappers {
    use super::*;

    #[allow(non_snake_case)]
    pub fn H5D__earray_crt_context() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_dst_context(index: &mut ChunkIndexState) {
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_fill(size: usize) -> Vec<u8> {
        vec![0; size]
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_encode(index: &ChunkIndexState) -> Vec<u8> {
        (index.entries.len() as u64).to_le_bytes().to_vec()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_debug(index: &ChunkIndexState) -> String {
        chunk_index_dump("earray", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_filt_decode(_bytes: &[u8]) -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_depend(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_create(index: &mut ChunkIndexState) {
        index.open = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_open(index: &mut ChunkIndexState) {
        index.open = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_insert(
        index: &mut ChunkIndexState,
        coord: Vec<u64>,
        addr: u64,
        size: usize,
    ) {
        chunk_index_insert(index, coord, addr, size);
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_resize(index: &mut ChunkIndexState, additional: usize) {
        index.entries.reserve(additional);
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        index.entries.values().cloned().collect()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_delete_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_crt_context() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_dst_context(index: &mut ChunkIndexState) {
        index.entries.clear();
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_store(index: &ChunkIndexState) -> Vec<u8> {
        (index.entries.len() as u64).to_le_bytes().to_vec()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_unfilt_encode(index: &ChunkIndexState) -> Vec<u8> {
        H5D__bt2_store(index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_unfilt_decode(_bytes: &[u8]) -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_unfilt_debug(index: &ChunkIndexState) -> String {
        chunk_index_dump("bt2", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree2_idx_depend(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_create(index: &mut ChunkIndexState) {
        index.open = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_open(index: &mut ChunkIndexState) {
        index.open = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_insert(
        index: &mut ChunkIndexState,
        coord: Vec<u64>,
        addr: u64,
        size: usize,
    ) {
        chunk_index_insert(index, coord, addr, size);
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        index.entries.values().cloned().collect()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_remove_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_size(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_dump(index: &ChunkIndexState) -> String {
        chunk_index_dump("bt2", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_crt_context() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_dst_context(index: &mut ChunkIndexState) {
        index.entries.clear();
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_fill(size: usize) -> Vec<u8> {
        vec![0; size]
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_encode(index: &ChunkIndexState) -> Vec<u8> {
        (index.entries.len() as u64).to_le_bytes().to_vec()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_decode(_bytes: &[u8]) -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_debug(index: &ChunkIndexState) -> String {
        chunk_index_dump("farray", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_depend(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_create(index: &mut ChunkIndexState) {
        index.open = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_open(index: &mut ChunkIndexState) {
        index.open = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_insert(
        index: &mut ChunkIndexState,
        coord: Vec<u64>,
        addr: u64,
        size: usize,
    ) {
        chunk_index_insert(index, coord, addr, size);
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        index.entries.values().cloned().collect()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_delete_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_size(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_dump(index: &ChunkIndexState) -> String {
        chunk_index_dump("farray", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_get_shared() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_cmp2(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
        H5D__chunk_cmp_coll_fill_info(left, right)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_cmp3(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
        H5D__chunk_cmp_coll_fill_info(left, right)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_decode_key(_bytes: &[u8]) -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_encode_key(index: &ChunkIndexState) -> Vec<u8> {
        (index.entries.len() as u64).to_le_bytes().to_vec()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_debug_key(index: &ChunkIndexState) -> String {
        chunk_index_dump("btree", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_shared_free(index: &mut ChunkIndexState) {
        index.entries.clear();
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_shared_create() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        index.entries.values().cloned().collect()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_copy_shutdown(index: &mut ChunkIndexState) {
        index.open = false;
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_size(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_dump(index: &ChunkIndexState) -> String {
        chunk_index_dump("btree", index)
    }

    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }
}

#[allow(non_snake_case)]
pub fn H5D__chunk_disjoint(left: &ChunkInfo, right: &ChunkInfo) -> bool {
    left.coord != right.coord
}

#[allow(non_snake_case)]
pub fn H5D_btree_debug(index: &ChunkIndexState) -> String {
    chunk_index_dump("btree", index)
}

#[allow(non_snake_case)]
pub fn H5D__select_io(dataset: &DatasetApi, spans: &[(usize, usize)]) -> Vec<Vec<u8>> {
    H5D__scatter_file(&dataset.raw, spans)
}

#[allow(non_snake_case)]
pub fn H5D_select_io_mem(src: &[u8], spans: &[(usize, usize)]) -> Vec<Vec<u8>> {
    H5D__scatter_mem(src, spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_storage_reads_and_writes() {
        let mut storage = H5D__compact_construct(b"abcd".to_vec());
        H5D__compact_writevv(&mut storage, 2, b"XY").unwrap();
        assert_eq!(H5D__compact_readvv(&storage, 0, 4).unwrap(), b"abXY");
        assert!(storage.dirty);
        H5D__compact_flush(&mut storage);
        assert!(!storage.dirty);
    }
}
