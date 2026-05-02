use crate::format::messages::data_layout::LayoutClass;
use crate::format::messages::filter_pipeline::FilterDesc;

/// Dataset creation properties (read from an existing dataset).
#[derive(Debug, Clone)]
pub struct DatasetCreate {
    pub layout: LayoutClass,
    pub chunk_dims: Option<Vec<u64>>,
    pub chunk_opts: Option<u8>,
    pub filters: Vec<FilterInfo>,
    pub external_files: Vec<ExternalStorageInfo>,
    pub virtual_mappings: Vec<VirtualMappingInfo>,
    pub virtual_spatial_tree: bool,
    pub fill_alloc_time: Option<u8>,
    pub fill_time: Option<u8>,
    pub fill_value_defined: bool,
    pub fill_value: Option<Vec<u8>>,
}

/// Simplified filter description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterInfo {
    pub id: u16,
    pub name: String,
    pub flags: u16,
    pub params: Vec<u32>,
}

/// External raw-data storage entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalStorageInfo {
    pub name: String,
    pub file_offset: u64,
    pub size: u64,
}

/// Virtual-dataset mapping entry stored in a dataset creation property list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualMappingInfo {
    pub file_name: String,
    pub dataset_name: String,
    pub source_select: VirtualSelectionInfo,
    pub virtual_select: VirtualSelectionInfo,
}

/// Serialized virtual-dataset source or destination selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualSelectionInfo {
    All,
    Points(Vec<Vec<u64>>),
    Regular {
        start: Vec<u64>,
        stride: Vec<u64>,
        count: Vec<u64>,
        block: Vec<u64>,
    },
    Irregular(Vec<IrregularHyperslabBlockInfo>),
}

/// Irregular hyperslab block in a virtual-dataset mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrregularHyperslabBlockInfo {
    pub start: Vec<u64>,
    pub block: Vec<u64>,
}

impl FilterInfo {
    pub fn from_desc(desc: &FilterDesc) -> Self {
        let name = match desc.id {
            1 => "deflate".to_string(),
            2 => "shuffle".to_string(),
            3 => "fletcher32".to_string(),
            4 => "szip".to_string(),
            5 => "nbit".to_string(),
            6 => "scaleoffset".to_string(),
            _ => desc
                .name
                .clone()
                .unwrap_or_else(|| format!("filter_{}", desc.id)),
        };
        Self {
            id: desc.id,
            name,
            flags: desc.flags,
            params: desc.client_data.clone(),
        }
    }
}

impl DatasetCreate {
    /// Extract dataset creation properties from a Dataset.
    pub fn from_dataset(ds: &crate::hl::dataset::Dataset) -> crate::Result<Self> {
        let info = ds.info()?;

        let filters: Vec<FilterInfo> = info
            .filter_pipeline
            .as_ref()
            .map(|p| p.filters.iter().map(FilterInfo::from_desc).collect())
            .unwrap_or_default();
        let external_files = ds.external_storage_entries_with_info(&info)?;
        let virtual_mappings = ds.virtual_mapping_infos_with_info(&info)?;

        Ok(Self {
            layout: info.layout.layout_class,
            chunk_dims: info.layout.chunk_dims,
            chunk_opts: info.layout.chunk_flags,
            filters,
            external_files,
            virtual_mappings,
            virtual_spatial_tree: false,
            fill_alloc_time: info.fill_value.as_ref().map(|fill| fill.alloc_time),
            fill_time: info.fill_value.as_ref().map(|fill| fill.fill_time),
            fill_value_defined: info
                .fill_value
                .as_ref()
                .map(|fill| fill.defined)
                .unwrap_or(false),
            fill_value: info.fill_value.and_then(|fill| fill.value),
        })
    }

    /// Whether the dataset is chunked.
    pub fn is_chunked(&self) -> bool {
        self.layout == LayoutClass::Chunked
    }

    /// Whether the dataset has any compression filters.
    pub fn is_compressed(&self) -> bool {
        self.filters.iter().any(|f| f.id == 1 || f.id == 4) // deflate or szip
    }

    /// Whether the dataset uses shuffle filter.
    pub fn has_shuffle(&self) -> bool {
        self.filters.iter().any(|f| f.id == 2)
    }

    /// Get the deflate compression level (if deflate is used).
    pub fn deflate_level(&self) -> Option<u32> {
        self.filters
            .iter()
            .find(|f| f.id == 1)
            .and_then(|f| f.params.first().copied())
    }

    /// Return chunk option flags for v4 chunked datasets.
    pub fn chunk_opts(&self) -> Option<u8> {
        self.chunk_opts
    }

    /// Set v4 chunk option flags.
    pub fn set_chunk_opts(&mut self, opts: Option<u8>) {
        self.chunk_opts = opts;
    }

    /// Return the number of external raw-storage files.
    pub fn external_count(&self) -> usize {
        self.external_files.len()
    }

    /// Return one external raw-storage entry by index.
    pub fn external(&self, index: usize) -> Option<&ExternalStorageInfo> {
        self.external_files.get(index)
    }

    /// Append an external raw-storage file entry.
    pub fn set_external<S: Into<String>>(&mut self, name: S, file_offset: u64, size: u64) {
        self.external_files.push(ExternalStorageInfo {
            name: name.into(),
            file_offset,
            size,
        });
    }

    /// Return one filter by pipeline index.
    pub fn filter(&self, index: usize) -> Option<&FilterInfo> {
        self.filters.get(index)
    }

    /// Return one filter by filter id.
    pub fn filter_by_id(&self, id: u16) -> Option<&FilterInfo> {
        self.filters.iter().find(|filter| filter.id == id)
    }

    /// Append or replace one filter in the dataset creation pipeline.
    pub fn set_filter<S: Into<String>>(&mut self, id: u16, name: S, flags: u16, params: Vec<u32>) {
        if let Some(filter) = self.filters.iter_mut().find(|filter| filter.id == id) {
            *filter = FilterInfo {
                id,
                name: name.into(),
                flags,
                params,
            };
        } else {
            self.filters.push(FilterInfo {
                id,
                name: name.into(),
                flags,
                params,
            });
        }
    }

    /// Enable the shuffle filter.
    pub fn set_shuffle(&mut self) {
        self.set_filter(2, "shuffle", 0, Vec::new());
    }

    /// Enable the NBit filter.
    pub fn set_nbit(&mut self) {
        self.set_filter(5, "nbit", 0, Vec::new());
    }

    /// Enable the ScaleOffset filter with client parameters.
    pub fn set_scaleoffset(&mut self, params: Vec<u32>) {
        self.set_filter(6, "scaleoffset", 0, params);
    }

    /// Enable the Fletcher32 checksum filter.
    pub fn set_fletcher32(&mut self) {
        self.set_filter(3, "fletcher32", 0, Vec::new());
    }

    /// Enable the SZip filter with client parameters.
    pub fn set_szip(&mut self, params: Vec<u32>) {
        self.set_filter(4, "szip", 0, params);
    }

    /// Return the number of filters in the dataset creation pipeline.
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Return the number of virtual-dataset mappings.
    pub fn virtual_count(&self) -> usize {
        self.virtual_mappings.len()
    }

    /// Append one virtual-dataset mapping.
    pub fn set_virtual(&mut self, mapping: VirtualMappingInfo) {
        self.virtual_mappings.push(mapping);
    }

    /// Return the source file name for one virtual-dataset mapping.
    pub fn virtual_filename(&self, index: usize) -> Option<&str> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| mapping.file_name.as_str())
    }

    /// Return the source dataset name for one virtual-dataset mapping.
    pub fn virtual_dsetname(&self, index: usize) -> Option<&str> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| mapping.dataset_name.as_str())
    }

    /// Return the source selection for one virtual-dataset mapping.
    pub fn virtual_srcspace(&self, index: usize) -> Option<&VirtualSelectionInfo> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| &mapping.source_select)
    }

    /// Return the virtual-dataset destination selection for one mapping.
    pub fn virtual_vspace(&self, index: usize) -> Option<&VirtualSelectionInfo> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| &mapping.virtual_select)
    }

    /// Whether this property list requests an HDF5 spatial tree for VDS lookups.
    ///
    /// The pure Rust reader materializes mapping coordinates directly and does
    /// not persist an HDF5 spatial-tree build flag, so existing files report the
    /// default disabled state.
    pub fn virtual_spatial_tree(&self) -> bool {
        self.virtual_spatial_tree
    }

    /// Set whether a VDS spatial tree should be requested.
    pub fn set_virtual_spatial_tree(&mut self, enabled: bool) {
        self.virtual_spatial_tree = enabled;
    }

    /// Set fill allocation time.
    pub fn set_alloc_time(&mut self, alloc_time: u8) {
        self.fill_alloc_time = Some(alloc_time);
    }

    /// Set raw fill value bytes.
    pub fn set_fill_value(&mut self, value: Option<Vec<u8>>) {
        self.fill_value_defined = value.is_some();
        self.fill_value = value;
    }
}
