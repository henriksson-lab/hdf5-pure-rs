use crate::format::messages::data_layout::LayoutClass;
use crate::format::messages::filter_pipeline::FilterDesc;

/// Dataset creation properties (read from an existing dataset).
#[derive(Debug, Clone)]
pub struct DatasetCreate {
    pub layout: LayoutClass,
    pub chunk_dims: Option<Vec<u64>>,
    pub filters: Vec<FilterInfo>,
    pub fill_value_defined: bool,
    pub fill_value: Option<Vec<u8>>,
}

/// Simplified filter description.
#[derive(Debug, Clone)]
pub struct FilterInfo {
    pub id: u16,
    pub name: String,
    pub params: Vec<u32>,
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
            .map(|p| p.filters.iter().map(FilterInfo::from_desc).collect())
            .unwrap_or_default();

        Ok(Self {
            layout: info.layout.layout_class,
            chunk_dims: info.layout.chunk_dims,
            filters,
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
}
