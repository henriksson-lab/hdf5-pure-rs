use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

use super::{Dataset, DatasetAccess};

impl Dataset {
    pub(super) fn resolve_virtual_source_path(
        vds_path: Option<&Path>,
        source: &str,
        access: &DatasetAccess,
    ) -> Result<PathBuf> {
        if source == "." {
            return vds_path.map(Path::to_path_buf).ok_or_else(|| {
                Error::Unsupported("same-file virtual dataset source has no file path".into())
            });
        }
        let source_path = Path::new(source);
        if source_path.is_absolute() && source_path.exists() {
            return Ok(source_path.to_path_buf());
        }

        if let Some(prefixed) =
            Self::resolve_virtual_source_with_access_prefix(vds_path, source_path, access)?
        {
            return Ok(prefixed);
        }

        if let Some(prefixed) = Self::resolve_virtual_source_with_env_prefix(vds_path, source_path)?
        {
            return Ok(prefixed);
        }

        if source_path.is_absolute() {
            return Ok(source_path.to_path_buf());
        }
        let base = vds_path.and_then(Path::parent).ok_or_else(|| {
            Error::Unsupported("relative virtual dataset source has no base file path".into())
        })?;
        Ok(base.join(source_path))
    }

    fn resolve_virtual_source_with_access_prefix(
        vds_path: Option<&Path>,
        source_path: &Path,
        access: &DatasetAccess,
    ) -> Result<Option<PathBuf>> {
        let Some(raw_prefix) = access.virtual_prefix() else {
            return Ok(None);
        };
        if raw_prefix.as_os_str().is_empty() || raw_prefix == Path::new(".") {
            return Ok(None);
        }

        let file_name = source_path.file_name().ok_or_else(|| {
            Error::InvalidFormat("virtual dataset source has no file name".into())
        })?;
        let prefix = Self::expand_virtual_prefix_origin_path(vds_path, raw_prefix)?;
        Ok(Some(prefix.join(file_name)))
    }

    fn resolve_virtual_source_with_env_prefix(
        vds_path: Option<&Path>,
        source_path: &Path,
    ) -> Result<Option<PathBuf>> {
        let Ok(prefixes) = std::env::var("HDF5_VDS_PREFIX") else {
            return Ok(None);
        };

        let file_name = source_path.file_name().ok_or_else(|| {
            Error::InvalidFormat("virtual dataset source has no file name".into())
        })?;

        for raw_prefix in prefixes.split(':') {
            if raw_prefix.is_empty() || raw_prefix == "." {
                continue;
            }
            let prefix = Self::expand_virtual_prefix_origin_str(vds_path, raw_prefix)?;
            let candidate = prefix.join(file_name);
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }

        Ok(None)
    }

    fn expand_virtual_prefix_origin_path(
        vds_path: Option<&Path>,
        prefix: &Path,
    ) -> Result<PathBuf> {
        if let Some(prefix_str) = prefix.to_str() {
            if prefix_str.starts_with("${ORIGIN}") {
                return Self::expand_virtual_prefix_origin_str(vds_path, prefix_str);
            }
        }
        Ok(prefix.to_path_buf())
    }

    fn expand_virtual_prefix_origin_str(vds_path: Option<&Path>, prefix: &str) -> Result<PathBuf> {
        const ORIGIN: &str = "${ORIGIN}";

        if let Some(rest) = prefix.strip_prefix(ORIGIN) {
            let origin_dir = vds_path
                .and_then(Path::parent)
                .map(Path::to_path_buf)
                .ok_or_else(|| {
                    Error::Unsupported("VDS ${ORIGIN} prefix has no base file path".into())
                })?;

            let trimmed = rest.strip_prefix(['/', '\\']).unwrap_or(rest);
            if trimmed.is_empty() {
                return Ok(origin_dir);
            }
            return Ok(origin_dir.join(trimmed));
        }

        Ok(PathBuf::from(prefix))
    }
}
