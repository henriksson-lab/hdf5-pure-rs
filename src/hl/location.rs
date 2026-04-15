/// Common interface for named HDF5 objects (File, Group, Dataset).
pub trait Location {
    /// Get the object's name/path within the file.
    fn name(&self) -> &str;

    /// List attribute names on this object.
    fn attr_names(&self) -> crate::Result<Vec<String>>;

    /// Get an attribute by name.
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute>;
}

impl Location for crate::hl::file::File {
    fn name(&self) -> &str {
        "/"
    }
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        self.attr_names()
    }
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute> {
        self.attr(name)
    }
}

impl Location for crate::hl::group::Group {
    fn name(&self) -> &str {
        self.name()
    }
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        self.attr_names()
    }
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute> {
        self.attr(name)
    }
}

impl Location for crate::hl::dataset::Dataset {
    fn name(&self) -> &str {
        self.name()
    }
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        self.attr_names()
    }
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute> {
        self.attr(name)
    }
}

/// Check if a named member exists in a group.
pub fn link_exists(group: &crate::hl::group::Group, name: &str) -> crate::Result<bool> {
    let members = group.member_names()?;
    Ok(members.iter().any(|n| n == name))
}
