/// Pure Rust object/region reference token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    object_token: u64,
    region: Option<Vec<u8>>,
    file_name: Option<String>,
    loc_id: Option<u64>,
}

impl Reference {
    /// Render an object token.
    pub fn print_token(token: u64) -> String {
        format!("{token:#x}")
    }

    /// Initialize reference package support.
    pub fn init_package() -> bool {
        true
    }

    /// Create an object reference.
    pub fn create_object(object_token: u64, file_name: Option<String>) -> Self {
        Self {
            object_token,
            region: None,
            file_name,
            loc_id: None,
        }
    }

    /// Create a region reference.
    pub fn create_region(object_token: u64, region: Vec<u8>, file_name: Option<String>) -> Self {
        Self {
            object_token,
            region: Some(region),
            file_name,
            loc_id: None,
        }
    }

    /// Destroy a reference. The pure Rust value is consumed.
    pub fn destroy(self) {}

    /// Set the associated location id.
    pub fn set_loc_id(&mut self, loc_id: u64) {
        self.loc_id = Some(loc_id);
    }

    /// Return the associated location id.
    pub fn get_loc_id(&self) -> Option<u64> {
        self.loc_id
    }

    /// Reopen the referenced file, represented here by returning its name.
    pub fn reopen_file(&self) -> Option<&str> {
        self.file_name.as_deref()
    }

    /// Return reference equality.
    pub fn equal(&self, other: &Self) -> bool {
        self == other
    }

    /// Copy a reference.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Return object token.
    pub fn get_obj_token(&self) -> u64 {
        self.object_token
    }

    /// Set object token.
    pub fn set_obj_token(&mut self, token: u64) {
        self.object_token = token;
    }

    /// Return region bytes.
    pub fn get_region(&self) -> Option<&[u8]> {
        self.region.as_deref()
    }

    /// Return file name.
    pub fn get_file_name(&self) -> Option<&str> {
        self.file_name.as_deref()
    }

    /// Encode a reference.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        Self::encode_obj_token_into(self.object_token, &mut out);
        Self::encode_region_into(self.region.as_deref(), &mut out);
        out
    }

    /// Encode an object token.
    pub fn encode_obj_token(&self) -> Vec<u8> {
        let mut out = Vec::new();
        Self::encode_obj_token_into(self.object_token, &mut out);
        out
    }

    /// Encode a region payload.
    pub fn encode_region(&self) -> Vec<u8> {
        let mut out = Vec::new();
        Self::encode_region_into(self.region.as_deref(), &mut out);
        out
    }

    /// Decode a region payload.
    pub fn decode_region(bytes: &[u8]) -> Option<Vec<u8>> {
        if bytes.is_empty() {
            None
        } else {
            Some(bytes.to_vec())
        }
    }

    /// Encode a heap reference payload.
    pub fn encode_heap(&self) -> Vec<u8> {
        self.encode()
    }

    /// Encode an object-token compatibility payload.
    pub fn encode_token_obj_compat(&self) -> Vec<u8> {
        self.encode_obj_token()
    }

    /// Decode an object-token compatibility payload.
    pub fn decode_token_obj_compat(bytes: &[u8]) -> Option<u64> {
        let word = bytes.get(..8)?;
        Some(u64::from_le_bytes(word.try_into().ok()?))
    }

    /// Decode a region-token compatibility payload.
    pub fn decode_token_region_compat(bytes: &[u8]) -> Option<Vec<u8>> {
        let len_word = bytes.get(..8)?;
        let len = u64::from_le_bytes(len_word.try_into().ok()?) as usize;
        let end = 8usize.checked_add(len)?;
        Some(bytes.get(8..end)?.to_vec())
    }

    /// Public object-reference constructor alias.
    pub fn create_object_api(object_token: u64, file_name: Option<String>) -> Self {
        Self::create_object(object_token, file_name)
    }

    /// Public region-reference constructor alias.
    pub fn create_region_api(
        object_token: u64,
        region: Vec<u8>,
        file_name: Option<String>,
    ) -> Self {
        Self::create_region(object_token, region, file_name)
    }

    /// Open-attribute helper returns the target token.
    pub fn open_attr_api_common(&self) -> u64 {
        self.object_token
    }

    /// Encode a region-token compatibility payload.
    pub fn encode_token_region_compat(&self) -> Vec<u8> {
        self.encode_region()
    }

    /// Public get-region alias.
    pub fn get_region_api(&self) -> Option<&[u8]> {
        self.get_region()
    }

    fn encode_obj_token_into(token: u64, out: &mut Vec<u8>) {
        out.extend_from_slice(&token.to_le_bytes());
    }

    fn encode_region_into(region: Option<&[u8]>, out: &mut Vec<u8>) {
        let len = region.map_or(0, <[u8]>::len) as u64;
        out.extend_from_slice(&len.to_le_bytes());
        if let Some(region) = region {
            out.extend_from_slice(region);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Reference;

    #[test]
    fn reference_aliases_roundtrip() {
        assert!(Reference::init_package());
        assert_eq!(Reference::print_token(42), "0x2a");
        let mut r = Reference::create_region(7, vec![1, 2, 3], Some("a.h5".into()));
        assert_eq!(r.get_obj_token(), 7);
        r.set_obj_token(8);
        r.set_loc_id(9);
        assert_eq!(r.get_loc_id(), Some(9));
        assert_eq!(r.reopen_file(), Some("a.h5"));
        assert_eq!(r.get_file_name(), Some("a.h5"));
        assert_eq!(r.get_region(), Some([1, 2, 3].as_slice()));
        assert!(r.equal(&r.copy()));
        assert_eq!(
            Reference::decode_token_obj_compat(&r.encode_obj_token()),
            Some(8)
        );
        assert_eq!(
            Reference::decode_token_region_compat(&r.encode_token_region_compat()),
            Some(vec![1, 2, 3])
        );
        assert_eq!(r.open_attr_api_common(), 8);
        assert_eq!(r.get_region_api(), Some([1, 2, 3].as_slice()));
        Reference::create_object_api(1, None).destroy();
        Reference::create_region_api(1, vec![4], None).destroy();
    }
}
