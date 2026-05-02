/// Attribute creation properties read from an existing attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributeCreate {
    char_encoding: u8,
}

impl AttributeCreate {
    pub(crate) fn from_attribute(attr: &crate::hl::attribute::Attribute) -> Self {
        Self {
            char_encoding: attr.info().char_encoding,
        }
    }

    /// Attribute name character encoding: 0=ASCII, 1=UTF-8.
    pub fn char_encoding(&self) -> u8 {
        self.char_encoding
    }

    /// Set attribute name character encoding: 0=ASCII, 1=UTF-8.
    pub fn set_char_encoding(&mut self, char_encoding: u8) {
        self.char_encoding = char_encoding;
    }
}
