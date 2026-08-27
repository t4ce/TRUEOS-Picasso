//! Runtime-owned storage for bytes embedded by a Picasso caller.
//!
//! This is intentionally independent from the host glTF importer. A Blueprint
//! creates one [`Picasso`], inserts its embedded assets, and keeps that owner
//! alive for as long as those assets are needed.

use redb::{Database, TableDefinition};
#[cfg(test)]
use redb::{ReadableDatabase, ReadableTable};

const ASSETS: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_runtime_assets_v1");

/// Failures exposed by Picasso's runtime asset boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PicassoError {
    EmptyAssetName,
    Storage,
}

impl core::fmt::Display for PicassoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyAssetName => f.write_str("embedded asset name must not be empty"),
            Self::Storage => f.write_str("Picasso asset storage operation failed"),
        }
    }
}

/// A running Picasso instance and its exact embedded assets.
///
/// Picasso keeps the storage backend private. It has no path and performs no
/// filesystem I/O. Re-inserting a name atomically replaces that asset while
/// leaving all other assets intact.
pub struct Picasso {
    assets: RuntimeAssetDatabase,
}

impl Picasso {
    /// Constructs a fresh runtime Picasso owner.
    pub fn new() -> Result<Self, PicassoError> {
        Ok(Self {
            assets: RuntimeAssetDatabase::new()?,
        })
    }

    /// Stores an embedded asset's bytes exactly under `name`.
    pub fn put_embedded_asset(&self, name: &str, bytes: &[u8]) -> Result<(), PicassoError> {
        self.assets.insert(name, bytes)
    }
}

struct RuntimeAssetDatabase {
    database: Database,
}

impl RuntimeAssetDatabase {
    fn new() -> Result<Self, PicassoError> {
        let database = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(|_| PicassoError::Storage)?;
        Ok(Self { database })
    }

    fn insert(&self, name: &str, bytes: &[u8]) -> Result<(), PicassoError> {
        validate_name(name)?;
        let write = self
            .database
            .begin_write()
            .map_err(|_| PicassoError::Storage)?;
        {
            let mut assets = write
                .open_table(ASSETS)
                .map_err(|_| PicassoError::Storage)?;
            assets
                .insert(name, bytes)
                .map_err(|_| PicassoError::Storage)?;
        }
        write.commit().map_err(|_| PicassoError::Storage)
    }

    #[cfg(test)]
    fn contains_exact(&self, name: &str, expected: &[u8]) -> Result<bool, PicassoError> {
        validate_name(name)?;
        let read = self
            .database
            .begin_read()
            .map_err(|_| PicassoError::Storage)?;
        let assets = match read.open_table(ASSETS) {
            Ok(assets) => assets,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(false),
            Err(_) => return Err(PicassoError::Storage),
        };
        assets
            .get(name)
            .map(|stored| stored.is_some_and(|value| value.value() == expected))
            .map_err(|_| PicassoError::Storage)
    }
}

fn validate_name(name: &str) -> Result<(), PicassoError> {
    if name.is_empty() {
        Err(PicassoError::EmptyAssetName)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_exact_bytes_without_a_file_backend() {
        let picasso = Picasso::new().unwrap();
        let bytes = [0, 1, 2, 0xff, 0, 4];

        picasso.put_embedded_asset("mesh", &bytes).unwrap();

        assert!(picasso.assets.contains_exact("mesh", &bytes).unwrap());
        assert!(!picasso.assets.contains_exact("mesh", b"different").unwrap());
    }

    #[test]
    fn replacement_does_not_change_other_assets() {
        let picasso = Picasso::new().unwrap();
        picasso.put_embedded_asset("a", b"first").unwrap();
        picasso.put_embedded_asset("b", b"second").unwrap();
        picasso.put_embedded_asset("a", b"replacement").unwrap();

        assert!(picasso.assets.contains_exact("a", b"replacement").unwrap());
        assert!(picasso.assets.contains_exact("b", b"second").unwrap());
        assert!(!picasso.assets.contains_exact("missing", b"").unwrap());
    }

    #[test]
    fn rejects_empty_names_without_creating_an_entry() {
        let picasso = Picasso::new().unwrap();
        assert_eq!(
            picasso.put_embedded_asset("", b"bytes"),
            Err(PicassoError::EmptyAssetName)
        );
        assert_eq!(
            picasso.assets.contains_exact("", b""),
            Err(PicassoError::EmptyAssetName)
        );
    }
}
