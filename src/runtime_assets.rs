//! Runtime-owned storage for bytes embedded by a Picasso caller.
//!
//! This is intentionally independent from the host glTF importer. A Blueprint
//! creates one [`Picasso`], inserts its embedded assets, and keeps that owner
//! alive for as long as those assets are needed.

use alloc::{sync::Arc, vec::Vec};
use redb::StorageBackend;
#[cfg(any(feature = "host", feature = "test-std"))]
extern crate std;
#[cfg(not(any(feature = "host", feature = "test-std")))]
use redb::io as backend_io;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use spin::RwLock;
#[cfg(any(feature = "host", feature = "test-std"))]
use std::io as backend_io;

const ASSET_LENGTHS: TableDefinition<&str, u64> =
    TableDefinition::new("picasso_runtime_asset_lengths_v2");
const ASSET_CHUNKS: TableDefinition<(&str, u64), &[u8]> =
    TableDefinition::new("picasso_runtime_asset_chunks_v2");
// Leave room for keys and page headers inside a 64 KiB redb page. A whole
// 40 MiB image value otherwise needs a 64 MiB page and larger backend growth.
const RUNTIME_CHUNK_BYTES: usize = 60 * 1024;
// The backend already owns every byte in RAM. Keep only a small working cache
// instead of redb's 1 GiB disk-oriented default, especially for image bundles.
const RUNTIME_CACHE_BYTES: usize = 1024 * 1024;

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

    /// Opens a prepared, cleanly closed database, taking ownership of its RAM
    /// buffer. Asset bytes are not copied or re-inserted. This owner is read-only.
    pub fn from_runtime_database_image(bytes: Vec<u8>) -> Result<Self, PicassoError> {
        Ok(Self {
            assets: RuntimeAssetDatabase::from_image(bytes)?,
        })
    }

    /// Compact and close an ingestion database, returning its portable redb
    /// image. The returned bytes can be compressed and opened by another owner.
    pub fn into_runtime_database_image(self) -> Result<Vec<u8>, PicassoError> {
        self.assets.into_image()
    }

    /// Stores an embedded asset's bytes exactly under `name`.
    pub fn put_embedded_asset(&self, name: &str, bytes: &[u8]) -> Result<(), PicassoError> {
        self.assets.insert(name, bytes)
    }

    /// Returns an owned copy of the exact bytes stored under `name`.
    ///
    /// The copy is deliberate: no redb transaction or storage guard crosses
    /// Picasso's public boundary, so callers can safely retain the bytes while
    /// Picasso continues serving the rest of the runtime asset catalog.
    pub fn embedded_asset(&self, name: &str) -> Result<Option<Vec<u8>>, PicassoError> {
        self.assets.get(name)
    }
}

struct RuntimeAssetDatabase {
    database: Database,
    image: Arc<RwLock<Vec<u8>>>,
    read_only: bool,
}

// The prepared image moves directly into this backend. redb may update its
// bookkeeping header on open/close, but a prepared image never grows and the
// public owner rejects asset writes. There is no filesystem backend.
struct RamImageBackend {
    image: Arc<RwLock<Vec<u8>>>,
    growable: bool,
}
impl core::fmt::Debug for RamImageBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RamImageBackend")
            .field("bytes", &self.image.read().len())
            .finish()
    }
}
impl StorageBackend for RamImageBackend {
    fn len(&self) -> Result<u64, backend_io::Error> {
        Ok(self.image.read().len() as u64)
    }
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), backend_io::Error> {
        let image = self.image.read();
        let start = usize::try_from(offset)
            .map_err(|_| backend_io::Error::other("RAM image offset overflow"))?;
        let end = start
            .checked_add(out.len())
            .ok_or_else(|| backend_io::Error::other("RAM image range overflow"))?;
        let bytes = image
            .get(start..end)
            .ok_or_else(|| backend_io::Error::other("RAM image read out of range"))?;
        out.copy_from_slice(bytes);
        Ok(())
    }
    fn set_len(&self, len: u64) -> Result<(), backend_io::Error> {
        let len = usize::try_from(len)
            .map_err(|_| backend_io::Error::other("RAM image size overflow"))?;
        let mut image = self.image.write();
        if !self.growable && image.len() != len {
            return Err(backend_io::Error::other(
                "prepared RAM image cannot be resized",
            ));
        }
        if len > image.len() {
            let extra = len - image.len();
            image
                .try_reserve(extra)
                .map_err(|_| backend_io::Error::other("RAM image allocation failed"))?;
        }
        image.resize(len, 0);
        Ok(())
    }
    fn sync_data(&self) -> Result<(), backend_io::Error> {
        Ok(())
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), backend_io::Error> {
        let mut image = self.image.write();
        let start = usize::try_from(offset)
            .map_err(|_| backend_io::Error::other("RAM image offset overflow"))?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| backend_io::Error::other("RAM image range overflow"))?;
        image
            .get_mut(start..end)
            .ok_or_else(|| backend_io::Error::other("RAM image write out of range"))?
            .copy_from_slice(bytes);
        Ok(())
    }
}

impl RuntimeAssetDatabase {
    fn new() -> Result<Self, PicassoError> {
        Self::open_image(Vec::new(), false)
    }

    fn from_image(bytes: Vec<u8>) -> Result<Self, PicassoError> {
        if bytes.is_empty() {
            return Err(PicassoError::Storage);
        }
        let result = Self::open_image(bytes, true)?;
        {
            let read = result
                .database
                .begin_read()
                .map_err(|_| PicassoError::Storage)?;
            read.open_table(ASSET_LENGTHS)
                .map_err(|_| PicassoError::Storage)?;
            read.open_table(ASSET_CHUNKS)
                .map_err(|_| PicassoError::Storage)?;
        }
        Ok(result)
    }

    fn open_image(bytes: Vec<u8>, read_only: bool) -> Result<Self, PicassoError> {
        let image = Arc::new(RwLock::new(bytes));
        let database = Database::builder()
            .set_cache_size(RUNTIME_CACHE_BYTES)
            .create_with_backend(RamImageBackend {
                image: image.clone(),
                growable: !read_only,
            })
            .map_err(|_| PicassoError::Storage)?;
        Ok(Self {
            database,
            image,
            read_only,
        })
    }

    fn into_image(mut self) -> Result<Vec<u8>, PicassoError> {
        if self.read_only {
            return Err(PicassoError::Storage);
        }
        self.database.compact().map_err(|_| PicassoError::Storage)?;
        drop(self.database);
        Arc::try_unwrap(self.image)
            .map_err(|_| PicassoError::Storage)
            .map(RwLock::into_inner)
    }

    fn insert(&self, name: &str, bytes: &[u8]) -> Result<(), PicassoError> {
        if self.read_only {
            return Err(PicassoError::Storage);
        }
        validate_name(name)?;
        let write = self
            .database
            .begin_write()
            .map_err(|_| PicassoError::Storage)?;
        {
            let mut lengths = write
                .open_table(ASSET_LENGTHS)
                .map_err(|_| PicassoError::Storage)?;
            let previous_length = lengths
                .get(name)
                .map_err(|_| PicassoError::Storage)?
                .map(|entry| entry.value())
                .unwrap_or(0);
            let mut chunks = write
                .open_table(ASSET_CHUNKS)
                .map_err(|_| PicassoError::Storage)?;
            for (index, chunk) in bytes.chunks(RUNTIME_CHUNK_BYTES).enumerate() {
                chunks
                    .insert((name, index as u64), chunk)
                    .map_err(|_| PicassoError::Storage)?;
            }
            for index in (bytes.len() as u64).div_ceil(RUNTIME_CHUNK_BYTES as u64)
                ..previous_length.div_ceil(RUNTIME_CHUNK_BYTES as u64)
            {
                chunks
                    .remove((name, index))
                    .map_err(|_| PicassoError::Storage)?;
            }
            lengths
                .insert(name, bytes.len() as u64)
                .map_err(|_| PicassoError::Storage)?;
        }
        write.commit().map_err(|_| PicassoError::Storage)
    }

    fn get(&self, name: &str) -> Result<Option<Vec<u8>>, PicassoError> {
        validate_name(name)?;
        let read = self
            .database
            .begin_read()
            .map_err(|_| PicassoError::Storage)?;
        let lengths = match read.open_table(ASSET_LENGTHS) {
            Ok(assets) => assets,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(_) => return Err(PicassoError::Storage),
        };
        let Some(length) = lengths.get(name).map_err(|_| PicassoError::Storage)? else {
            return Ok(None);
        };
        let length = usize::try_from(length.value()).map_err(|_| PicassoError::Storage)?;
        let chunks = read
            .open_table(ASSET_CHUNKS)
            .map_err(|_| PicassoError::Storage)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| PicassoError::Storage)?;
        for index in 0..length.div_ceil(RUNTIME_CHUNK_BYTES) {
            let chunk = chunks
                .get((name, index as u64))
                .map_err(|_| PicassoError::Storage)?
                .ok_or(PicassoError::Storage)?;
            if chunk.value().len() != (length - bytes.len()).min(RUNTIME_CHUNK_BYTES) {
                return Err(PicassoError::Storage);
            }
            bytes.extend_from_slice(chunk.value());
        }
        Ok(Some(bytes))
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
    fn prepared_image_opens_without_copying_or_reinserting_assets() {
        let source = Picasso::new().unwrap();
        let bytes = alloc::vec![0x35; 3 * RUNTIME_CHUNK_BYTES + 17];
        source.put_embedded_asset("image", &bytes).unwrap();
        source.put_embedded_asset("empty", b"").unwrap();
        let image = source.into_runtime_database_image().unwrap();
        let original_buffer = image.as_ptr();
        let owner = Picasso::from_runtime_database_image(image).unwrap();
        assert_eq!(owner.assets.image.read().as_ptr(), original_buffer);
        assert_eq!(owner.embedded_asset("image").unwrap().unwrap(), bytes);
        assert_eq!(owner.embedded_asset("empty").unwrap().unwrap(), b"");
        assert_eq!(owner.embedded_asset("missing").unwrap(), None);
        assert_eq!(
            owner.put_embedded_asset("image", b"replacement"),
            Err(PicassoError::Storage)
        );
    }

    #[test]
    fn invalid_or_empty_prepared_images_are_rejected() {
        assert!(Picasso::from_runtime_database_image(Vec::new()).is_err());
        assert!(Picasso::from_runtime_database_image(alloc::vec![0; 8192]).is_err());
    }

    #[test]
    fn stores_exact_bytes_without_a_file_backend() {
        let picasso = Picasso::new().unwrap();
        let bytes = [0, 1, 2, 0xff, 0, 4];

        picasso.put_embedded_asset("mesh", &bytes).unwrap();

        assert_eq!(
            picasso.embedded_asset("mesh").unwrap(),
            Some(bytes.to_vec())
        );
        assert_ne!(
            picasso.embedded_asset("mesh").unwrap(),
            Some(b"different".to_vec())
        );
    }

    #[test]
    fn replacement_does_not_change_other_assets() {
        let picasso = Picasso::new().unwrap();
        picasso.put_embedded_asset("a", b"first").unwrap();
        picasso.put_embedded_asset("b", b"second").unwrap();
        picasso.put_embedded_asset("a", b"replacement").unwrap();

        assert_eq!(
            picasso.embedded_asset("a").unwrap(),
            Some(b"replacement".to_vec())
        );
        assert_eq!(
            picasso.embedded_asset("b").unwrap(),
            Some(b"second".to_vec())
        );
        assert_eq!(picasso.embedded_asset("missing").unwrap(), None);
    }

    #[test]
    fn rejects_empty_names_without_creating_an_entry() {
        let picasso = Picasso::new().unwrap();
        assert_eq!(
            picasso.put_embedded_asset("", b"bytes"),
            Err(PicassoError::EmptyAssetName)
        );
        assert_eq!(
            picasso.embedded_asset(""),
            Err(PicassoError::EmptyAssetName)
        );
    }

    #[test]
    fn assets_larger_than_the_cache_survive_independent_reads_and_replacement() {
        let picasso = Picasso::new().unwrap();
        let large = alloc::vec![0x5a; 3 * RUNTIME_CACHE_BYTES + 17];
        picasso.put_embedded_asset("large", &large).unwrap();
        picasso.put_embedded_asset("other", b"keep").unwrap();
        let retained = picasso.embedded_asset("large").unwrap().unwrap();
        picasso.put_embedded_asset("large", b"replacement").unwrap();
        assert_eq!(retained, large);
        assert_eq!(picasso.embedded_asset("other").unwrap().unwrap(), b"keep");
        assert_eq!(
            picasso.embedded_asset("large").unwrap().unwrap(),
            b"replacement"
        );
    }

    #[test]
    fn chunked_replacement_releases_tail_and_preserves_empty_assets() {
        let picasso = Picasso::new().unwrap();
        let bytes = alloc::vec![0x35; 2 * RUNTIME_CHUNK_BYTES + 1];
        picasso.put_embedded_asset("image", &bytes).unwrap();
        picasso
            .put_embedded_asset("image/0", b"independent")
            .unwrap();
        picasso.put_embedded_asset("image", b"").unwrap();
        assert_eq!(picasso.embedded_asset("image").unwrap(), Some(Vec::new()));
        assert_eq!(
            picasso.embedded_asset("image/0").unwrap().unwrap(),
            b"independent"
        );
        let read = picasso.assets.database.begin_read().unwrap();
        let chunks = read.open_table(ASSET_CHUNKS).unwrap();
        for index in 0..3 {
            assert!(chunks.get(("image", index)).unwrap().is_none());
        }
    }
}
