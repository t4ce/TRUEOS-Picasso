//! TRUEOS Picasso.
//!
//! `core` is the bare-metal, `no_std` contract.  The host importer, glTF
//! parser, redb Dealer backend, CLI, and filesystem MASS adapter are enabled
//! only by the `host` feature.
#![cfg_attr(not(feature = "host"), no_std)]

pub mod core;
pub use core::*;

/// Renderer- and input-backend-neutral camera data and fly-camera controls.
pub mod cam;

/// Reusable neutral reference-grid geometry for line-list capable backends.
pub mod grid;
pub use grid::{GRID_INDICES, GRID_VERTICES};

/// Concrete shared-DDR execution ring for a platform that has already mapped
/// one allocation into both CPU and GPU address spaces.  This remains
/// allocation-, I/O-, and runtime-free so it is usable by a TRUEOS Blueprint.
pub mod cubism;

pub use cubism::{
    CoherentVisibility, CpuSlot, CubismError, DealerRingRecord, ExecRing, ExecSlotHeader,
    PublishedSlot, SharedByteRange, VisibilityOps,
};

#[cfg(feature = "host")]
#[path = "glTFredb.rs"]
mod host;

#[cfg(feature = "host")]
pub use host::*;

/// Small GLB chunk-oriented storage API.
///
/// This host-only surface is kept alongside the higher-level [`Store`] API:
/// `Store` preserves the parsed glTF graph, while this module provides direct
/// access to the original GLB, JSON chunk, and BIN chunk.
#[cfg(feature = "host")]
pub(crate) mod glb_library {
    //! Minimal GLB -> redb dump library.
    //!
    //! Cargo.toml:
    //! [dependencies]
    //! redb = "4.2"
    //! gltf = "1.4"
    //!
    //! Storage layout inside one redb table ("glb_dump_v1"):
    //!
    //!   <asset_id>/source.glb   -> original complete GLB bytes
    //!   <asset_id>/gltf.json    -> raw JSON chunk from the GLB
    //!   <asset_id>/buffer.bin   -> raw BIN chunk, when present
    //!
    //! Importing the same asset_id again is idempotent: the keys are overwritten.
    //! Existing unrelated redb tables are untouched.

    use std::error::Error;
    use std::fmt;

    use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

    /// One table acts as the "collection" for imported GLB assets.
    ///
    /// Using a dedicated table avoids interfering with tables you may already
    /// have seeded in the same redb database.
    pub const GLB_DUMP: TableDefinition<&str, &[u8]> = TableDefinition::new("glb_dump_v1");

    pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TrueosFsError(i32);

    impl fmt::Display for TrueosFsError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TRUEOSFS operation failed with error code {}", self.0)
        }
    }

    impl Error for TrueosFsError {}

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ImportStats {
        pub asset_id: String,
        pub source_bytes: usize,
        pub json_bytes: usize,
        pub bin_bytes: usize,
    }

    /// Open an existing redb database, or create it if it does not exist.
    ///
    /// `Database::create` is intentionally used here because redb opens an
    /// existing valid database without truncating it.
    pub fn open_database(path: &str) -> Result<Database> {
        Ok(Database::create(path)?)
    }

    /// Convenience entry point:
    ///
    ///     v::vfs_async::block_on(
    ///         dump_glb_file("assets.redb", "models/robot", "trueosfs:disc3/robot.glb")
    ///     )?;
    ///
    /// This preserves all existing unrelated tables in `assets.redb`.
    pub async fn dump_glb_file(
        db_path: &str,
        asset_id: &str,
        glb_path: &str,
    ) -> Result<ImportStats> {
        let bytes = v::vfs_async::read_file(glb_path.as_bytes())
            .await
            .map_err(TrueosFsError)?;
        let db = open_database(db_path)?;
        dump_glb_bytes(&db, asset_id, &bytes)
    }

    /// Dump an in-memory GLB into redb.
    ///
    /// The GLB is validated/split with `gltf::Glb::from_slice`, but nothing is
    /// transformed. You retain the exact original GLB plus its raw JSON/BIN
    /// chunks for fast direct access later.
    pub fn dump_glb_bytes(db: &Database, asset_id: &str, glb_bytes: &[u8]) -> Result<ImportStats> {
        let asset_id = normalize_asset_id(asset_id)?;
        let glb = gltf::Glb::from_slice(glb_bytes)?;

        let source_key = key(&asset_id, "source.glb");
        let json_key = key(&asset_id, "gltf.json");
        let bin_key = key(&asset_id, "buffer.bin");

        let json = glb.json.as_ref();
        let bin = glb.bin.as_ref().map(|chunk| chunk.as_ref());

        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(GLB_DUMP)?;

            // Upsert: rerunning an import for the same asset replaces it.
            table.insert(source_key.as_str(), glb_bytes)?;
            table.insert(json_key.as_str(), json)?;

            match bin {
                Some(bytes) => {
                    table.insert(bin_key.as_str(), bytes)?;
                }
                None => {
                    // Important for idempotency:
                    // if an older import had a BIN chunk and the new one does not,
                    // remove the stale value.
                    table.remove(bin_key.as_str())?;
                }
            }
        }
        write_txn.commit()?;

        Ok(ImportStats {
            asset_id,
            source_bytes: glb_bytes.len(),
            json_bytes: json.len(),
            bin_bytes: bin.map_or(0, |bytes| bytes.len()),
        })
    }

    /// Read the original complete GLB back from redb.
    pub fn load_glb(db: &Database, asset_id: &str) -> Result<Option<Vec<u8>>> {
        let asset_id = normalize_asset_id(asset_id)?;
        let source_key = key(&asset_id, "source.glb");

        let read_txn = db.begin_read()?;
        let table = match read_txn.open_table(GLB_DUMP) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(err) => return Err(err.into()),
        };

        Ok(table
            .get(source_key.as_str())?
            .map(|value| value.value().to_vec()))
    }

    /// Read only the raw glTF JSON chunk.
    pub fn load_json_chunk(db: &Database, asset_id: &str) -> Result<Option<Vec<u8>>> {
        load_part(db, asset_id, "gltf.json")
    }

    /// Read only the raw GLB BIN chunk.
    pub fn load_bin_chunk(db: &Database, asset_id: &str) -> Result<Option<Vec<u8>>> {
        load_part(db, asset_id, "buffer.bin")
    }

    fn load_part(db: &Database, asset_id: &str, part: &str) -> Result<Option<Vec<u8>>> {
        let asset_id = normalize_asset_id(asset_id)?;
        let part_key = key(&asset_id, part);

        let read_txn = db.begin_read()?;
        let table = match read_txn.open_table(GLB_DUMP) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(err) => return Err(err.into()),
        };

        Ok(table
            .get(part_key.as_str())?
            .map(|value| value.value().to_vec()))
    }

    pub(crate) fn normalize_asset_id(asset_id: &str) -> Result<String> {
        let id = asset_id.trim().trim_matches('/');

        if id.is_empty() {
            return Err("asset_id must not be empty".into());
        }

        if id.contains('\0') {
            return Err("asset_id must not contain NUL bytes".into());
        }

        Ok(id.to_owned())
    }

    fn key(asset_id: &str, part: &str) -> String {
        format!("{asset_id}/{part}")
    }
}

#[cfg(feature = "host")]
pub use glb_library::{
    GLB_DUMP, ImportStats, dump_glb_bytes, dump_glb_file, load_bin_chunk, load_glb,
    load_json_chunk, open_database,
};

#[cfg(test)]
mod test;
