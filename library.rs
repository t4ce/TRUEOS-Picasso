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
use std::fs;
use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

/// One table acts as the "collection" for imported GLB assets.
///
/// Using a dedicated table avoids interfering with tables you may already
/// have seeded in the same redb database.
pub const GLB_DUMP: TableDefinition<&str, &[u8]> = TableDefinition::new("glb_dump_v1");

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync + 'static>>;

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
pub fn open_database(path: impl AsRef<Path>) -> Result<Database> {
    Ok(Database::create(path)?)
}

/// Convenience entry point:
///
///     dump_glb_file("assets.redb", "models/robot", "robot.glb")?;
///
/// This preserves all existing unrelated tables in `assets.redb`.
pub fn dump_glb_file(
    db_path: impl AsRef<Path>,
    asset_id: &str,
    glb_path: impl AsRef<Path>,
) -> Result<ImportStats> {
    let bytes = fs::read(glb_path)?;
    let db = open_database(db_path)?;
    dump_glb_bytes(&db, asset_id, &bytes)
}

/// Dump an in-memory GLB into redb.
///
/// The GLB is validated/split with `gltf::Glb::from_slice`, but nothing is
/// transformed. You retain the exact original GLB plus its raw JSON/BIN
/// chunks for fast direct access later.
pub fn dump_glb_bytes(
    db: &Database,
    asset_id: &str,
    glb_bytes: &[u8],
) -> Result<ImportStats> {
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

fn normalize_asset_id(asset_id: &str) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_id_normalization() {
        assert_eq!(normalize_asset_id("/models/robot/").unwrap(), "models/robot");
        assert!(normalize_asset_id("").is_err());
        assert!(normalize_asset_id("///").is_err());
    }
}
