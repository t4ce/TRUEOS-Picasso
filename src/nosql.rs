//! Experimental filesystem-backed object storage for large Picasso payloads.
//!
//! Redb stores the catalog; payload bytes remain ordinary files so TRUEOS can
//! later replace the host filesystem adapter with its native async VFS.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

const META: TableDefinition<&str, u64> = TableDefinition::new("mass_meta_v1");
const OBJECTS: TableDefinition<u64, &[u8]> = TableDefinition::new("mass_objects_v1");
const RECORD_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MassId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObjectInfo {
    pub id: MassId,
    pub byte_length: u64,
    pub label: String,
}

#[derive(Debug)]
pub enum Error {
    DatabaseOpen(redb::DatabaseError),
    Transaction(redb::TransactionError),
    Table(redb::TableError),
    Storage(redb::StorageError),
    Commit(redb::CommitError),
    Io(std::io::Error),
    NotFound(u64),
    InvalidLabel,
    InvalidRecord,
    InvalidRange,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DatabaseOpen(e) => write!(f, "database open: {e}"),
            Self::Transaction(e) => write!(f, "database transaction: {e}"),
            Self::Table(e) => write!(f, "database table: {e}"),
            Self::Storage(e) => write!(f, "database storage: {e}"),
            Self::Commit(e) => write!(f, "database commit: {e}"),
            Self::Io(e) => write!(f, "filesystem: {e}"),
            Self::NotFound(id) => write!(f, "MASS object {id} does not exist"),
            Self::InvalidLabel => f.write_str("object label contains a NUL byte"),
            Self::InvalidRecord => f.write_str("invalid catalog record"),
            Self::InvalidRange => f.write_str("requested byte range is outside the object"),
        }
    }
}

impl std::error::Error for Error {}

macro_rules! error_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for Error {
            fn from(error: $source) -> Self {
                Self::$variant(error)
            }
        }
    };
}

error_from!(redb::DatabaseError, DatabaseOpen);
error_from!(redb::TransactionError, Transaction);
error_from!(redb::TableError, Table);
error_from!(redb::StorageError, Storage);
error_from!(redb::CommitError, Commit);
error_from!(std::io::Error, Io);

pub type Result<T> = std::result::Result<T, Error>;

/// Experimental MASS store. Published objects are immutable.
pub struct Store {
    catalog: Database,
    root: PathBuf,
}

impl Store {
    /// Opens or creates a catalog and its filesystem payload directories.
    pub fn open(catalog_path: impl AsRef<Path>, root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_owned();
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("staging"))?;
        Ok(Self {
            catalog: Database::create(catalog_path)?,
            root,
        })
    }

    /// Durably publishes one immutable payload and returns its stable ID.
    pub fn put(&self, label: &str, bytes: &[u8]) -> Result<MassId> {
        if label.contains('\0') {
            return Err(Error::InvalidLabel);
        }
        let id = self.reserve_id()?;
        let staging = self
            .root
            .join("staging")
            .join(format!("{:016x}.part", id.0));
        let destination = self.object_path(id);

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&staging, &destination)?;
        sync_directory(&self.root.join("objects"))?;

        let record = encode_record(bytes.len() as u64, label);
        let write = self.catalog.begin_write()?;
        {
            let mut objects = write.open_table(OBJECTS)?;
            objects.insert(id.0, record.as_slice())?;
        }
        write.commit()?;
        log_os_core::global_log_with_target_level(
            "storage",
            log_os_core::LogLevel::Info,
            format_args!(
                "MASS published object={} bytes={} label={label}",
                id.0,
                bytes.len()
            ),
        );
        Ok(id)
    }

    pub fn info(&self, id: MassId) -> Result<ObjectInfo> {
        let read = self.catalog.begin_read()?;
        let objects = read.open_table(OBJECTS)?;
        let value = objects.get(id.0)?.ok_or(Error::NotFound(id.0))?;
        decode_record(id, value.value())
    }

    pub fn read(&self, id: MassId) -> Result<Vec<u8>> {
        let info = self.info(id)?;
        self.read_range(id, 0, info.byte_length)
    }

    /// Reads exactly `length` bytes without loading the whole object.
    pub fn read_range(&self, id: MassId, offset: u64, length: u64) -> Result<Vec<u8>> {
        let info = self.info(id)?;
        let end = offset.checked_add(length).ok_or(Error::InvalidRange)?;
        if end > info.byte_length || length > usize::MAX as u64 {
            return Err(Error::InvalidRange);
        }
        let mut file = File::open(self.object_path(id))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; length as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn reserve_id(&self) -> Result<MassId> {
        let write = self.catalog.begin_write()?;
        let id = {
            let mut meta = write.open_table(META)?;
            let next = meta
                .get("next_object")?
                .map(|value| value.value())
                .unwrap_or(1);
            meta.insert(
                "next_object",
                next.checked_add(1).ok_or(Error::InvalidRecord)?,
            )?;
            MassId(next)
        };
        write.commit()?;
        Ok(id)
    }

    fn object_path(&self, id: MassId) -> PathBuf {
        self.root
            .join("objects")
            .join(format!("{:016x}.mass", id.0))
    }
}

fn encode_record(byte_length: u64, label: &str) -> Vec<u8> {
    let mut record = Vec::with_capacity(10 + label.len());
    record.extend_from_slice(&RECORD_VERSION.to_le_bytes());
    record.extend_from_slice(&byte_length.to_le_bytes());
    record.extend_from_slice(label.as_bytes());
    record
}

fn decode_record(id: MassId, record: &[u8]) -> Result<ObjectInfo> {
    let version = record.get(..2).ok_or(Error::InvalidRecord)?;
    if u16::from_le_bytes(version.try_into().map_err(|_| Error::InvalidRecord)?) != RECORD_VERSION {
        return Err(Error::InvalidRecord);
    }
    let length = record.get(2..10).ok_or(Error::InvalidRecord)?;
    let byte_length = u64::from_le_bytes(length.try_into().map_err(|_| Error::InvalidRecord)?);
    let label = std::str::from_utf8(record.get(10..).ok_or(Error::InvalidRecord)?)
        .map_err(|_| Error::InvalidRecord)?
        .to_owned();
    Ok(ObjectInfo {
        id,
        byte_length,
        label,
    })
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
