//! Picasso's durable, glTF-faithful import boundary.
//!
//! A collection is deliberately absent here: glTF nodes describe transform
//! hierarchy, while collections are an editor concern and must not imply parentage.

#[path = "MASS.rs"]
pub mod mass;

use std::{collections::BTreeMap, path::Path};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

const META: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_meta_v1");
const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_records_v1");
const BLOBS: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_blob_chunks_v1");
const TRACKING: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_tracking_v1");
const CHUNK: usize = 64 * 1024;

#[derive(Debug)]
pub enum Error {
    Database(redb::Error),
    DatabaseOpen(redb::DatabaseError),
    Transaction(redb::TransactionError),
    Table(redb::TableError),
    Storage(redb::StorageError),
    Commit(redb::CommitError),
    Io(std::io::Error),
    Gltf(gltf::Error),
    Json(serde_json::Error),
    MissingBuffer(String),
    DataUri,
    BufferLength {
        index: usize,
        actual: usize,
        declared: usize,
    },
    UnknownRevision(u64),
    UnknownRecord(String),
    InvalidTrackingState,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Database(e) => write!(f, "database: {e}"),
            Self::DatabaseOpen(e) => write!(f, "database open: {e}"),
            Self::Transaction(e) => write!(f, "database transaction: {e}"),
            Self::Table(e) => write!(f, "database table: {e}"),
            Self::Storage(e) => write!(f, "database storage: {e}"),
            Self::Commit(e) => write!(f, "database commit: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Gltf(e) => write!(f, "invalid glTF: {e}"),
            Self::Json(e) => write!(f, "invalid record encoding: {e}"),
            Self::MissingBuffer(uri) => write!(f, "missing payload for external buffer `{uri}`"),
            Self::DataUri => f.write_str("invalid data URI"),
            Self::BufferLength {
                index,
                actual,
                declared,
            } => write!(
                f,
                "buffer {index} is {actual} bytes; glTF declares {declared}"
            ),
            Self::UnknownRevision(id) => write!(f, "no published revision {id}"),
            Self::UnknownRecord(id) => write!(f, "no normalized record `{id}`"),
            Self::InvalidTrackingState => {
                f.write_str("a record must be included before it can be respected or tested")
            }
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

error_from!(redb::Error, Database);
error_from!(redb::DatabaseError, DatabaseOpen);
error_from!(redb::TransactionError, Transaction);
error_from!(redb::TableError, Table);
error_from!(redb::StorageError, Storage);
error_from!(redb::CommitError, Commit);
error_from!(std::io::Error, Io);
error_from!(gltf::Error, Gltf);
error_from!(serde_json::Error, Json);

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RevisionState {
    Staging,
    Complete,
}

/// Mutable bare-metal implementation status, deliberately kept outside the
/// immutable imported record.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tracking {
    pub included: bool,
    pub fully_respected: bool,
    pub tested: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub encoding: u16,
    pub id: String,
    pub latest_revision: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Revision {
    pub encoding: u16,
    pub id: u64,
    pub asset_id: String,
    pub state: RevisionState,
    pub source_blob: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub encoding: u16,
    pub id: String,
    pub name: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub encoding: u16,
    pub id: String,
    pub name: Option<String>,
    pub mesh: Option<String>,
    pub children: Vec<String>,
    pub matrix: [[f32; 4]; 4],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh {
    pub encoding: u16,
    pub id: String,
    pub name: Option<String>,
    pub primitives: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Primitive {
    pub encoding: u16,
    pub id: String,
    pub attributes: BTreeMap<String, String>,
    pub indices: Option<String>,
    pub material: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accessor {
    pub encoding: u16,
    pub id: String,
    pub buffer_view: Option<String>,
    pub offset: usize,
    pub count: usize,
    pub component_type: u32,
    pub element_type: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BufferView {
    pub encoding: u16,
    pub id: String,
    pub buffer: String,
    pub offset: usize,
    pub length: usize,
    pub stride: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Buffer {
    pub encoding: u16,
    pub id: String,
    pub byte_length: usize,
    pub blob: String,
}

/// An append-only redb store. Re-importing an asset always creates a new revision.
pub struct Store {
    db: Database,
}

impl Store {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            db: Database::create(path)?,
        })
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            db: Database::open(path)?,
        })
    }

    /// Imports `.gltf` JSON or `.glb`. `external` maps URI strings to their exact bytes.
    /// Data URI and GLB BIN buffers are resolved automatically and all resolved bytes are retained.
    pub fn import(
        &self,
        asset_id: &str,
        source: &[u8],
        external: &BTreeMap<String, Vec<u8>>,
    ) -> Result<u64> {
        let parsed = Prepared::parse(source, external)?; // All fallible source work precedes publication.
        let write = self.db.begin_write()?;
        let revision_id = {
            let mut meta = write.open_table(META)?;
            let next = meta
                .get("next_revision")?
                .map(|v| decode::<u64>(v.value()))
                .transpose()?
                .unwrap_or(1);
            meta.insert("next_revision", encode(&(next + 1))?.as_slice())?;
            next
        };
        let prefix = format!("r/{revision_id}");
        let mut records = write.open_table(RECORDS)?;
        let source_blob = format!("{prefix}/blob/source");
        put_blob(&write, &source_blob, source)?;
        put(
            &mut records,
            &format!("{prefix}/revision"),
            &Revision {
                encoding: 1,
                id: revision_id,
                asset_id: asset_id.into(),
                state: RevisionState::Staging,
                source_blob: source_blob.clone(),
            },
        )?;
        for (i, bytes) in parsed.buffers.iter().enumerate() {
            let id = eid(revision_id, "buffer", i);
            let blob = format!("{prefix}/blob/buffer/{i}");
            put_blob(&write, &blob, bytes)?;
            put(
                &mut records,
                &id,
                &Buffer {
                    encoding: 1,
                    id: id.clone(),
                    byte_length: bytes.len(),
                    blob,
                },
            )?;
        }
        for scene in parsed.doc.scenes() {
            let id = eid(revision_id, "scene", scene.index());
            put(
                &mut records,
                &id,
                &Scene {
                    encoding: 1,
                    id: id.clone(),
                    name: scene.name().map(str::to_owned),
                },
            )?;
            put(
                &mut records,
                &format!("{id}/roots"),
                &scene
                    .nodes()
                    .map(|n| eid(revision_id, "node", n.index()))
                    .collect::<Vec<_>>(),
            )?;
        }
        for node in parsed.doc.nodes() {
            let id = eid(revision_id, "node", node.index());
            put(
                &mut records,
                &id,
                &Node {
                    encoding: 1,
                    id: id.clone(),
                    name: node.name().map(str::to_owned),
                    mesh: node.mesh().map(|m| eid(revision_id, "mesh", m.index())),
                    children: node
                        .children()
                        .map(|n| eid(revision_id, "node", n.index()))
                        .collect(),
                    matrix: node.transform().matrix(),
                },
            )?;
        }
        for mesh in parsed.doc.meshes() {
            let id = eid(revision_id, "mesh", mesh.index());
            let primitives = mesh
                .primitives()
                .enumerate()
                .map(|(i, _)| format!("{id}/primitive/{i}"))
                .collect();
            put(
                &mut records,
                &id,
                &Mesh {
                    encoding: 1,
                    id: id.clone(),
                    name: mesh.name().map(str::to_owned),
                    primitives,
                },
            )?;
            for (i, p) in mesh.primitives().enumerate() {
                let pid = format!("{id}/primitive/{i}");
                let attributes = p
                    .attributes()
                    .map(|(s, a)| (format!("{s:?}"), eid(revision_id, "accessor", a.index())))
                    .collect();
                put(
                    &mut records,
                    &pid,
                    &Primitive {
                        encoding: 1,
                        id: pid.clone(),
                        attributes,
                        indices: p.indices().map(|a| eid(revision_id, "accessor", a.index())),
                        material: p.material().index(),
                    },
                )?;
            }
        }
        for a in parsed.doc.accessors() {
            let id = eid(revision_id, "accessor", a.index());
            put(
                &mut records,
                &id,
                &Accessor {
                    encoding: 1,
                    id: id.clone(),
                    buffer_view: a.view().map(|v| eid(revision_id, "buffer_view", v.index())),
                    offset: a.offset(),
                    count: a.count(),
                    component_type: a.data_type().as_gl_enum(),
                    element_type: format!("{:?}", a.dimensions()),
                },
            )?;
        }
        for v in parsed.doc.views() {
            let id = eid(revision_id, "buffer_view", v.index());
            put(
                &mut records,
                &id,
                &BufferView {
                    encoding: 1,
                    id: id.clone(),
                    buffer: eid(revision_id, "buffer", v.buffer().index()),
                    offset: v.offset(),
                    length: v.length(),
                    stride: v.stride(),
                },
            )?;
        }
        let asset_key = format!("asset/{asset_id}");
        put(
            &mut records,
            &asset_key,
            &Asset {
                encoding: 1,
                id: asset_id.into(),
                latest_revision: revision_id,
            },
        )?;
        put(
            &mut records,
            &format!("{prefix}/revision"),
            &Revision {
                encoding: 1,
                id: revision_id,
                asset_id: asset_id.into(),
                state: RevisionState::Complete,
                source_blob,
            },
        )?;
        drop(records);
        seed_tracking_for_revision(&write, revision_id, &asset_key)?;
        write.commit()?;
        Ok(revision_id)
    }

    /// Adds false tracking entries for every normalized record that predates
    /// the tracking overlay. Existing tracking decisions are preserved.
    pub fn initialize_tracking(&self) -> Result<usize> {
        let write = self.db.begin_write()?;
        let keys = {
            let records = write.open_table(RECORDS)?;
            let mut keys = Vec::new();
            for entry in records.iter()? {
                let (key, _) = entry?;
                keys.push(key.value().to_owned());
            }
            keys
        };
        let inserted = insert_missing_tracking(&write, keys)?;
        write.commit()?;
        Ok(inserted)
    }

    pub fn tracking(&self, record_id: &str) -> Result<Option<Tracking>> {
        let read = self.db.begin_read()?;
        let table = match read.open_table(TRACKING) {
            Ok(table) => table,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        table
            .get(record_id)?
            .map(|value| decode(value.value()))
            .transpose()
    }

    /// Updates implementation status without mutating the imported asset.
    pub fn set_tracking(&self, record_id: &str, tracking: Tracking) -> Result<()> {
        if (tracking.fully_respected || tracking.tested) && !tracking.included {
            return Err(Error::InvalidTrackingState);
        }
        let write = self.db.begin_write()?;
        {
            let records = write.open_table(RECORDS)?;
            if records.get(record_id)?.is_none() {
                return Err(Error::UnknownRecord(record_id.to_owned()));
            }
        }
        {
            let mut table = write.open_table(TRACKING)?;
            table.insert(record_id, encode(&tracking)?.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }
    pub fn revision(&self, id: u64) -> Result<Revision> {
        let read = self.db.begin_read()?;
        let t = read.open_table(RECORDS)?;
        let v = t
            .get(format!("r/{id}/revision").as_str())?
            .ok_or(Error::UnknownRevision(id))?;
        let r: Revision = decode(v.value())?;
        if r.state == RevisionState::Complete {
            Ok(r)
        } else {
            Err(Error::UnknownRevision(id))
        }
    }
    pub fn blob(&self, blob_id: &str) -> Result<Vec<u8>> {
        let read = self.db.begin_read()?;
        let t = read.open_table(BLOBS)?;
        let mut out = Vec::new();
        for i in 0.. {
            let key = format!("{blob_id}/{i:08}");
            match t.get(key.as_str())? {
                Some(v) => out.extend_from_slice(v.value().as_ref()),
                None => break,
            }
        }
        Ok(out)
    }
}

struct Prepared {
    doc: gltf::Document,
    buffers: Vec<Vec<u8>>,
}
impl Prepared {
    fn parse(source: &[u8], external: &BTreeMap<String, Vec<u8>>) -> Result<Self> {
        let gltf = gltf::Gltf::from_slice(source)?;
        let doc = gltf.document;
        let mut buffers = Vec::new();
        for buffer in doc.buffers() {
            let bytes = match buffer.source() {
                gltf::buffer::Source::Bin => gltf.blob.clone().unwrap_or_default(),
                gltf::buffer::Source::Uri(uri) => {
                    if let Some(data) = uri.strip_prefix("data:") {
                        let (_, b64) = data.split_once(",").ok_or(Error::DataUri)?;
                        STANDARD.decode(b64).map_err(|_| Error::DataUri)?
                    } else {
                        external
                            .get(uri)
                            .cloned()
                            .ok_or_else(|| Error::MissingBuffer(uri.into()))?
                    }
                }
            };
            if bytes.len() < buffer.length() {
                return Err(Error::BufferLength {
                    index: buffer.index(),
                    actual: bytes.len(),
                    declared: buffer.length(),
                });
            }
            buffers.push(bytes);
        }
        Ok(Self { doc, buffers })
    }
}
fn eid(revision: u64, kind: &str, index: usize) -> String {
    format!("r/{revision}/{kind}/{index}")
}
fn encode<T: Serialize>(v: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(v)?)
}
fn decode<T: for<'a> Deserialize<'a>>(v: &[u8]) -> Result<T> {
    Ok(serde_json::from_slice(v)?)
}
fn put<T: Serialize>(table: &mut redb::Table<&str, &[u8]>, key: &str, value: &T) -> Result<()> {
    let bytes = encode(value)?;
    table.insert(key, bytes.as_slice())?;
    Ok(())
}
fn put_blob(write: &redb::WriteTransaction, id: &str, bytes: &[u8]) -> Result<()> {
    let mut t = write.open_table(BLOBS)?;
    for (i, chunk) in bytes.chunks(CHUNK).enumerate() {
        let key = format!("{id}/{i:08}");
        t.insert(key.as_str(), chunk)?;
    }
    Ok(())
}

fn seed_tracking_for_revision(
    write: &redb::WriteTransaction,
    revision_id: u64,
    asset_key: &str,
) -> Result<()> {
    let prefix = format!("r/{revision_id}/");
    let keys = {
        let records = write.open_table(RECORDS)?;
        let mut keys = vec![asset_key.to_owned()];
        for entry in records.iter()? {
            let (key, _) = entry?;
            if key.value().starts_with(&prefix) {
                keys.push(key.value().to_owned());
            }
        }
        keys
    };
    insert_missing_tracking(write, keys)?;
    Ok(())
}

fn insert_missing_tracking(write: &redb::WriteTransaction, keys: Vec<String>) -> Result<usize> {
    let mut tracking = write.open_table(TRACKING)?;
    let default = encode(&Tracking::default())?;
    let mut inserted = 0;
    for key in keys {
        if tracking.get(key.as_str())?.is_none() {
            tracking.insert(key.as_str(), default.as_slice())?;
            inserted += 1;
        }
    }
    Ok(inserted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn json() -> Vec<u8> {
        br#"{"asset":{"version":"2.0"},"buffers":[{"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA","byteLength":12}],"bufferViews":[{"buffer":0,"byteLength":12}],"accessors":[{"bufferView":0,"componentType":5126,"count":1,"type":"VEC3","min":[0,0,0],"max":[0,0,0]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"nodes":[{"mesh":0},{"children":[0]}],"scenes":[{"nodes":[1]}],"scene":0}"#.to_vec()
    }
    #[test]
    fn persists_graph_and_source() {
        let p = std::env::temp_dir().join(format!("picasso-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let s = Store::create(&p).unwrap();
        let r = s.import("cube", &json(), &BTreeMap::new()).unwrap();
        assert_eq!(s.revision(r).unwrap().state, RevisionState::Complete);
        assert_eq!(s.blob(&format!("r/{r}/blob/source")).unwrap(), json());
        drop(s);
        let s = Store::open(&p).unwrap();
        assert!(s.revision(r).is_ok());
        let _ = fs::remove_file(p);
    }
    #[test]
    fn reimports_are_immutable() {
        let p = std::env::temp_dir().join(format!("picasso-r-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let s = Store::create(&p).unwrap();
        let a = s.import("a", &json(), &BTreeMap::new()).unwrap();
        let b = s.import("a", &json(), &BTreeMap::new()).unwrap();
        assert_ne!(a, b);
        assert_eq!(s.revision(a).unwrap().id, a);
        let _ = fs::remove_file(p);
    }
    #[test]
    fn failed_import_is_invisible() {
        let p = std::env::temp_dir().join(format!("picasso-f-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let s = Store::create(&p).unwrap();
        assert!(
            s.import(
                "bad",
                br#"{"asset":{"version":"2.0"},"buffers":[{"uri":"x.bin","byteLength":4}]}"#,
                &BTreeMap::new()
            )
            .is_err()
        );
        assert!(s.revision(1).is_err());
        let _ = fs::remove_file(p);
    }

    #[test]
    fn tracking_starts_false_and_respects_progression() {
        let p = std::env::temp_dir().join(format!("picasso-t-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let s = Store::create(&p).unwrap();
        let revision = s.import("tracked", &json(), &BTreeMap::new()).unwrap();
        let node = eid(revision, "node", 0);
        assert_eq!(s.tracking(&node).unwrap(), Some(Tracking::default()));
        assert!(
            s.set_tracking(
                &node,
                Tracking {
                    included: false,
                    fully_respected: true,
                    tested: false,
                }
            )
            .is_err()
        );
        let progressed = Tracking {
            included: true,
            fully_respected: false,
            tested: true,
        };
        s.set_tracking(&node, progressed).unwrap();
        assert_eq!(s.tracking(&node).unwrap(), Some(progressed));
        let _ = fs::remove_file(p);
    }
}
