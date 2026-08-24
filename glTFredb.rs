//! Picasso's durable, glTF-faithful import boundary.
//!
//! A collection is deliberately absent here: glTF nodes describe transform
//! hierarchy, while collections are an editor concern and must not imply parentage.

use std::{collections::BTreeMap, path::Path};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const META: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_meta_v1");
const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_records_v1");
const BLOBS: TableDefinition<&str, &[u8]> = TableDefinition::new("picasso_blob_chunks_v1");
const CHUNK: usize = 64 * 1024;

#[derive(Debug, Error)]
pub enum Error {
    #[error("database: {0}")]
    Database(#[from] redb::Error),
    #[error("database open: {0}")]
    DatabaseOpen(#[from] redb::DatabaseError),
    #[error("database transaction: {0}")]
    Transaction(#[from] redb::TransactionError),
    #[error("database table: {0}")]
    Table(#[from] redb::TableError),
    #[error("database storage: {0}")]
    Storage(#[from] redb::StorageError),
    #[error("database commit: {0}")]
    Commit(#[from] redb::CommitError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid glTF: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("invalid record encoding: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing payload for external buffer `{0}`")]
    MissingBuffer(String),
    #[error("invalid data URI")]
    DataUri,
    #[error("buffer {index} is {actual} bytes; glTF declares {declared}")]
    BufferLength {
        index: usize,
        actual: usize,
        declared: usize,
    },
    #[error("no published revision {0}")]
    UnknownRevision(u64),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RevisionState {
    Staging,
    Complete,
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
        write.commit()?;
        Ok(revision_id)
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
}
