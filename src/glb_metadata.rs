//! Cheap host inspection before payload decoding or database admission.

use super::{Error, Result};
use std::{fs::File, io::Read, path::Path};

/// A GLB's exact JSON description. No BIN payload is read or decoded.
///
/// This reports authored metadata, including unknown fields and extensions;
/// it does not assert renderer compatibility or validate accessor payloads.
/// Image dimensions are not inferred from texture names or descriptor counts.
#[derive(Clone, Debug)]
pub struct GlbMetadata {
    pub file_size: u64,
    pub json_bytes: usize,
    pub raw_json: Vec<u8>,
    pub descriptor: serde_json::Value,
}

/// Reads only the GLB header, first chunk header, and JSON chunk.
///
/// Full glTF admission remains [`super::Store::import_file`]. In particular,
/// later chunk headers and BIN bytes are deliberately not inspected here.
pub fn inspect_glb_file(path: impl AsRef<Path>) -> Result<GlbMetadata> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    inspect_glb_reader(&mut file, file_size)
}

fn inspect_glb_reader(reader: &mut impl Read, file_size: u64) -> Result<GlbMetadata> {
    let mut header = [0; 20];
    reader.read_exact(&mut header)?;
    let word = |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    if &header[..4] != b"glTF" || word(4) != 2 {
        return Err(Error::InvalidGlb("expected GLB version 2"));
    }
    if u64::from(word(8)) != file_size || !file_size.is_multiple_of(4) {
        return Err(Error::InvalidGlb(
            "declared file length mismatch or alignment",
        ));
    }
    let json_bytes = word(12) as usize;
    if &header[16..20] != b"JSON"
        || json_bytes == 0
        || !json_bytes.is_multiple_of(4)
        || json_bytes as u64 > file_size.saturating_sub(20)
    {
        return Err(Error::InvalidGlb("invalid first JSON chunk"));
    }
    let mut raw_json = vec![0; json_bytes];
    reader.read_exact(&mut raw_json)?;
    let descriptor: serde_json::Value = serde_json::from_slice(&raw_json)?;
    if descriptor
        .get("asset")
        .and_then(|v| v.get("version"))
        .and_then(|v| v.as_str())
        != Some("2.0")
    {
        return Err(Error::InvalidGlb("descriptor asset.version must be 2.0"));
    }
    Ok(GlbMetadata {
        file_size,
        json_bytes,
        raw_json,
        descriptor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn descriptor_prefix() -> Vec<u8> {
        let mut json = br#"{"asset":{"version":"2.0","extras":{"retained":true}},"materials":[{"name":"surface"}]}"#.to_vec();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bytes = b"glTF".to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        // Deliberately promise a BIN payload absent from the reader: inspecting
        // metadata must stop at the JSON boundary instead of touching it.
        bytes.extend_from_slice(&(20 + json.len() as u32 + 1024).to_le_bytes());
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"JSON");
        bytes.extend_from_slice(&json);
        bytes
    }

    #[test]
    fn reads_only_json_and_preserves_unknown_metadata() {
        let prefix = descriptor_prefix();
        let mut reader = Cursor::new(prefix.clone());
        let metadata = inspect_glb_reader(&mut reader, prefix.len() as u64 + 1024).unwrap();
        assert_eq!(reader.position(), prefix.len() as u64);
        assert_eq!(metadata.raw_json, prefix[20..]);
        assert_eq!(metadata.json_bytes, prefix.len() - 20);
        assert_eq!(metadata.descriptor["asset"]["extras"]["retained"], true);
        assert_eq!(metadata.descriptor["materials"][0]["name"], "surface");
    }

    #[test]
    fn rejects_bad_headers_before_allocating_payload() {
        let prefix = descriptor_prefix();
        let size = prefix.len() as u64 + 1024;
        for offset in [0, 4, 8, 12, 16] {
            let mut bad = prefix.clone();
            bad[offset] ^= 1;
            assert!(inspect_glb_reader(&mut Cursor::new(bad), size).is_err());
        }
        let mut huge = prefix;
        huge[12..16].copy_from_slice(&0xffff_fffcu32.to_le_bytes());
        assert!(inspect_glb_reader(&mut Cursor::new(huge), size).is_err());
    }
}
