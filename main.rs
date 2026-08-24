use std::{collections::BTreeMap, env, fs, path::Path};

use trueos_picasso::Store;

fn main() {
    let mut args = env::args().skip(1);
    let Some(source_path) = args.next() else {
        return usage();
    };
    if source_path == "track-init" {
        let Some(database_path) = args.next() else {
            return usage();
        };
        if args.next().is_some() {
            return usage();
        }
        let store = match Store::open(&database_path) {
            Ok(store) => store,
            Err(error) => return fail(&format!("cannot open database: {error}")),
        };
        return match store.initialize_tracking() {
            Ok(count) => println!("initialized tracking for {count} records"),
            Err(error) => fail(&format!("tracking initialization failed: {error}")),
        };
    }
    let Some(database_path) = args.next() else {
        return usage();
    };
    if args.next().is_some() {
        return usage();
    }

    let source = match fs::read(&source_path) {
        Ok(source) => source,
        Err(error) => return fail(&format!("cannot read {source_path}: {error}")),
    };
    let asset_id = Path::new(&source_path)
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or("asset");
    let store = match Store::create(&database_path).or_else(|_| Store::open(&database_path)) {
        Ok(store) => store,
        Err(error) => return fail(&format!("cannot open database: {error}")),
    };
    // Blender commonly writes a `.gltf` beside external `.bin` files.  Resolve
    // those exact URI names here; the library keeps their bytes immutable.
    let mut external = BTreeMap::new();
    if let Ok(parsed) = gltf::Gltf::from_slice(&source) {
        let parent = Path::new(&source_path)
            .parent()
            .unwrap_or_else(|| Path::new("."));
        for buffer in parsed.document.buffers() {
            if let gltf::buffer::Source::Uri(uri) = buffer.source()
                && !uri.starts_with("data:")
                && let Ok(bytes) = fs::read(parent.join(uri))
            {
                external.insert(uri.to_owned(), bytes);
            }
        }
    }
    match store.import(asset_id, &source, &external) {
        Ok(revision) => println!("published asset `{asset_id}`, revision {revision}"),
        Err(error) => fail(&format!("import failed: {error}")),
    }
}

fn usage() {
    eprintln!(
        "usage:\n  trueos-picasso <scene.gltf|scene.glb> <picasso.redb>\n  trueos-picasso track-init <picasso.redb>"
    );
    std::process::exit(2);
}
fn fail(message: &str) {
    eprintln!("{message}");
    std::process::exit(1);
}
