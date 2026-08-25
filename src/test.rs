//! Consolidated std-backed test suite for Picasso.

extern crate std;

mod core_tests {
    use crate::core::*;

    #[test]
    fn identity_is_valid_and_zero_scale_is_allowed() {
        assert!(TransformValue::IDENTITY.is_valid());
        let collapsed = TransformValue {
            scale: [0.0; 3],
            ..TransformValue::IDENTITY
        };
        assert!(collapsed.is_valid());
    }

    #[test]
    fn transform_references_are_resource_relative() {
        let refs = TransformRefList {
            states: TransformStateRange {
                resource: SharedResourceId(7),
                offset: 64,
                byte_length: 4 * 48,
                state_count: 4,
                state_stride: 48,
                generation: 9,
            },
            references: PreparedRange {
                resource: ResourceId(11),
                offset: 0,
                byte_length: 4 * 4,
                revision: 1,
            },
            reference_count: 4,
        };
        assert_eq!(refs.states.resource, SharedResourceId(7));
        assert_eq!(refs.references.byte_length, 16);
    }
}

mod grid_tests {
    use crate::grid::*;

    #[test]
    fn grid_is_three_independent_line_segments() {
        assert_eq!(GRID_VERTICES.len(), 6);
        assert_eq!(GRID_INDICES, [0, 1, 0, 1, 0, 1]);
    }
}

mod cam_tests {
    use crate::cam::*;

    fn camera() -> Camera {
        Camera {
            position: [0.0; 3],
            rotation: Quaternion::IDENTITY,
            projection: Projection::Perspective {
                yfov: 1.0,
                znear: 0.1,
                zfar: None,
                aspect_ratio: None,
            },
        }
    }

    #[test]
    fn wasd_moves_at_configured_speed() {
        let mut fly = FlyCam::new(camera(), 4.0);
        fly.step(
            Wasd {
                w: true,
                ..Wasd::default()
            },
            0.5,
        );
        assert_eq!(fly.camera.position, [0.0, 0.0, -2.0]);
        fly.set_speed(8.0);
        assert_eq!(fly.speed(), 8.0);
    }

    #[test]
    fn mouse_look_uses_a_unit_quaternion() {
        let mut fly = FlyCam::new(camera(), 1.0);
        fly.look(30.0, -12.0);
        let norm = fly
            .camera
            .rotation
            .0
            .into_iter()
            .map(|v| v * v)
            .sum::<f32>();
        assert!((norm - 1.0).abs() < 1.0e-5);
        assert!(fly.camera.rotation.rotate([0.0, 0.0, -1.0])[0] > 0.0);
    }
}

mod cubism_tests {
    use crate::SharedResourceId;
    use crate::cubism::*;

    extern crate std;

    use std::alloc::{Layout, alloc_zeroed, dealloc};
    use std::sync::Mutex;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CacheOp {
        Data(SharedByteRange),
        Publish(SharedByteRange),
        Invalidate(SharedByteRange),
    }

    struct RecordedVisibility(Mutex<std::vec::Vec<CacheOp>>);

    impl RecordedVisibility {
        fn new() -> Self {
            Self(Mutex::new(std::vec::Vec::new()))
        }

        fn operations(&self) -> std::vec::Vec<CacheOp> {
            self.0.lock().unwrap().clone()
        }
    }

    impl VisibilityOps for RecordedVisibility {
        fn cpu_make_gpu_visible(&self, range: SharedByteRange) -> Result<()> {
            self.0.lock().unwrap().push(CacheOp::Data(range));
            Ok(())
        }

        fn cpu_publish_slot(&self, range: SharedByteRange) -> Result<()> {
            self.0.lock().unwrap().push(CacheOp::Publish(range));
            Ok(())
        }

        fn gpu_make_cpu_visible(&self, range: SharedByteRange) -> Result<()> {
            self.0.lock().unwrap().push(CacheOp::Invalidate(range));
            Ok(())
        }
    }

    struct Region {
        ptr: *mut u8,
        layout: Layout,
    }

    impl Region {
        fn new(bytes: usize) -> Self {
            let layout = Layout::from_size_align(bytes, 64).unwrap();
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }
    }

    impl Drop for Region {
        fn drop(&mut self) {
            unsafe { dealloc(self.ptr, self.layout) }
        }
    }

    #[test]
    fn publish_submit_retire_reuse() {
        const COUNT: usize = 4;
        const STRIDE: usize = 256;
        const BYTES: usize = COUNT * STRIDE;

        let memory = Region::new(BYTES);
        let ring = unsafe {
            ExecRing::from_raw_parts(memory.ptr, BYTES, SharedResourceId(7), COUNT, STRIDE).unwrap()
        };
        unsafe { ring.initialize_fresh() };

        let visibility = CoherentVisibility;
        let mut cpu = ring.try_acquire().unwrap();
        let index = cpu.slot_index();
        let generation = cpu.generation();

        cpu.payload_mut()[..5].copy_from_slice(b"hello");
        let published = cpu.publish(5, 42, &visibility).unwrap();

        assert_eq!(published.payload(), b"hello");
        assert_eq!(
            published.payload_range(),
            SharedByteRange {
                resource: SharedResourceId(7),
                offset: 64,
                byte_length: 5,
            }
        );
        assert_eq!(published.resource_revision(), 42);

        published.mark_in_flight(100).unwrap();
        assert_eq!(
            published.mark_in_flight(101),
            Err(CubismError::NotCpuSealed)
        );
        assert!(ring.retire(index, generation, 99, &visibility).is_err());
        ring.retire(index, generation, 100, &visibility).unwrap();
    }

    #[test]
    fn noncoherent_visibility_orders_data_then_control_then_retire_invalidate() {
        const STRIDE: usize = 256;
        let memory = Region::new(STRIDE);
        let ring = unsafe {
            ExecRing::from_raw_parts(memory.ptr, STRIDE, SharedResourceId(11), 1, STRIDE).unwrap()
        };
        unsafe { ring.initialize_fresh() };
        let visibility = RecordedVisibility::new();
        let mut cpu = ring.try_acquire().unwrap();
        cpu.payload_mut()[..3].copy_from_slice(b"cmd");
        let published = cpu.publish(3, 1, &visibility).unwrap();
        published.mark_in_flight(1).unwrap();
        ring.retire(0, 0, 1, &visibility).unwrap();

        let range = |offset, byte_length| SharedByteRange {
            resource: SharedResourceId(11),
            offset,
            byte_length,
        };
        assert_eq!(
            visibility.operations(),
            std::vec![
                CacheOp::Data(range(0, 67)),
                CacheOp::Publish(range(0, 64)),
                CacheOp::Invalidate(range(64, 3)),
            ]
        );
    }
}

mod mass_tests {
    use crate::mass::*;

    use super::*;
    use std::fs;

    #[test]
    fn publishes_reopens_and_reads_ranges() {
        let base = std::env::temp_dir().join(format!("picasso-mass-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let catalog = base.join("catalog.redb");
        let payloads = base.join("payloads");

        let store = Store::open(&catalog, &payloads).unwrap();
        let id = store.put("texture/albedo", b"0123456789").unwrap();
        assert_eq!(store.read_range(id, 3, 4).unwrap(), b"3456");
        assert_eq!(store.info(id).unwrap().byte_length, 10);
        drop(store);

        let reopened = Store::open(&catalog, &payloads).unwrap();
        assert_eq!(reopened.read(id).unwrap(), b"0123456789");
        assert!(reopened.read_range(id, 9, 2).is_err());
        let _ = fs::remove_dir_all(base);
    }
}

mod gltf_tests {
    use crate::*;
    use std::collections::BTreeMap;

    use super::*;
    use std::fs;
    fn json() -> Vec<u8> {
        br#"{"asset":{"version":"2.0"},"buffers":[{"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA","byteLength":12}],"bufferViews":[{"buffer":0,"byteLength":12}],"accessors":[{"bufferView":0,"componentType":5126,"count":1,"type":"VEC3","min":[0,0,0],"max":[0,0,0]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"nodes":[{"mesh":0},{"children":[0]}],"scenes":[{"nodes":[1]}],"scene":0}"#.to_vec()
    }
    fn camera_json() -> Vec<u8> {
        br#"{"asset":{"version":"2.0"},"cameras":[{"name":"Main","type":"perspective","perspective":{"yfov":1.0,"znear":0.1}}],"nodes":[{"camera":0,"translation":[1,2,3]}],"scenes":[{"nodes":[0]}],"scene":0}"#.to_vec()
    }

    #[test]
    fn camera_projection_and_node_pose_are_normalized_separately() {
        let p = std::env::temp_dir().join(format!("picasso-camera-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let store = Store::create(p.to_str().unwrap()).unwrap();
        let revision = store
            .import("camera", &camera_json(), &BTreeMap::new())
            .unwrap();
        let node = store.node(&eid(revision, "node", 0)).unwrap();
        let camera_id = node.camera.unwrap();
        assert_eq!(node.matrix[3][..3], [1.0, 2.0, 3.0]);
        assert!(matches!(
            store.camera(&camera_id).unwrap().projection,
            crate::cam::Projection::Perspective { zfar: None, .. }
        ));
        let _ = fs::remove_file(p);
    }
    #[test]
    fn persists_graph_and_source() {
        let p = std::env::temp_dir().join(format!("picasso-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let s = Store::create(p.to_str().unwrap()).unwrap();
        let r = s.import("cube", &json(), &BTreeMap::new()).unwrap();
        assert_eq!(s.revision(r).unwrap().state, RevisionState::Complete);
        assert_eq!(s.blob(&format!("r/{r}/blob/source")).unwrap(), json());
        drop(s);
        let s = Store::open(p.to_str().unwrap()).unwrap();
        assert!(s.revision(r).is_ok());
        let _ = fs::remove_file(p);
    }
    #[test]
    fn reimports_are_immutable() {
        let p = std::env::temp_dir().join(format!("picasso-r-{}", std::process::id()));
        let _ = fs::remove_file(&p);
        let s = Store::create(p.to_str().unwrap()).unwrap();
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
        let s = Store::create(p.to_str().unwrap()).unwrap();
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
        let s = Store::create(p.to_str().unwrap()).unwrap();
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

mod glb_library_tests {
    use crate::glb_library;

    #[test]
    fn asset_id_normalization() {
        assert_eq!(
            glb_library::normalize_asset_id("/models/robot/").unwrap(),
            "models/robot"
        );
        assert!(glb_library::normalize_asset_id("").is_err());
        assert!(glb_library::normalize_asset_id("///").is_err());
    }
}
