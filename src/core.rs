//! Bare-metal Picasso scene descriptors.
//!
//! These describe data already materialized in shared DDR5. The concrete
//! ownership-transfer protocol is [`crate::cubism::ExecRing`]; keeping that
//! protocol in one module prevents a Blueprint and executor from silently
//! using incompatible slot state machines.

/// Stable identity of a prepared Dealer resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct ResourceId(pub u64);

/// Opaque identity of one live, materialized shared resource.
///
/// The platform resolves this to its own buffer/VM mapping. Picasso never
/// observes the corresponding GPU virtual address or buffer handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct SharedResourceId(pub u64);

/// A sealed immutable byte range in a prepared resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedRange {
    pub resource: ResourceId,
    pub offset: u64,
    pub byte_length: u64,
    /// Changes only when Dealer publishes a new prepared resource revision.
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineLoop,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
}

/// One primitive directly executable by a topology-capable backend. Picasso
/// never rewrites its topology merely to fit a renderer shortcut.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutablePrimitive {
    pub topology: PrimitiveTopology,
    pub vertices: PreparedRange,
    pub indices: Option<PreparedRange>,
    pub index_format: Option<IndexFormat>,
    pub vertex_stride: u32,
    pub vertex_count: u32,
    pub index_count: u32,
}

/// Stable index into a retained transform-state table.
///
/// This is deliberately not a GPU address. The executor resolves the table's
/// [`SharedResourceId`] and bounds-checks this index before dispatch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct TransformId(pub u32);

/// Scale, quaternion rotation, then translation.
///
/// The explicit padding gives the shared-memory form a stable 48-byte stride.
/// Zero scale is valid: animation and topology tools may intentionally
/// collapse selected vertices. A consumer must reject non-finite values and a
/// zero-length quaternion before publishing work.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(16))]
pub struct TransformValue {
    pub translation: [f32; 3],
    pub translation_pad: f32,
    /// Quaternion in x, y, z, w order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub scale_pad: f32,
}

impl TransformValue {
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        translation_pad: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        scale_pad: 0.0,
    };

    pub fn is_valid(self) -> bool {
        let finite = self
            .translation
            .into_iter()
            .chain(self.rotation)
            .chain(self.scale)
            .all(f32::is_finite);
        let norm_squared = self.rotation.into_iter().map(|v| v * v).sum::<f32>();
        finite && norm_squared > 1.0e-12
    }
}

impl Default for TransformValue {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Mesh-local vertices selected by one inline transform value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VertexSelection {
    All,
    Range {
        first_vertex: u32,
        vertex_count: u32,
    },
    /// Packed `u32` mesh-local vertex indices prepared by Dealer.
    IndexList {
        indices: PreparedRange,
        index_count: u32,
    },
}

/// One literal transform operation.
///
/// A worklist of these records covers the broadcast case with one `All`
/// record, arbitrary subsets with ranges/index lists, and fully independent
/// motion with one single-vertex record per vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformByValue {
    pub selection: VertexSelection,
    pub value: TransformValue,
}

/// Mutable, retained transform states restored with a Cubism shared region.
/// `generation` lets the GPU skip an unchanged table without consulting redb.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformStateRange {
    pub resource: SharedResourceId,
    pub offset: u64,
    pub byte_length: u64,
    pub state_count: u32,
    pub state_stride: u32,
    pub generation: u64,
}

/// Mesh-local references into a retained [`TransformStateRange`].
///
/// `references` contains packed `u32` [`TransformId`] values. It may contain
/// one reference for the whole mesh, one per authored range, or one per
/// vertex. Geometry remains immutable and is never duplicated merely because
/// several instances or vertices select different transforms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformRefList {
    pub states: TransformStateRange,
    pub references: PreparedRange,
    pub reference_count: u32,
}

/// Where the common transform evaluator leaves its result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransformExecution {
    /// Resolve references while drawing; canonical vertices remain untouched.
    EvaluateAtDraw,
    /// Write a derived vertex stream from canonical source geometry.
    Materialize,
    /// The GPU result becomes the authoritative input of the next generation.
    Evolve,
}

const _: () = {
    assert!(core::mem::size_of::<TransformValue>() == 48);
    assert!(core::mem::align_of::<TransformValue>() == 16);
};
