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

/// GPU virtual address of shared DDR5 already mapped by the platform.
///
/// This is intentionally only an address in the vGPU-visible allocation. It
/// is not a GuC, PPGTT, MMIO, engine-context, or other driver-private handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct GpuAddress(pub u64);

/// A sealed immutable byte range in a prepared resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedRange {
    pub resource: ResourceId,
    pub gpu_address: GpuAddress,
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
