//! Bare-metal Picasso execution contract.
//!
//! These are descriptors for data already materialized in shared DDR5.  They
//! intentionally do not parse glTF, open redb, allocate, or perform I/O.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Stable identity of a prepared Dealer resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct ResourceId(pub u64);

/// GPU virtual address of shared DDR5 already mapped by the platform.
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotState {
    Free,
    CpuWriting,
    CpuSealed,
    GpuInFlight,
    GpuComplete,
    /// The header was not initialized by this Picasso ABI version. It is never
    /// treated as reclaimable memory.
    Unknown,
}

impl SlotState {
    const fn bits(self) -> u32 {
        match self {
            Self::Free => 0,
            Self::CpuWriting => 1,
            Self::CpuSealed => 2,
            Self::GpuInFlight => 3,
            Self::GpuComplete => 4,
            Self::Unknown => u32::MAX,
        }
    }
    const fn from_bits(bits: u32) -> Self {
        match bits {
            0 => Self::Free,
            1 => Self::CpuWriting,
            2 => Self::CpuSealed,
            3 => Self::GpuInFlight,
            4 => Self::GpuComplete,
            _ => Self::Unknown,
        }
    }
}

/// Cache-line aligned control block placed beside a shared-DDR slot. Payload
/// bytes live in a platform allocation whose GPU address is in `ExecutionSlot`.
#[repr(C, align(64))]
pub struct SlotHeader {
    state: AtomicU32,
    payload_bytes: AtomicU32,
    generation: AtomicU64,
    timeline: AtomicU64,
}

impl SlotHeader {
    pub const fn new() -> Self {
        Self {
            state: AtomicU32::new(0),
            payload_bytes: AtomicU32::new(0),
            generation: AtomicU64::new(0),
            timeline: AtomicU64::new(0),
        }
    }
    pub fn state(&self) -> SlotState {
        SlotState::from_bits(self.state.load(Ordering::Acquire))
    }
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
    pub fn timeline(&self) -> u64 {
        self.timeline.load(Ordering::Acquire)
    }
}

impl Default for SlotHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Platform-owned shared allocation. `cpu_payload` is provided at acquire time
/// so no_std Picasso never assumes a virtual-memory or allocator policy.
#[derive(Clone, Copy)]
pub struct ExecutionSlot<'a> {
    pub header: &'a SlotHeader,
    pub gpu_address: GpuAddress,
    pub capacity: u32,
}

pub struct ExecutionRing<'a> {
    slots: &'a [ExecutionSlot<'a>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RingError {
    InvalidSlot,
    PayloadTooLarge,
    NotFree,
    NotCpuWriting,
    NotCpuSealed,
    NotGpuInFlight,
}

pub struct CpuWrite<'a> {
    slot: ExecutionSlot<'a>,
    payload: &'a mut [u8],
}
pub struct SealedExecution<'a> {
    slot: ExecutionSlot<'a>,
    payload_bytes: u32,
    generation: u64,
}

impl<'a> ExecutionRing<'a> {
    pub const fn new(slots: &'a [ExecutionSlot<'a>]) -> Self {
        Self { slots }
    }
    pub const fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Atomically acquires a free slot. The caller supplies the CPU mapping of
    /// the exact allocation described by that slot.
    pub fn try_begin_write(
        &self,
        index: usize,
        payload: &'a mut [u8],
    ) -> Result<CpuWrite<'a>, RingError> {
        let slot = *self.slots.get(index).ok_or(RingError::InvalidSlot)?;
        if payload.len() > slot.capacity as usize {
            return Err(RingError::PayloadTooLarge);
        }
        slot.header
            .state
            .compare_exchange(
                SlotState::Free.bits(),
                SlotState::CpuWriting.bits(),
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .map_err(|_| RingError::NotFree)?;
        Ok(CpuWrite { slot, payload })
    }

    /// Seals CPU writes with release ordering. The platform submission path is
    /// responsible for any Intel cache/domain flush required by its mapping.
    pub fn seal(write: CpuWrite<'a>, payload_bytes: u32) -> Result<SealedExecution<'a>, RingError> {
        if payload_bytes as usize > write.payload.len() {
            return Err(RingError::PayloadTooLarge);
        }
        write
            .slot
            .header
            .payload_bytes
            .store(payload_bytes, Ordering::Relaxed);
        let generation = write
            .slot
            .header
            .generation
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        write
            .slot
            .header
            .state
            .compare_exchange(
                SlotState::CpuWriting.bits(),
                SlotState::CpuSealed.bits(),
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map_err(|_| RingError::NotCpuWriting)?;
        Ok(SealedExecution {
            slot: write.slot,
            payload_bytes,
            generation,
        })
    }

    /// Called only after the executor has accepted the command stream.
    pub fn mark_submitted(
        &self,
        sealed: SealedExecution<'a>,
        timeline: u64,
    ) -> Result<(), RingError> {
        sealed
            .slot
            .header
            .timeline
            .store(timeline, Ordering::Relaxed);
        sealed
            .slot
            .header
            .state
            .compare_exchange(
                SlotState::CpuSealed.bits(),
                SlotState::GpuInFlight.bits(),
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map(|_| ())
            .map_err(|_| RingError::NotCpuSealed)
    }

    /// Called by the executor's exact completion/fence retirement path.
    pub fn mark_complete(&self, index: usize, timeline: u64) -> Result<(), RingError> {
        let slot = *self.slots.get(index).ok_or(RingError::InvalidSlot)?;
        if slot.header.timeline() != timeline {
            return Err(RingError::NotGpuInFlight);
        }
        slot.header
            .state
            .compare_exchange(
                SlotState::GpuInFlight.bits(),
                SlotState::GpuComplete.bits(),
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map(|_| ())
            .map_err(|_| RingError::NotGpuInFlight)
    }

    /// Reclaims a completed slot for a future CPU generation.
    pub fn reclaim(&self, index: usize) -> Result<(), RingError> {
        let slot = *self.slots.get(index).ok_or(RingError::InvalidSlot)?;
        slot.header
            .state
            .compare_exchange(
                SlotState::GpuComplete.bits(),
                SlotState::Free.bits(),
                Ordering::Release,
                Ordering::Relaxed,
            )
            .map(|_| ())
            .map_err(|_| RingError::NotGpuInFlight)
    }
}

impl<'a> CpuWrite<'a> {
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.payload
    }
}
impl<'a> SealedExecution<'a> {
    pub const fn gpu_address(&self) -> GpuAddress {
        self.slot.gpu_address
    }
    pub const fn payload_bytes(&self) -> u32 {
        self.payload_bytes
    }
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_transfers_one_slot_without_copying() {
        let header = SlotHeader::new();
        let slots = [ExecutionSlot {
            header: &header,
            gpu_address: GpuAddress(0x4000),
            capacity: 64,
        }];
        let ring = ExecutionRing::new(&slots);
        let mut payload = [0u8; 64];
        let mut write = ring.try_begin_write(0, &mut payload).unwrap();
        write.bytes_mut()[0..4].copy_from_slice(b"cube");
        let sealed = ExecutionRing::seal(write, 4).unwrap();
        assert_eq!(sealed.gpu_address(), GpuAddress(0x4000));
        ring.mark_submitted(sealed, 9).unwrap();
        assert_eq!(header.state(), SlotState::GpuInFlight);
        ring.mark_complete(0, 9).unwrap();
        ring.reclaim(0).unwrap();
        assert_eq!(header.state(), SlotState::Free);
        assert_eq!(&payload[..4], b"cube");
    }
}
