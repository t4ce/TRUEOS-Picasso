//! Cubism.rs
//!
//! Shared-DDR execution ring for tight CPU -> GPU ownership handoff.
//!
//! Hot path: shared memory + atomics only. No redb access.
//! Redb may persist `DealerRingRecord` and recreate/map the ring at boot.
//!
//! Sequence protocol per slot:
//!   even = FREE at generation N
//!   odd  = PUBLISHED / GPU-owned at generation N
//!   retire -> next even sequence
//!
//! Platform-specific cache maintenance is supplied through `VisibilityOps`.

use core::fmt;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::slice;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};

pub type Result<T> = core::result::Result<T, CubismError>;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugState {
    Free = 0,
    CpuWriting = 1,
    CpuSealed = 2,
    GpuInFlight = 3,
    GpuComplete = 4,
}

#[repr(C, align(64))]
pub struct ExecSlotHeader {
    /// even = free, odd = published; sequence / 2 is generation.
    pub sequence: AtomicU64,
    pub payload_bytes: AtomicU32,
    pub resource_revision: AtomicU64,
    pub gpu_timeline: AtomicU64,
    /// Diagnostics + multi-producer claim guard. Sequence remains authoritative.
    pub debug_state: AtomicU32,
    _reserved: [u8; 28],
}

const _: () = {
    assert!(size_of::<ExecSlotHeader>() == 64);
    assert!(align_of::<ExecSlotHeader>() == 64);
};

/// Persist something like this in redb. Never poll it in the hot path.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DealerRingRecord {
    pub resource_id: u128,
    pub layout_version: u16,
    pub slot_count: u16,
    pub slot_bytes: u64,
    pub payload_kind: u32,
    pub last_checkpoint_generation: u64,
}

/// Hardware/OS-specific CPU<->GPU visibility hooks.
pub trait VisibilityOps {
    /// Called after CPU payload writes, immediately before Release publication.
    fn cpu_make_gpu_visible(&self, cpu_payload: *const u8, gpu_payload: u64, bytes: usize);

    /// Called after timeline completion and before making the slot FREE again.
    fn gpu_make_cpu_visible(&self, cpu_payload: *const u8, gpu_payload: u64, bytes: usize);
}

/// Suitable only when your actual mapping is coherent, or for bring-up/tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoherentVisibility;

impl VisibilityOps for CoherentVisibility {
    #[inline]
    fn cpu_make_gpu_visible(&self, _: *const u8, _: u64, _: usize) {}

    #[inline]
    fn gpu_make_cpu_visible(&self, _: *const u8, _: u64, _: usize) {}
}

/// A live ring over preallocated shared DDR.
///
/// This object does not own the mapping.
pub struct ExecRing {
    base: NonNull<u8>,
    total_bytes: usize,
    slot_count: usize,
    slot_stride: usize,
    payload_capacity: usize,
    gpu_base: u64,
    producer_hint: AtomicU64,
}

unsafe impl Send for ExecRing {}
unsafe impl Sync for ExecRing {}

impl fmt::Debug for ExecRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecRing")
            .field("total_bytes", &self.total_bytes)
            .field("slot_count", &self.slot_count)
            .field("slot_stride", &self.slot_stride)
            .field("payload_capacity", &self.payload_capacity)
            .field("gpu_base", &format_args!("{:#x}", self.gpu_base))
            .finish()
    }
}

impl ExecRing {
    /// Build a ring over an already allocated/mapped shared region.
    ///
    /// # Safety
    /// `cpu_base` must be valid, writable, 64-byte aligned, stable for the
    /// lifetime of the ring, and correspond byte-for-byte to `gpu_base`.
    pub unsafe fn from_raw_parts(
        cpu_base: *mut u8,
        total_bytes: usize,
        gpu_base: u64,
        slot_count: usize,
        slot_stride: usize,
    ) -> Result<Self> {
        if cpu_base.is_null() {
            return Err(CubismError::NullSharedMemory);
        }
        if (cpu_base as usize) % 64 != 0 {
            return Err(CubismError::MisalignedSharedMemory);
        }
        if slot_count == 0 {
            return Err(CubismError::InvalidSlotCount);
        }
        if slot_stride < size_of::<ExecSlotHeader>() || slot_stride % 64 != 0 {
            return Err(CubismError::InvalidSlotStride);
        }

        let required = slot_count
            .checked_mul(slot_stride)
            .ok_or(CubismError::SizeOverflow)?;

        if required > total_bytes {
            return Err(CubismError::SharedMemoryTooSmall {
                required,
                available: total_bytes,
            });
        }

        Ok(Self {
            base: unsafe { NonNull::new_unchecked(cpu_base) },
            total_bytes,
            slot_count,
            slot_stride,
            payload_capacity: slot_stride - size_of::<ExecSlotHeader>(),
            gpu_base,
            producer_hint: AtomicU64::new(0),
        })
    }

    /// Initialize a brand-new ring at generation 0.
    ///
    /// # Safety
    /// Nothing else may touch the mapping while initialization runs.
    pub unsafe fn initialize_fresh(&self) {
        for i in 0..self.slot_count {
            unsafe {
                core::ptr::write(
                    self.header_ptr(i),
                    ExecSlotHeader {
                        sequence: AtomicU64::new(0),
                        payload_bytes: AtomicU32::new(0),
                        resource_revision: AtomicU64::new(0),
                        gpu_timeline: AtomicU64::new(0),
                        debug_state: AtomicU32::new(DebugState::Free as u32),
                        _reserved: [0; 28],
                    },
                );
            }
        }
        self.producer_hint.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    #[inline]
    pub fn slot_stride(&self) -> usize {
        self.slot_stride
    }

    #[inline]
    pub fn payload_capacity(&self) -> usize {
        self.payload_capacity
    }

    /// Nonblocking fast path. Add your async waiter around `RingFull`.
    pub fn try_acquire(&self) -> Result<CpuSlot<'_>> {
        let start = self.producer_hint.fetch_add(1, Ordering::Relaxed) as usize;

        for n in 0..self.slot_count {
            let index = (start + n) % self.slot_count;
            let h = unsafe { &*self.header_ptr(index) };
            let seq = h.sequence.load(Ordering::Acquire);

            if seq & 1 != 0 {
                continue;
            }

            if h.debug_state
                .compare_exchange(
                    DebugState::Free as u32,
                    DebugState::CpuWriting as u32,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                continue;
            }

            // Sequence is authoritative. Recheck after taking CPU claim guard.
            if h.sequence.load(Ordering::Acquire) != seq {
                h.debug_state
                    .store(DebugState::Free as u32, Ordering::Release);
                continue;
            }

            return Ok(CpuSlot {
                ring: self,
                index,
                generation: seq / 2,
                published: false,
                _not_send: PhantomData,
            });
        }

        Err(CubismError::RingFull)
    }

    /// Re-open a known published generation on the executor side.
    pub fn published(&self, index: usize, generation: u64) -> Result<PublishedSlot<'_>> {
        self.validate_index(index)?;
        let h = unsafe { &*self.header_ptr(index) };
        let expected = published_sequence(generation)?;
        let actual = h.sequence.load(Ordering::Acquire);

        if actual != expected {
            return Err(CubismError::GenerationMismatch { expected, actual });
        }

        Ok(PublishedSlot {
            ring: self,
            index,
            generation,
        })
    }

    /// Free exactly one generation after its GPU timeline has completed.
    pub fn retire<V: VisibilityOps>(
        &self,
        index: usize,
        generation: u64,
        completed_timeline: u64,
        visibility: &V,
    ) -> Result<()> {
        self.validate_index(index)?;
        let h = unsafe { &*self.header_ptr(index) };

        let expected = published_sequence(generation)?;
        let actual = h.sequence.load(Ordering::Acquire);
        if actual != expected {
            return Err(CubismError::GenerationMismatch { expected, actual });
        }

        let submitted = h.gpu_timeline.load(Ordering::Acquire);
        if submitted == 0 {
            return Err(CubismError::NotSubmitted);
        }
        if completed_timeline < submitted {
            return Err(CubismError::TimelineNotComplete {
                submitted,
                completed: completed_timeline,
            });
        }

        let bytes = h.payload_bytes.load(Ordering::Acquire) as usize;
        let cpu_payload = self.payload_ptr(index);
        let gpu_payload = self.gpu_payload_address(index);

        visibility.gpu_make_cpu_visible(cpu_payload.cast_const(), gpu_payload, bytes);
        fence(Ordering::Release);

        h.debug_state
            .store(DebugState::GpuComplete as u32, Ordering::Relaxed);
        h.payload_bytes.store(0, Ordering::Relaxed);
        h.resource_revision.store(0, Ordering::Relaxed);
        h.gpu_timeline.store(0, Ordering::Relaxed);

        let next = generation
            .checked_add(1)
            .and_then(|g| g.checked_mul(2))
            .ok_or(CubismError::GenerationOverflow)?;

        h.sequence.store(next, Ordering::Release);
        h.debug_state
            .store(DebugState::Free as u32, Ordering::Release);

        Ok(())
    }

    #[inline]
    pub fn gpu_payload_address(&self, index: usize) -> u64 {
        self.gpu_base + (index * self.slot_stride + size_of::<ExecSlotHeader>()) as u64
    }

    #[inline]
    fn validate_index(&self, index: usize) -> Result<()> {
        if index < self.slot_count {
            Ok(())
        } else {
            Err(CubismError::InvalidSlotIndex(index))
        }
    }

    #[inline]
    unsafe fn header_ptr(&self, index: usize) -> *mut ExecSlotHeader {
        unsafe {
            self.base
                .as_ptr()
                .add(index * self.slot_stride)
                .cast::<ExecSlotHeader>()
        }
    }

    #[inline]
    fn payload_ptr(&self, index: usize) -> *mut u8 {
        unsafe {
            self.base
                .as_ptr()
                .add(index * self.slot_stride + size_of::<ExecSlotHeader>())
        }
    }
}

/// Producer-owned direct view into one shared-DDR payload.
pub struct CpuSlot<'a> {
    ring: &'a ExecRing,
    index: usize,
    generation: u64,
    published: bool,
    _not_send: PhantomData<*mut ()>,
}

impl<'a> CpuSlot<'a> {
    #[inline]
    pub fn slot_index(&self) -> usize {
        self.index
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.ring.payload_capacity
    }

    /// Zero-copy mutable payload in shared DDR.
    #[inline]
    pub fn payload_mut(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self.ring.payload_ptr(self.index),
                self.ring.payload_capacity,
            )
        }
    }

    /// Seal CPU writes and transfer ownership to executor/GPU.
    pub fn publish<V: VisibilityOps>(
        mut self,
        payload_bytes: u32,
        resource_revision: u64,
        visibility: &V,
    ) -> Result<PublishedSlot<'a>> {
        if payload_bytes as usize > self.ring.payload_capacity {
            return Err(CubismError::PayloadTooLarge {
                requested: payload_bytes as usize,
                capacity: self.ring.payload_capacity,
            });
        }

        let h = unsafe { &*self.ring.header_ptr(self.index) };

        h.payload_bytes.store(payload_bytes, Ordering::Relaxed);
        h.resource_revision
            .store(resource_revision, Ordering::Relaxed);
        h.gpu_timeline.store(0, Ordering::Relaxed);

        let cpu_payload = self.ring.payload_ptr(self.index);
        let gpu_payload = self.ring.gpu_payload_address(self.index);

        visibility.cpu_make_gpu_visible(
            cpu_payload.cast_const(),
            gpu_payload,
            payload_bytes as usize,
        );

        fence(Ordering::Release);
        h.debug_state
            .store(DebugState::CpuSealed as u32, Ordering::Relaxed);
        h.sequence
            .store(published_sequence(self.generation)?, Ordering::Release);

        self.published = true;

        Ok(PublishedSlot {
            ring: self.ring,
            index: self.index,
            generation: self.generation,
        })
    }
}

impl Drop for CpuSlot<'_> {
    fn drop(&mut self) {
        if self.published {
            return;
        }

        // Abandoned before publication: no generation change.
        let h = unsafe { &*self.ring.header_ptr(self.index) };
        h.payload_bytes.store(0, Ordering::Relaxed);
        h.resource_revision.store(0, Ordering::Relaxed);
        h.gpu_timeline.store(0, Ordering::Relaxed);
        h.debug_state
            .store(DebugState::Free as u32, Ordering::Release);
    }
}

/// Immutable executor/GPU-owned view of an exact published generation.
#[derive(Clone, Copy)]
pub struct PublishedSlot<'a> {
    ring: &'a ExecRing,
    index: usize,
    generation: u64,
}

impl<'a> PublishedSlot<'a> {
    #[inline]
    pub fn slot_index(&self) -> usize {
        self.index
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[inline]
    pub fn gpu_address(&self) -> u64 {
        self.ring.gpu_payload_address(self.index)
    }

    #[inline]
    pub fn payload_bytes(&self) -> u32 {
        unsafe { &*self.ring.header_ptr(self.index) }
            .payload_bytes
            .load(Ordering::Acquire)
    }

    #[inline]
    pub fn resource_revision(&self) -> u64 {
        unsafe { &*self.ring.header_ptr(self.index) }
            .resource_revision
            .load(Ordering::Acquire)
    }

    #[inline]
    pub fn gpu_timeline(&self) -> u64 {
        unsafe { &*self.ring.header_ptr(self.index) }
            .gpu_timeline
            .load(Ordering::Acquire)
    }

    /// Attach the exact timeline point returned by your GPU executor.
    pub fn mark_in_flight(&self, timeline: u64) -> Result<()> {
        if timeline == 0 {
            return Err(CubismError::InvalidTimeline);
        }

        let h = unsafe { &*self.ring.header_ptr(self.index) };
        let expected = published_sequence(self.generation)?;
        let actual = h.sequence.load(Ordering::Acquire);

        if actual != expected {
            return Err(CubismError::GenerationMismatch { expected, actual });
        }

        h.debug_state
            .compare_exchange(
                DebugState::CpuSealed as u32,
                DebugState::GpuInFlight as u32,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| CubismError::NotCpuSealed)?;
        h.gpu_timeline.store(timeline, Ordering::Release);
        Ok(())
    }

    /// CPU diagnostic/read-only view while this generation is published.
    #[inline]
    pub fn payload(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self.ring.payload_ptr(self.index).cast_const(),
                self.payload_bytes() as usize,
            )
        }
    }
}

#[inline]
fn published_sequence(generation: u64) -> Result<u64> {
    generation
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or(CubismError::GenerationOverflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CubismError {
    NullSharedMemory,
    MisalignedSharedMemory,
    InvalidSlotCount,
    InvalidSlotStride,
    InvalidSlotIndex(usize),
    SharedMemoryTooSmall { required: usize, available: usize },
    PayloadTooLarge { requested: usize, capacity: usize },
    RingFull,
    SizeOverflow,
    GenerationOverflow,
    InvalidTimeline,
    NotCpuSealed,
    NotSubmitted,
    GenerationMismatch { expected: u64, actual: u64 },
    TimelineNotComplete { submitted: u64, completed: u64 },
}

impl fmt::Display for CubismError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullSharedMemory => write!(f, "shared memory pointer is null"),
            Self::MisalignedSharedMemory => write!(f, "shared memory base must be 64-byte aligned"),
            Self::InvalidSlotCount => write!(f, "slot_count must be non-zero"),
            Self::InvalidSlotStride => write!(f, "slot_stride must be >= 64 and 64-byte aligned"),
            Self::InvalidSlotIndex(i) => write!(f, "invalid slot index {i}"),
            Self::SharedMemoryTooSmall {
                required,
                available,
            } => {
                write!(
                    f,
                    "shared memory too small: need {required}, have {available}"
                )
            }
            Self::PayloadTooLarge {
                requested,
                capacity,
            } => {
                write!(
                    f,
                    "payload too large: requested {requested}, capacity {capacity}"
                )
            }
            Self::RingFull => write!(f, "no free execution slot"),
            Self::SizeOverflow => write!(f, "ring size overflow"),
            Self::GenerationOverflow => write!(f, "slot generation overflow"),
            Self::InvalidTimeline => write!(f, "timeline 0 is reserved"),
            Self::NotCpuSealed => write!(f, "slot is not sealed for executor submission"),
            Self::NotSubmitted => write!(f, "slot has no GPU timeline attached"),
            Self::GenerationMismatch { expected, actual } => {
                write!(
                    f,
                    "generation mismatch: expected sequence {expected}, found {actual}"
                )
            }
            Self::TimelineNotComplete {
                submitted,
                completed,
            } => {
                write!(
                    f,
                    "timeline incomplete: submitted {submitted}, completed {completed}"
                )
            }
        }
    }
}

impl core::error::Error for CubismError {}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::alloc::{Layout, alloc_zeroed, dealloc};

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
            ExecRing::from_raw_parts(memory.ptr, BYTES, 0x1_0000_0000, COUNT, STRIDE).unwrap()
        };
        unsafe { ring.initialize_fresh() };

        let visibility = CoherentVisibility;
        let mut cpu = ring.try_acquire().unwrap();
        let index = cpu.slot_index();
        let generation = cpu.generation();

        cpu.payload_mut()[..5].copy_from_slice(b"hello");
        let published = cpu.publish(5, 42, &visibility).unwrap();

        assert_eq!(published.payload(), b"hello");
        assert_eq!(published.resource_revision(), 42);

        published.mark_in_flight(100).unwrap();
        assert_eq!(
            published.mark_in_flight(101),
            Err(CubismError::NotCpuSealed)
        );
        assert!(ring.retire(index, generation, 99, &visibility).is_err());
        ring.retire(index, generation, 100, &visibility).unwrap();
    }
}
