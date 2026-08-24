//! TRUEOS Picasso.
//!
//! `core` is the bare-metal, `no_std` contract.  The host importer, glTF
//! parser, redb Dealer backend, CLI, and filesystem MASS adapter are enabled
//! only by the `host` feature.
#![cfg_attr(not(feature = "host"), no_std)]

pub mod core;
pub use core::*;

/// Concrete shared-DDR execution ring for a platform that has already mapped
/// one allocation into both CPU and GPU address spaces.  This remains
/// allocation-, I/O-, and runtime-free so it is usable by a TRUEOS Blueprint.
#[path = "Cubism.rs"]
pub mod cubism;

pub use cubism::{
    CoherentVisibility, CpuSlot, CubismError, DealerRingRecord, ExecRing, ExecSlotHeader,
    PublishedSlot, VisibilityOps,
};

#[cfg(feature = "host")]
#[path = "glTFredb.rs"]
mod host;

#[cfg(feature = "host")]
pub use host::*;
