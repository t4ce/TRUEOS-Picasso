//! TRUEOS Picasso.
//!
//! `core` is the bare-metal, `no_std` contract.  The host importer, glTF
//! parser, redb Dealer backend, CLI, and filesystem MASS adapter are enabled
//! only by the `host` feature.
#![cfg_attr(not(feature = "host"), no_std)]

pub mod core;
pub use core::*;

#[cfg(feature = "host")]
#[path = "glTFredb.rs"]
mod host;

#[cfg(feature = "host")]
pub use host::*;
