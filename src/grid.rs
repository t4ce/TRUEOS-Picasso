//! Minimal reference-grid geometry.
//!
//! The grid is an indexed line list with three independent line segments.
//! It deliberately carries no colour, transform, or rendering policy, so a
//! consumer can use it as a neutral scene reference.

/// Positions for the three line segments in [`GRID_INDICES`].
///
/// Each pair is a directed, one-unit world-space basis axis.  Consumers own
/// camera/projection policy, so this module deliberately stores these as world
/// coordinates rather than pre-projected screen-space guide lines.
pub static GRID_VERTICES: [[f32; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// `u32` index pairs for [`GRID_VERTICES`], suitable for a line-list draw.
pub static GRID_INDICES: [u32; 6] = [0, 1, 0, 1, 0, 1];
