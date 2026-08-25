//! Minimal reference-grid geometry.
//!
//! The grid is an indexed line list with three independent line segments.
//! It deliberately carries no colour, transform, or rendering policy, so a
//! consumer can use it as a neutral scene reference.

/// Positions for the three line segments in [`GRID_INDICES`].
pub static GRID_VERTICES: [[f32; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.20, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.20, 0.0],
    [0.0, 0.0, 0.0],
    [-0.848_528, -0.848_528, 0.0],
];

/// `u32` index pairs for [`GRID_VERTICES`], suitable for a line-list draw.
pub static GRID_INDICES: [u32; 6] = [0, 1, 0, 1, 0, 1];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_three_independent_line_segments() {
        assert_eq!(GRID_VERTICES.len(), 6);
        assert_eq!(GRID_INDICES, [0, 1, 0, 1, 0, 1]);
    }
}
