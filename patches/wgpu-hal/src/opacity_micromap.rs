//! openRTS's fork: opacity micromaps (VK_EXT_opacity_micromap, D3D12's
//! DXR 1.2 opacity micromap arrays) for the triangles of a BLAS.
//!
//! wgpu has no such resource: a backend's device makes one
//! (`create_opacity_micromap`), and the next BLAS the device creates takes
//! the ones given by `set_next_blas_opacity_micromaps` for its geometries
//! (its build sizes, its creation, its builds), so wgpu-core needs no change.

/// A BLAS geometry's micromap: each triangle's index into `data`'s
/// micromaps, or a special index (the whole triangle one state).
pub struct OpacityMicromapDescriptor<'a> {
    /// Every micromap's level: 4^level micro-triangles.
    pub subdivision_level: u32,
    /// How many micromaps `data` holds, 4-state (2 bits a micro-triangle,
    /// in the bird-curve order), `bytes_per_micromap` each.
    pub micromap_count: u32,
    pub data: &'a [u8],
    /// One a triangle of the geometry, in its index order: a micromap, or
    /// FULLY_TRANSPARENT (-1), FULLY_OPAQUE (-2), FULLY_UNKNOWN_TRANSPARENT
    /// (-3), FULLY_UNKNOWN_OPAQUE (-4).
    pub indices: &'a [i32],
}

pub const FULLY_TRANSPARENT: i32 = -1;
pub const FULLY_OPAQUE: i32 = -2;
pub const FULLY_UNKNOWN_TRANSPARENT: i32 = -3;
pub const FULLY_UNKNOWN_OPAQUE: i32 = -4;

/// The bytes of a 4-state micromap of `level` (at least one).
pub fn bytes_per_micromap(level: u32) -> usize {
    ((1usize << (2 * level)) * 2).div_ceil(8).max(1)
}
