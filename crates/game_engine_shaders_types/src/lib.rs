#![cfg_attr(not(test), no_std)]

pub mod packet;
pub mod shapes;

use bytemuck::{Pod, Zeroable};
pub use spirv_std::glam::*;

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PushConstants {
    pub view_proj: Mat4,
    pub position_offset: Vec2,
    pub time: f32,
    pub factor: f32,
}

/// Creates a 2D orthographic projection matrix for Vulkan.
///
/// This matrix transforms a 2D coordinate system where:
/// - (0, 0) is the top-left corner of the screen.
/// - (width, height) is the bottom-right corner.
///
/// ...into Vulkan's Normalized Device Coordinates (NDC) clip space where:
/// - (-1, -1) is the top-left corner.
/// - (1, 1) is the bottom-right corner.
/// - Z is mapped from -1.0 (near plane) to 1.0 (far plane).
///
/// # Arguments
/// * `width` - The width of the viewport in pixels.
/// * `height` - The height of the viewport in pixels.
pub fn create_projection_matrix(width: f32, height: f32) -> Mat4 {
    // Note on `orthographic_rh`: `rh` stands for "Right-Handed".
    // Vulkan uses a right-handed coordinate system for its clip space.
    //
    // The arguments are (left, right, bottom, top, near, far).
    //
    // A key difference from OpenGL is that in Vulkan's screen space, the Y-axis
    // points DOWNWARDS. To compensate for this and create an intuitive top-left
    // origin, we swap the `bottom` and `top` arguments.
    // So `bottom` is `height` and `top` is `0.0`.
    Mat4::orthographic_rh(
        0.0,    // left
        width,  // right
        height, // bottom
        0.0,    // top
        -1.0,   // near clipping plane
        1.0,    // far clipping plane
    )
}

pub fn create_camera_matrix(width: f32, height: f32, pos: Vec2, zoom: f32) -> Mat4 {
    // 1. Basic Ortho (Screen space: 0,0 top left)
    let projection = Mat4::orthographic_rh(0.0, width, height, 0.0, -1.0, 1.0);

    // 2. View Matrix
    // We want to zoom into the center of the screen, so we:
    // Translate screen center to origin -> Scale -> Translate back -> Apply Pan
    let center = vec3(width / 2.0, height / 2.0, 0.0);

    let view = Mat4::from_translation(center)
        * Mat4::from_scale(vec3(zoom, zoom, 1.0))
        * Mat4::from_translation(-center)
        * Mat4::from_translation(vec3(-pos.x, -pos.y, 0.0));
    projection * view
}
