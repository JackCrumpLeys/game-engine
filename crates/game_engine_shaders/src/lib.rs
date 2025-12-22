#![allow(unexpected_cfgs)]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![no_std]

use game_engine_shaders_types::packet::{CIRCLE_SHAPE_INDEX, GpuCirclePacket, InstancePointer};
use game_engine_shaders_types::{
    PushConstants, Vec2, Vec4,
    packet::{GpuTrianglePacket, Interpolate, TRIANGLE_SHAPE_INDEX},
    vec2,
};
use paste::paste;
use spirv_std::spirv;

use crate::{render::Renderable, sdf::SdfShape};

mod render;
mod sdf;
pub mod triangle;
#[macro_use]
mod macros;

// Helper to fetch and interpolate from the separate buffers
fn fetch<T: Copy + Interpolate>(idx: u32, old: &[T], new: &[T], factor: f32) -> T {
    let a = old[idx as usize];
    let b = new[idx as usize];
    a.interpolate(&b, factor)
}

// # Uber-Shader Renderer Entry Point
//
// This macro generates the `main_vs` (Vertex) and `main_fs` (Fragment) entry points
// for a generic 2D SDF renderer. It utilizes a "Bindless-style" architecture via
// descriptor indexing to support multiple shape types in a single draw call.
//
// ## Input Architecture
//
// The renderer expects data to be organized into two Descriptor Sets:
//
// **Descriptor Set 1: The Master Index**
// *   **Binding 0 (`instance_map`):** A storage buffer of `InstancePointer` structs.
//     *   Maps `gl_InstanceIndex` -> `(ShapeTypeID, LocalIndex)`.
//     *   This allows the CPU to sort draw order (Z-index) without moving heavy shape data.
//
// **Descriptor Set 0: The Data Banks (Ping-Pong Buffers)**
// *   Contains paired storage buffers (`old`, `new`) for *each* shape type defined in the macro.
// *   **Bindings:** Defined explicitly in the macro call (e.g., Triangle Old: 0, Triangle New: 1).
// *   **Interpolation:** The shader automatically interpolates between `old` and `new` based on
//     `PushConstants::factor` to smooth out physics/logic ticks.
//
// ## Vertex Shader Logic
// 1.  Reads `InstancePointer` using `gl_InstanceIndex`.
// 2.  Fetches the specific shape packet (Triangle, Circle, etc.) using `LocalIndex`.
// 3.  Calculates the AABB (Axis-Aligned Bounding Box) of the shape via `SdfShape::bounds()`.
// 4.  Generates a Quad (2 triangles) covering that AABB on the fly (no vertex buffer required).
//
// ## Fragment Shader Logic
// 1.  Receives the `InstancePointer` and interpolated `uv` coordinates from VS.
// 2.  Refetches the shape packet.
// 3.  Delegates to `Renderable::pixel_color()` to compute the final RGBA value.
//
// ## Push Constants
// Expects a `PushConstants` struct.
define_renderer!(
    // Shape ID Constant => Name     : Type              [OldBinding, NewBinding]
    TRIANGLE_SHAPE_INDEX => triangle : GpuTrianglePacket [0, 1],
    CIRCLE_SHAPE_INDEX   => circle   : GpuCirclePacket   [2, 3],

);
// BINDING CONSTANTS MUST MATCH THE VALUES USED IN THE MACRO CALL ABOVE!
// I COuld odo this in build.rs but ehhh
