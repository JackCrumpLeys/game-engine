#![allow(unexpected_cfgs)]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![no_std]

use game_engine_shaders_types::packet::{
    CIRCLE_SHAPE_INDEX, GpuCirclePacket, GpuEntityHead, GpuStrokePacket, InstancePointer,
    STROKE_SHAPE_INDEX, TRIANGLE_SHAPE_INDEX,
};
use game_engine_shaders_types::{
    PushConstants, Vec2, Vec4,
    packet::{GpuTrianglePacket, Interpolate},
    vec2,
};
use paste::paste;
use spirv_std::spirv;

pub mod sdf;
pub mod triangle;
#[macro_use]
mod macros;

// Helper to fetch and interpolate from the separate buffers
fn fetch<T: Copy + Interpolate>(idx: u32, old: &[T], new: &[T], factor: f32) -> T {
    let a = old[idx as usize];
    let b = new[idx as usize];
    a.interpolate(&b, factor)
}

// Stack-based uber-shader.
// Each shape type has its own ping-pong buffer pair.
// The GPU iterates an entity's operation stack to resolve bounds (VS) and color (FS).
define_renderer!(
    TRIANGLE_SHAPE_INDEX => triangle : GpuTrianglePacket [0, 1],
    CIRCLE_SHAPE_INDEX   => circle   : GpuCirclePacket   [2, 3],
    STROKE_SHAPE_INDEX   => stroke   : GpuStrokePacket   [4, 5],
);
