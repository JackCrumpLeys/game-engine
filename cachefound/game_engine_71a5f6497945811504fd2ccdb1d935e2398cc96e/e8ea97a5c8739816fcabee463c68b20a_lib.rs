é#![allow(unexpected_cfgs)]
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
 ”*cascade08
”¹ ¹Ù*cascade08
Ùè è˜*cascade08
˜¿ ¿À*cascade08
Àï ïò*cascade08
òø øù*cascade08
ùú úû*cascade08
û‡ ‡ˆ*cascade08
ˆŠ Š‹*cascade08
‹› ›œ*cascade08
œ¨ ¨©*cascade08
©Á ÁÄ*cascade08
ÄÔ ÔÕ*cascade08
ÕÖ ÖØ*cascade08
Øá áâ*cascade08
âö ö÷*cascade08
÷ü üı*cascade08
ıÿ ÿ€*cascade08
€„ „…*cascade08
…Š Š‹*cascade08
‹Œ Œ*cascade08
§ §¨*cascade08
¨« «®*cascade08
®¯ ¯²*cascade08
²³ ³´*cascade08
´µ µ¶*cascade08
¶¸ ¸¹*cascade08
¹º º»*cascade08
»¾ ¾À*cascade08
ÀÁ ÁÄ*cascade08
ÄÅ ÅÇ*cascade08
ÇÊ ÊË*cascade08
ËÌ ÌÎ*cascade08
ÎÏ ÏÑ*cascade08
ÑÒ Òİ*cascade08
İŞ Şá*cascade08
áâ âè*cascade08
èé "(71a5f6497945811504fd2ccdb1d935e2398cc96e2Xfile:///home/jackc/Documents/code/rust/game_engine/crates/game_engine_shaders/src/lib.rs:2file:///home/jackc/Documents/code/rust/game_engine