#![allow(unexpected_cfgs)]
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![no_std]

use game_engine_shaders_types::packet::{
    CIRCLE_SHAPE_INDEX, CLIP_RECT_SHAPE_INDEX, GpuCirclePacket, GpuClipRectPacket, GpuEntityHead,
    GpuLineSegmentPacket, GpuLinearGradientPacket, GpuGlyphPacket, GpuBezierPathPacket, GpuRadialGradientPacket,
    GpuRectPacket, GpuRoundedRectPacket, GpuSpritePacket, GpuStrokePacket, InstancePointer,
    LINEAR_GRADIENT_SHAPE_INDEX, LINESEGMENT_SHAPE_INDEX, GLYPH_SHAPE_INDEX, BEZIER_PATH_SHAPE_INDEX,
    RADIAL_GRADIENT_SHAPE_INDEX, RECT_SHAPE_INDEX, ROUNDED_RECT_SHAPE_INDEX,
    SPRITE_SHAPE_INDEX, STROKE_SHAPE_INDEX, TRIANGLE_SHAPE_INDEX,
};
use game_engine_shaders_types::{
    PushConstants, Vec2, Vec4,
    packet::{GpuTrianglePacket, Interpolate},
    vec2,
};
use paste::paste;
use spirv_std::spirv;

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
    // shape_id => name : type [old_idx, new_idx]
    TRIANGLE_SHAPE_INDEX => triangle : GpuTrianglePacket [0, 1],
    CIRCLE_SHAPE_INDEX   => circle   : GpuCirclePacket   [2, 3],
    // Render a colored stroke around shape stack
    STROKE_SHAPE_INDEX   => stroke   : GpuStrokePacket   [4, 5],
    // Render a rectangle with radius per corner
    ROUNDED_RECT_SHAPE_INDEX => rounded_rect : GpuRoundedRectPacket [6, 7],
    // Line from point A to B
    LINESEGMENT_SHAPE_INDEX => line_seg  : GpuLineSegmentPacket [8, 9],
    RECT_SHAPE_INDEX        => rect      : GpuRectPacket        [10, 11],
    // this is a special layer overidding the color inside of the stack shader with a gradient
    LINEAR_GRADIENT_SHAPE_INDEX => linear_grad : GpuLinearGradientPacket [12, 13],
    RADIAL_GRADIENT_SHAPE_INDEX => radial_grad : GpuRadialGradientPacket [14, 15],
    // Only render stack up until now within the clip rect
    CLIP_RECT_SHAPE_INDEX   => clip_rect : GpuClipRectPacket    [16, 17],
    // Sample color from a texture. Right now its a base shape as a equalateral square
    // TODO:  Ideally this could just pervide pixel color to the stack later
    SPRITE_SHAPE_INDEX      => sprite    : GpuSpritePacket      [18, 19],
    // Analytic glyph: evaluates distance fields on the fly from flattened line segments
    GLYPH_SHAPE_INDEX       => glyph      : GpuGlyphPacket        [20, 21],
    BEZIER_PATH_SHAPE_INDEX => bezier_path : GpuBezierPathPacket [22, 23],
);
