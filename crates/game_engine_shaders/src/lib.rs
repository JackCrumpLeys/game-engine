#![allow(unexpected_cfgs)]
#![allow(unused_imports)]
#![no_std]

use core::f32::consts::PI;
use game_engine_shaders_types::PushConstants;
use spirv_std::glam::{Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

// The "branchless" HSL to RGB conversion from GLSL
// c.x = Hue, c.y = Saturation, c.z = Lightness
fn hsl2rgb(c: Vec3) -> Vec3 {
    let k = Vec3::new(0.0, 4.0, 2.0);
    // (c.x * 6.0 + k) % 6.0
    let t = (Vec3::splat(c.x) * 6.0 + k) % 6.0;

    // clamp(abs(t - 3.0) - 1.0, 0.0, 1.0)
    let rgb = ((t - 3.0).abs() - 1.0).clamp(Vec3::ZERO, Vec3::ONE);

    // c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0))
    c.z + c.y * (rgb - 0.5) * (1.0 - (2.0 * c.z - 1.0).abs())
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(push_constant)] constants: &PushConstants,
    output: &mut Vec4,
) {
    // 1. Normalize Coordinates (Fix Aspect Ratio)
    // frag_coord.xy is in pixels (e.g., 400.5, 300.5)
    // We divide by the MINIMUM dimension to ensure circles stay circular on wide screens
    let dim = constants.resolution.x.min(constants.resolution.y);
    let uv = (frag_coord.truncate().truncate() - (constants.resolution * 0.5)) / dim;

    // 2. Polar Coordinates
    let dist = uv.length();
    let angle = uv.y.atan2(uv.x); // -PI to PI

    // 3. The Spiral Math
    // Angle normalized to 0..1
    let norm_angle = angle / (2.0 * PI);

    // Twist factor (dist * 5.0) + Rotation (-time * 0.5)
    let hue_raw = norm_angle + (dist * 2.0) - (constants.time * 0.5);

    // 4. Wrap to 0.0..1.0
    // IMPORTANT: Rust's `%` operator can return negatives.
    // .fract() behaves like GLSL fract(), handling negatives correctly for looping.
    let hue = hue_raw.fract();

    // 5. Color Conversion
    let color = hsl2rgb(Vec3::new(hue, 1.0, 0.5));

    *output = color.extend(1.0);
}

#[spirv(vertex)]
pub fn main_vs(
    position: Vec2,
    #[spirv(push_constant)] constants: &PushConstants,
    #[spirv(position)] out_pos: &mut Vec4,
) {
    // Move the triangle based on the offset, but don't rotate/scale it yet
    let pos = position + constants.position_offset;
    *out_pos = pos.extend(0.0).extend(1.0);
}
