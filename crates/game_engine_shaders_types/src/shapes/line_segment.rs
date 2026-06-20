use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug, Default)]
#[repr(C)]
pub struct GpuLineSegmentPacket {
    pub start: Vec2,
    pub end: Vec2,
    pub thickness: f32,
    pub _padding1: [f32; 3],
    pub color: ColorRGBA,
}

impl GpuLineSegmentPacket {
    pub fn new(start: Vec2, end: Vec2, thickness: f32) -> Self {
        Self { start, end, thickness, _padding1: [0.0; 3], color: ColorRGBA::default() }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self { self.color = color; self }
}

impl GpuBindingProvider for GpuLineSegmentPacket {
    const OLD_BINDING: u32 = 8;
    const NEW_BINDING: u32 = 9;
}

impl SdfShape for GpuLineSegmentPacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        let min_p = self.start.min(self.end) - Vec2::splat(self.thickness * 0.5);
        let max_p = self.start.max(self.end) + Vec2::splat(self.thickness * 0.5);
        let size = max_p - min_p;
        let center = min_p + size * 0.5;
        Bounds { center, size }
    }

    fn apply(&self, uv: Vec2, _current: SdfResult) -> SdfResult {
        let pa = uv - self.start;
        let ba = self.end - self.start;
        let h = (pa.dot(ba) / ba.length_squared()).clamp(0.0, 1.0);
        let d = (pa - ba * h).length();
        let dist = d - self.thickness * 0.5;
        let color = if dist <= 0.0 { self.color.get() } else { Vec4::ZERO };
        SdfResult { dist, color }
    }
}
