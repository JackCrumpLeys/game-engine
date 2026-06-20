use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::Vec2;
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

// ── Linear Gradient ──────────────────────────────────────────────────

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuLinearGradientPacket {
    pub start: Vec2,
    pub end: Vec2,
    pub color_start: ColorRGBA,
    pub color_end: ColorRGBA,
}

impl GpuLinearGradientPacket {
    pub fn new(start: Vec2, end: Vec2, color_start: ColorRGBA, color_end: ColorRGBA) -> Self {
        Self { start, end, color_start, color_end }
    }
}

impl GpuBindingProvider for GpuLinearGradientPacket {
    const OLD_BINDING: u32 = 12;
    const NEW_BINDING: u32 = 13;
}

impl SdfShape for GpuLinearGradientPacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds { current }

    fn apply(&self, uv: Vec2, current: SdfResult) -> SdfResult {
        if current.dist > 0.0 { return current; }
        let pa = uv - self.start;
        let ba = self.end - self.start;
        let t = (pa.dot(ba) / ba.length_squared()).clamp(0.0, 1.0);
        let color = self.color_start.get().lerp(self.color_end.get(), t);
        SdfResult { dist: current.dist, color }
    }
}

// ── Radial Gradient ──────────────────────────────────────────────────

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuRadialGradientPacket {
    pub center: Vec2,
    pub radius: f32,
    pub _padding: f32,
    pub color_in: ColorRGBA,
    pub color_out: ColorRGBA,
}

impl GpuRadialGradientPacket {
    pub fn new(center: Vec2, radius: f32, color_in: ColorRGBA, color_out: ColorRGBA) -> Self {
        Self { center, radius, _padding: 0.0, color_in, color_out }
    }
}

impl GpuBindingProvider for GpuRadialGradientPacket {
    const OLD_BINDING: u32 = 14;
    const NEW_BINDING: u32 = 15;
}

impl SdfShape for GpuRadialGradientPacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds { current }

    fn apply(&self, uv: Vec2, current: SdfResult) -> SdfResult {
        if current.dist > 0.0 { return current; }
        let dist = (uv - self.center).length();
        let t = (dist / self.radius).clamp(0.0, 1.0);
        let color = self.color_in.get().lerp(self.color_out.get(), t);
        SdfResult { dist: current.dist, color }
    }
}
