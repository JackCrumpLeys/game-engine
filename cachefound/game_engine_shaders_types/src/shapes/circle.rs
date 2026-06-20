use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug, Default)]
#[repr(C)]
pub struct GpuCirclePacket {
    pub color: ColorRGBA,
    pub center: Vec2,
    pub radius: f32,
    pub _padding: [f32; 1],
}

impl GpuCirclePacket {
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius, color: ColorRGBA::default(), _padding: [0.0] }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self { self.color = color; self }
}

impl GpuBindingProvider for GpuCirclePacket {
    const OLD_BINDING: u32 = 2;
    const NEW_BINDING: u32 = 3;
}

impl SdfShape for GpuCirclePacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        Bounds { center: self.center, size: Vec2::splat(self.radius * 2.0) }
    }

    fn apply(&self, uv: Vec2, _current: SdfResult) -> SdfResult {
        let local_p = uv - self.center;
        let dist = local_p.length() - self.radius;
        let color = if dist <= 0.0 { self.color.get() } else { Vec4::ZERO };
        SdfResult { dist, color }
    }
}
