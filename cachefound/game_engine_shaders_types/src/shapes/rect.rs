use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug, Default)]
#[repr(C)]
pub struct GpuRectPacket {
    pub center: Vec2,
    pub size: Vec2,
    pub color: ColorRGBA,
}

impl GpuRectPacket {
    pub fn new(center: Vec2, size: Vec2) -> Self {
        Self { center, size, color: ColorRGBA::default() }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self { self.color = color; self }
}

impl GpuBindingProvider for GpuRectPacket {
    const OLD_BINDING: u32 = 10;
    const NEW_BINDING: u32 = 11;
}

impl SdfShape for GpuRectPacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        Bounds { center: self.center, size: self.size }
    }

    fn apply(&self, uv: Vec2, _current: SdfResult) -> SdfResult {
        let local_p = uv - self.center;
        let d = local_p.abs() - (self.size * 0.5);
        let dist = d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0);
        let color = if dist <= 0.0 { self.color.get() } else { Vec4::ZERO };
        SdfResult { dist, color }
    }
}
