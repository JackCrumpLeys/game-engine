use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug, Default)]
#[repr(C)]
pub struct GpuRoundedRectPacket {
    pub center: Vec2,
    pub size: Vec2,
    pub color: ColorRGBA,
    pub radii: Vec4,
}

impl GpuRoundedRectPacket {
    pub fn new(center: Vec2, size: Vec2, radii: Vec4) -> Self {
        Self { center, size, radii, color: ColorRGBA::default() }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self { self.color = color; self }
}

impl GpuBindingProvider for GpuRoundedRectPacket {
    const OLD_BINDING: u32 = 6;
    const NEW_BINDING: u32 = 7;
}

impl SdfShape for GpuRoundedRectPacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        Bounds { center: self.center, size: self.size }
    }

    fn apply(&self, uv: Vec2, _current: SdfResult) -> SdfResult {
        let p = uv - self.center;
        let r = if p.x > 0.0 {
            if p.y > 0.0 { self.radii.x } else { self.radii.y }
        } else if p.y > 0.0 { self.radii.z } else { self.radii.w };
        let q = p.abs() - (self.size * 0.5) + Vec2::splat(r);
        let dist = q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - r;
        let color = if dist <= 0.0 { self.color.get() } else { Vec4::ZERO };
        SdfResult { dist, color }
    }
}
