use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuClipRectPacket {
    pub center: Vec2,
    pub size: Vec2,
}

impl GpuClipRectPacket {
    pub fn new(center: Vec2, size: Vec2) -> Self {
        Self { center, size }
    }
}

impl GpuBindingProvider for GpuClipRectPacket {
    const OLD_BINDING: u32 = 16;
    const NEW_BINDING: u32 = 17;
}

impl SdfShape for GpuClipRectPacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds {
        let min_current = current.center - current.size * 0.5;
        let max_current = current.center + current.size * 0.5;
        let min_clip = self.center - self.size * 0.5;
        let max_clip = self.center + self.size * 0.5;
        let min_intersect = min_current.max(min_clip);
        let max_intersect = max_current.min(max_clip);
        let size = (max_intersect - min_intersect).max(Vec2::ZERO);
        let center = min_intersect + size * 0.5;
        Bounds { center, size }
    }

    fn apply(&self, uv: Vec2, current: SdfResult) -> SdfResult {
        let local_p = uv - self.center;
        let d = local_p.abs() - (self.size * 0.5);
        let clip_dist = d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0);
        let dist = current.dist.max(clip_dist);
        let color = if dist <= 0.0 { current.color } else { Vec4::ZERO };
        SdfResult { dist, color }
    }
}
