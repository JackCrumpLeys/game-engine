use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::Vec2;
use spirv_std::image::Image2d;
use spirv_std::Sampler;
use spirv_std::RuntimeArray;
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuSpritePacket {
    pub center: Vec2,
    pub size: Vec2,
    pub texture_id: u32,
    pub _pad: [u32; 3],
}

impl GpuSpritePacket {
    pub fn new(center: Vec2, size: Vec2, texture_id: u32) -> Self {
        Self { center, size, texture_id, _pad: [0; 3] }
    }
}

impl GpuBindingProvider for GpuSpritePacket {
    const OLD_BINDING: u32 = 18;
    const NEW_BINDING: u32 = 19;
}

impl SdfShape for GpuSpritePacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds {
        if current.size.x == 0.0 && current.size.y == 0.0 {
            return Bounds { center: self.center, size: self.size };
        }
        let min_current = current.center - current.size * 0.5;
        let max_current = current.center + current.size * 0.5;
        let half_size = self.size * 0.5;
        let min_new = self.center - half_size;
        let max_new = self.center + half_size;
        let min_final = min_current.min(min_new);
        let max_final = max_current.max(max_new);
        Bounds {
            center: (min_final + max_final) * 0.5,
            size: max_final - min_final,
        }
    }

    fn apply(&self, _uv: Vec2, current: SdfResult) -> SdfResult { current }

    fn apply_with_context(
        &self,
        uv: Vec2,
        current: SdfResult,
        textures: &RuntimeArray<Image2d>,
        sampler: &Sampler,
        _path_curves: &[crate::packet::GpuCurve],
    ) -> SdfResult {
        let half_size = self.size * 0.5;
        let p = uv - self.center;
        let d = p.abs() - half_size;
        let dist = d.max(Vec2::ZERO).length() + d.max_element().min(0.0);
        let final_dist = current.dist.min(dist);

        if dist <= 0.0 {
            let tex_uv = (p + half_size) / self.size;
            let tex_uv = spirv_std::glam::vec2(tex_uv.x, 1.0 - tex_uv.y);
            unsafe {
                let img = textures.index(self.texture_id as usize);
                let color = img.sample(*sampler, tex_uv);
                let inv_alpha = 1.0 - color.w;
                let final_color = color + current.color * inv_alpha;
                SdfResult { dist: final_dist, color: final_color }
            }
        } else {
            SdfResult { dist: final_dist, color: current.color }
        }
    }
}
