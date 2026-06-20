use crate::packet::{GpuBindingProvider, Interpolate};
use crate::sdf::{Bounds, SdfResult, SdfShape};
use crate::shapes::ColorRGBA;
use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::RuntimeArray;
use spirv_std::Sampler;
use spirv_std::glam::Vec2;
use spirv_std::image::Image2d;
use spirv_std::num_traits::Float;
use crate::packet::GpuCurve;
use crate::shapes::bezier_path::{sd_bezier, bezier_winding};

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuGlyphPacket {
    pub color: ColorRGBA,
    pub offset: Vec2,
    pub bbox_min: Vec2,
    pub bbox_max: Vec2,
    pub scale: f32,
    pub line_start: u32,
    pub line_count: u32,
    pub _pad: [u32; 3], // Pad struct to exact 64 bytes for strict GPU alignment
}

impl GpuGlyphPacket {
    pub fn with_color(mut self, color: ColorRGBA) -> Self {
        self.color = color;
        self
    }
}

impl GpuBindingProvider for GpuGlyphPacket {
    const OLD_BINDING: u32 = 20;
    const NEW_BINDING: u32 = 21;
}

impl SdfShape for GpuGlyphPacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds {
        let min_p = self.offset + self.bbox_min * self.scale;
        let max_p = self.offset + self.bbox_max * self.scale;
        let center = (min_p + max_p) * 0.5;
        let size = max_p - min_p;

        if current.size.x == 0.0 && current.size.y == 0.0 {
            return Bounds { center, size };
        }

        let min_current = current.center - current.size * 0.5;
        let max_current = current.center + current.size * 0.5;
        let min_final = min_current.min(min_p);
        let max_final = max_current.max(max_p);

        Bounds {
            center: (min_final + max_final) * 0.5,
            size: max_final - min_final,
        }
    }

    fn apply(&self, _uv: Vec2, current: SdfResult) -> SdfResult {
        current
    }

    // Truly analytical, perfect evaluation on the GPU!
    fn apply_with_context(
        &self,
        uv: Vec2,
        current: SdfResult,
        _textures: &RuntimeArray<Image2d>,
        _sampler: &Sampler,
        path_curves: &[GpuCurve],
    ) -> SdfResult {
        let p = (uv - self.offset) / self.scale;
        let mut min_dist = 1e9f32;
        let mut winding = 0;

        // FIXED: Sub-pixel jitter prevents exact-vertex ray hits (solves horizontal glitching)
        let ray_p = spirv_std::glam::vec2(p.x, p.y + 0.01);

        let start = self.line_start as usize;
        let end = start + self.line_count as usize;

        let mut i = start;
        while i < end {
            let curve = path_curves[i];
            
            // min_dist still evaluates exactly from `p`
            min_dist = min_dist.min(sd_bezier(p, curve.p0, curve.p1, curve.p2));
            
            // winding evaluates from the jittered `ray_p`
            winding += bezier_winding(ray_p, curve.p0, curve.p1, curve.p2);
            i += 1;
        }

        let glyph_dist = if winding != 0 { -min_dist } else { min_dist };
        let world_dist = glyph_dist * self.scale; // Exact screen space scaling for perfect AA

        let final_dist = current.dist.min(world_dist);

        if world_dist <= 0.0 {
            let c = self.color.get();
            let final_color = c + current.color * (1.0 - c.w);
            SdfResult {
                dist: final_dist,
                color: final_color,
            }
        } else {
            SdfResult {
                dist: final_dist,
                color: current.color,
            }
        }
    }
}
