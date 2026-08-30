
use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::Vec2;
use spirv_std::num_traits::Float;
use crate::packet::{Interpolate, GpuBindingProvider, GpuCurve};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuBezierPathPacket {
    pub color: ColorRGBA,
    pub offset: Vec2,
    pub bbox_min: Vec2,
    pub bbox_max: Vec2,
    pub thickness: f32,
    pub line_start: u32,
    pub line_count: u32,
    pub closed: u32, // 1 for filled polygon, 0 for stroked path
    pub _pad: [u32; 2],
}

impl GpuBindingProvider for GpuBezierPathPacket {
    const OLD_BINDING: u32 = 22; // Next available bindings
    const NEW_BINDING: u32 = 23;
}

// ---- GPU MATH FUNCTIONS ----
// Exact distance to a quadratic bezier curve
pub fn sd_bezier(pos: Vec2, p0: Vec2, p1: Vec2, p2: Vec2) -> f32 {
    let a = p1 - p0;
    let b = p0 - 2.0 * p1 + p2;
    let c = a * 2.0;
    let d = p0 - pos;

    let dot_b = b.dot(b);
    if dot_b < 1e-5 {
        // Degenerate to a straight line
        let pa = pos - p0;
        let ba = p2 - p0;
        let ba_len_sq = ba.length_squared();
        if ba_len_sq < 1e-5 {
            return pa.length();
        }
        let h = (pa.dot(ba) / ba_len_sq).clamp(0.0, 1.0);
        return (pa - ba * h).length();
    }

    let kk = 1.0 / dot_b;
    let kx = kk * a.dot(b);
    let ky = kk * (2.0 * a.dot(a) + d.dot(b)) / 3.0;
    let kz = kk * d.dot(a);

    let p = ky - kx * kx;
    let p3 = p * p * p;
    let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    let mut h = q * q + 4.0 * p3;

    if h >= 0.0 {
        h = h.sqrt();
        let x = (spirv_std::glam::vec2(h, -h) - q) / 2.0;
        let uv = x.signum() * x.abs().powf(1.0 / 3.0);
        let t = (uv.x + uv.y - kx).clamp(0.0, 1.0);
        return (d + (c + b * t) * t).length();
    }

    // 1. FIXED: Correctly derive z without causing negative square roots
    let z = (-p).sqrt();
    
    // 2. FIXED: Clamp the acos input to [-1.0, 1.0] to protect against float drift
    let v = (q / (p * z * 2.0)).clamp(-1.0, 1.0).acos() / 3.0;
    
    let m = v.cos();
    let n = v.sin() * 1.732_050_8; // sqrt(3)
    let t = (spirv_std::glam::vec2(m + m, -n - m) * z - kx).clamp(Vec2::ZERO, Vec2::ONE);
    let qx = d + (c + b * t.x) * t.x;
    let qy = d + (c + b * t.y) * t.y;
    qx.length_squared().min(qy.length_squared()).sqrt()
}

// Ray-casting winding number for point-in-polygon
pub fn bezier_winding(pos: Vec2, p0: Vec2, p1: Vec2, p2: Vec2) -> i32 {
    let a = p0.y - 2.0 * p1.y + p2.y;
    let b = 2.0 * (p1.y - p0.y);
    let c = p0.y - pos.y;

    let ax = p0.x - 2.0 * p1.x + p2.x;
    let bx = 2.0 * (p1.x - p0.x);

    let mut winding = 0;

    if a.abs() < 1e-5 {
        if b.abs() > 1e-5 {
            let t = -c / b;
            if (0.0..1.0).contains(&t) {
                let x = p0.x + t * bx + t * t * ax;
                if x > pos.x { 
                    winding += if b > 0.0 { 1 } else { -1 }; 
                }
            }
        }
    } else {
        let delta = b * b - 4.0 * a * c;
        if delta > 0.0 {
            let d = delta.sqrt();
            
            // FIXED: Numerically stable quadratic roots to prevent Catastrophic Cancellation
            let q = if b > 0.0 { -0.5 * (b + d) } else { -0.5 * (b - d) };
            let root1 = q / a;
            let root2 = c / q;
            
            // Maintain root order: t1 is always the downward crossing, t2 is upward
            let (t1, t2) = if b > 0.0 { (root1, root2) } else { (root2, root1) };

            if (0.0..1.0).contains(&t1) {
                let x = p0.x + t1 * bx + t1 * t1 * ax;
                if x > pos.x { winding -= 1; }
            }
            if (0.0..1.0).contains(&t2) {
                let x = p0.x + t2 * bx + t2 * t2 * ax;
                if x > pos.x { winding += 1; }
            }
        }
    }
    winding
}

impl SdfShape for GpuBezierPathPacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds {
        let min_p = self.offset + self.bbox_min;
        let max_p = self.offset + self.bbox_max;
        
        if current.size.x == 0.0 && current.size.y == 0.0 {
            return Bounds { center: (min_p + max_p) * 0.5, size: max_p - min_p };
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

    fn apply_with_context(
        &self,
        uv: Vec2,
        current: SdfResult,
        _textures: &spirv_std::RuntimeArray<spirv_std::image::Image!(2D, type=f32, sampled)>,
        _sampler: &spirv_std::Sampler,
        path_curves: &[GpuCurve],
    ) -> SdfResult {
        let p = uv - self.offset;
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
            
            if self.closed == 1 {
                // winding evaluates from the jittered `ray_p`
                winding += bezier_winding(ray_p, curve.p0, curve.p1, curve.p2);
            }
            i += 1;
        }

        let mut final_dist = min_dist;
        if self.closed == 1 {
            final_dist = if winding != 0 { -min_dist } else { min_dist };
        } else {
            final_dist -= self.thickness * 0.5;
        }

        let world_dist = current.dist.min(final_dist);
        let color = if final_dist <= 0.0 { self.color.get() } else { current.color };
        
        SdfResult { dist: world_dist, color }
    }
}
