use crate::packet::{GpuBindingProvider, Interpolate};
use crate::sdf::{Bounds, SdfResult, SdfShape};
use crate::shapes::ColorRGBA;
use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug, Default)]
#[repr(C)]
pub struct GpuTrianglePacket {
    pub vertices: [Vec2; 3],
    pub _padding: [f32; 2],
    pub color: ColorRGBA,
}

impl GpuTrianglePacket {
    pub fn new(v1: Vec2, v2: Vec2, v3: Vec2) -> Self {
        Self {
            vertices: [v1, v2, v3],
            _padding: [0.0; 2],
            color: ColorRGBA::default(),
        }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self {
        self.color = color;
        self
    }
}

impl GpuBindingProvider for GpuTrianglePacket {
    const OLD_BINDING: u32 = 0;
    const NEW_BINDING: u32 = 1;
}

impl SdfShape for GpuTrianglePacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];
        let mut i = 0;
        while i < 3 {
            min = min.min(self.vertices[i]);
            max = max.max(self.vertices[i]);
            i += 1;
        }
        let size = max - min;
        let center = min + (size * 0.5);
        Bounds { center, size }
    }

    fn apply(&self, p: Vec2, _current: SdfResult) -> SdfResult {
        let mut d = f32::MAX;
        let mut sign = 1.0;
        let mut i = 0;
        let mut j = 2;
        while i < 3 {
            let v_curr = self.vertices[i];
            let v_prev = self.vertices[j];
            let e = v_prev - v_curr;
            let w = p - v_curr;
            let b = w - e * (w.dot(e) / e.length_squared()).clamp(0.0, 1.0);
            d = d.min(b.length_squared());
            let cond_y = (p.y >= v_curr.y) != (p.y >= v_prev.y);
            let cond_x = (e.x * w.y) > (e.y * w.x);
            if cond_y && (cond_x == (p.y >= v_curr.y)) {
                sign = -sign;
            }
            j = i;
            i += 1;
        }
        let dist = sign * d.sqrt();
        let color = if dist <= 0.0 {
            self.color.get()
        } else {
            Vec4::ZERO
        };
        SdfResult { dist, color }
    }
}
