use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::Vec2;
use crate::packet::{Interpolate, GpuBindingProvider};
use crate::shapes::ColorRGBA;
use crate::sdf::{Bounds, SdfResult, SdfShape};

#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuStrokePacket {
    pub color: ColorRGBA,
    pub thickness: f32,
    pub alignment: f32, // -1.0 = Inner, 0.0 = Center, 1.0 = Outer
    pub _pad: [f32; 2],
}

impl GpuStrokePacket {
    pub fn new(color: ColorRGBA, thickness: f32) -> Self {
        Self { 
            color, 
            thickness, 
            alignment: 0.0, // Center by default
            _pad: [0.0; 2] 
        }
    }
    
    pub fn outer(mut self) -> Self {
        self.alignment = 1.0;
        self
    }
    
    pub fn inner(mut self) -> Self {
        self.alignment = -1.0;
        self
    }
}

impl GpuBindingProvider for GpuStrokePacket {
    const OLD_BINDING: u32 = 4;
    const NEW_BINDING: u32 = 5;
}

impl SdfShape for GpuStrokePacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds {
        // Expand bounds based on alignment
        let half_t = self.thickness * 0.5;
        let offset = self.alignment * half_t;
        let expansion = half_t + offset; // Outer needs full thickness expansion, inner needs 0

        Bounds {
            center: current.center,
            size: current.size + Vec2::splat(expansion.max(0.0) * 2.0),
        }
    }

    fn apply(&self, _uv: Vec2, current: SdfResult) -> SdfResult {
        let half_t = self.thickness * 0.5;
        
        // Offset the distance field based on alignment
        let offset = self.alignment * half_t;
        let stroke_center = current.dist - offset;
        
        let stroke_dist = stroke_center.abs() - half_t;
        
        let color = if stroke_dist <= 0.0 { self.color.get() } else { current.color };
        SdfResult { dist: current.dist, color }
    }
}
