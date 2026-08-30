use spirv_std::glam::{Vec2, Vec4};

#[derive(Copy, Clone)]
pub struct Bounds {
    pub center: Vec2,
    pub size: Vec2,
}

impl Bounds {
    pub fn empty() -> Self {
        Self { center: Vec2::ZERO, size: Vec2::ZERO }
    }
}

#[derive(Copy, Clone)]
pub struct SdfResult {
    pub dist: f32,
    pub color: Vec4,
}

impl SdfResult {
    pub fn empty() -> Self {
        Self { dist: 1e9, color: Vec4::ZERO }
    }
}

pub trait SdfShape {
    fn compute_bounds(&self, current: Bounds) -> Bounds;

    fn apply(&self, _uv: Vec2, current: SdfResult) -> SdfResult {
        current // Base ignore implementation
    }

    #[allow(unused_variables)]
    fn apply_with_context(
        &self,
        uv: Vec2,
        current: SdfResult,
        textures: &spirv_std::RuntimeArray<spirv_std::image::Image!(2D, type=f32, sampled)>,
        sampler: &spirv_std::Sampler,
        path_curves: &[crate::packet::GpuCurve],
    ) -> SdfResult {
        self.apply(uv, current) // Defaults to pure-math apply
    }
}

