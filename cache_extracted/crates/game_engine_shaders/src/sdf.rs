use game_engine_shaders_types::packet::{GpuCirclePacket, GpuStrokePacket, GpuTrianglePacket};
use game_engine_shaders_types::shapes::{Circle, Polygon, Rect, Stroked};
use spirv_std::glam::{Vec2, Vec4};
use spirv_std::num_traits::Float;

// ============================================================
// SdfShape — the original trait, used for distance + bounds
// ============================================================

pub trait SdfShape {
    fn bounds(&self) -> (Vec2, Vec2); // (Center, Size)
    fn dist(&self, uv: Vec2) -> f32;
    fn is_inside(&self, uv: Vec2) -> bool {
        self.dist(uv) < 0.0
    }
}

// --- SdfShape impls for primitive shapes ---

impl<const N: usize> SdfShape for Polygon<N> {
    fn bounds(&self) -> (Vec2, Vec2) {
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        let mut i = 0;
        while i < N {
            min = min.min(self.vertices[i]);
            max = max.max(self.vertices[i]);
            i += 1;
        }

        let size = max - min;
        let center = min + (size * 0.5);
        (center, size)
    }

    fn dist(&self, p: Vec2) -> f32 {
        let mut d = f32::MAX;
        let mut sign = 1.0;

        let mut i = 0;
        let mut j = N - 1;

        while i < N {
            let v_curr = self.vertices[i];
            let v_prev = self.vertices[j];

            let e = v_prev - v_curr;
            let w = p - v_curr;

            let b = w - e * (w.dot(e) / e.length_squared()).clamp(0.0, 1.0);
            d = d.min(b.length_squared());

            let cond_y = (p.y >= v_curr.y) != (p.y >= v_prev.y);
            let cond_x = (e.x * w.y) > (e.y * w.x);

            if cond_y && cond_x {
                sign = -sign;
            }

            j = i;
            i += 1;
        }

        sign * d.sqrt()
    }
}

impl SdfShape for Circle {
    fn bounds(&self) -> (Vec2, Vec2) {
        (self.center, Vec2::splat(self.radius * 2.0))
    }

    fn dist(&self, uv: Vec2) -> f32 {
        let local_p = uv - self.center;
        local_p.length() - self.radius
    }
}

impl SdfShape for Rect {
    fn bounds(&self) -> (Vec2, Vec2) {
        (self.center, self.size)
    }

    fn dist(&self, uv: Vec2) -> f32 {
        let local_p = uv - self.center;
        let d = local_p.abs() - (self.size * 0.5);
        d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
    }
}

impl SdfShape for Stroked<Polygon<3>> {
    fn bounds(&self) -> (Vec2, Vec2) {
        let (c, s) = self.inner.bounds();
        (c, s + Vec2::splat(self.thickness * 2.0))
    }
    fn dist(&self, uv: Vec2) -> f32 {
        self.inner.dist(uv).abs() - self.thickness
    }
}

impl SdfShape for Stroked<Circle> {
    fn bounds(&self) -> (Vec2, Vec2) {
        let (c, s) = self.inner.bounds();
        (c, s + Vec2::splat(self.thickness * 2.0))
    }
    fn dist(&self, uv: Vec2) -> f32 {
        self.inner.dist(uv).abs() - self.thickness
    }
}

// Packets delegate to their inner Stroked<Shape>
impl SdfShape for GpuTrianglePacket {
    fn bounds(&self) -> (Vec2, Vec2) {
        self.polygon.bounds()
    }
    fn dist(&self, uv: Vec2) -> f32 {
        self.polygon.dist(uv)
    }
}

impl SdfShape for GpuCirclePacket {
    fn bounds(&self) -> (Vec2, Vec2) {
        self.circle.bounds()
    }
    fn dist(&self, uv: Vec2) -> f32 {
        self.circle.dist(uv)
    }
}

// ============================================================
// Stack-based rendering traits
// ============================================================

#[derive(Copy, Clone)]
pub struct Bounds {
    pub center: Vec2,
    pub size: Vec2,
}

impl Bounds {
    pub fn empty() -> Self {
        Self {
            center: Vec2::ZERO,
            size: Vec2::ZERO,
        }
    }
}

#[derive(Copy, Clone)]
pub struct SdfResult {
    pub dist: f32,
    pub color: Vec4,
}

impl SdfResult {
    pub fn empty() -> Self {
        Self {
            dist: 1e9,
            color: Vec4::ZERO,
        }
    }
}

/// Shapes compute their AABB, modifiers expand it.
pub trait ComputeBounds {
    fn compute_bounds(&self, current: Bounds) -> Bounds;
}

/// Shapes set distance, modifiers transform distance/color.
pub trait ComputeSdf {
    fn apply(&self, uv: Vec2, current: SdfResult) -> SdfResult;
}

// ============================================================
// ComputeBounds — delegates to SdfShape::bounds()
// ============================================================

impl ComputeBounds for GpuTrianglePacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        let (center, size) = self.polygon.inner.bounds();
        Bounds { center, size }
    }
}

impl ComputeBounds for GpuCirclePacket {
    fn compute_bounds(&self, _current: Bounds) -> Bounds {
        let (center, size) = self.circle.inner.bounds();
        Bounds { center, size }
    }
}

impl ComputeBounds for GpuStrokePacket {
    fn compute_bounds(&self, current: Bounds) -> Bounds {
        Bounds {
            center: current.center,
            size: current.size + Vec2::splat(self.thickness * 2.0),
        }
    }
}

// ============================================================
// ComputeSdf — delegates to SdfShape::dist(), hard edges
// ============================================================

impl ComputeSdf for GpuTrianglePacket {
    fn apply(&self, uv: Vec2, _current: SdfResult) -> SdfResult {
        let d = self.polygon.inner.dist(uv);
        let color = if d > 0.0 {
            self.polygon.inner.color.get()
        } else {
            Vec4::ZERO
        };
        SdfResult { dist: d, color }
    }
}

impl ComputeSdf for GpuCirclePacket {
    fn apply(&self, uv: Vec2, _current: SdfResult) -> SdfResult {
        let d = self.circle.inner.dist(uv);
        let color = if d <= 0.0 {
            self.circle.inner.color.get()
        } else {
            Vec4::ZERO
        };
        SdfResult { dist: d, color }
    }
}

impl ComputeSdf for GpuStrokePacket {
    fn apply(&self, _uv: Vec2, current: SdfResult) -> SdfResult {
        let half_t = self.thickness * 0.5;
        let stroke_dist = current.dist.abs() - half_t;

        // Hard edge: inside stroke band → stroke color, otherwise keep current
        let color = if stroke_dist <= 0.0 {
            self.color.get()
        } else {
            current.color
        };

        SdfResult {
            dist: current.dist,
            color,
        }
    }
}
