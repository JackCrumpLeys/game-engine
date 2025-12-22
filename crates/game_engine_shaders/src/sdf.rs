use game_engine_shaders_types::packet::{GpuCirclePacket, GpuPacket, GpuTrianglePacket};
use game_engine_shaders_types::shapes::{Circle, Polygon, Rect, Stroked};
use spirv_std::glam::{Mat4, Vec2, Vec4};
use spirv_std::num_traits::Float;

// The core trait. Both Primitive shapes and Stroked wrappers implement this.
pub trait SdfShape {
    // Used in Vertex Shader to calculate quad size
    fn bounds(&self) -> (Vec2, Vec2); // (Center, Size)

    // Used in Fragment Shader to render
    fn dist(&self, uv: Vec2) -> f32;

    // Quicker inside/outside test
    fn is_inside(&self, uv: Vec2) -> bool {
        self.dist(uv) < 0.0
    }
}

impl SdfShape for GpuTrianglePacket {
    fn bounds(&self) -> (Vec2, Vec2) {
        self.polygon.bounds()
    }

    fn dist(&self, uv: Vec2) -> f32 {
        self.polygon.dist(uv)
    }

    fn is_inside(&self, uv: Vec2) -> bool {
        self.polygon.is_inside(uv)
    }
}

impl SdfShape for Circle {
    fn bounds(&self) -> (Vec2, Vec2) {
        (self.center, Vec2::splat(self.radius * 2.0))
    }

    fn dist(&self, uv: Vec2) -> f32 {
        // Localize the world coordinate by subtracting the circle's center
        let local_p = uv - self.center;
        local_p.length() - self.radius
    }
}

impl SdfShape for Rect {
    fn bounds(&self) -> (Vec2, Vec2) {
        (self.center, self.size)
    }

    fn dist(&self, uv: Vec2) -> f32 {
        // Localize the world coordinate
        let local_p = uv - self.center;
        let d = local_p.abs() - (self.size * 0.5);
        d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
    }

    fn is_inside(&self, uv: Vec2) -> bool {
        let local_p = uv - self.center;
        let d = local_p.abs() - (self.size * 0.5);
        d.x <= 0.0 && d.y <= 0.0
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

impl<T: SdfShape + GpuPacket> SdfShape for Stroked<T> {
    fn bounds(&self) -> (Vec2, Vec2) {
        let (c, s) = self.inner.bounds();
        (c, s + self.thickness)
    }
    fn dist(&self, uv: Vec2) -> f32 {
        self.inner.dist(uv).abs() - (self.thickness)
    }
}

impl<const N: usize> SdfShape for Polygon<N> {
    fn bounds(&self) -> (Vec2, Vec2) {
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        for i in 0..N {
            min = min.min(self.vertices[i]);
            max = max.max(self.vertices[i]);
        }

        let size = max - min;
        let center = min + (size * 0.5);
        // Add padding for AA and strokes
        (center, size)
    }

    fn dist(&self, p: Vec2) -> f32 {
        let mut d = f32::MAX;
        let mut sign = 1.0;

        let mut i = 0;
        let mut j = N - 1; // Previous vertex index

        while i < N {
            let v_curr = self.vertices[i];
            let v_prev = self.vertices[j];

            let e = v_prev - v_curr;
            let w = p - v_curr;

            // 1. Distance to Line Segment
            // Project p onto edge e, clamp between 0 and 1
            let b = w - e * (w.dot(e) / e.length_squared()).clamp(0.0, 1.0);
            d = d.min(b.length_squared());

            // 2. Winding Number (Sign) Logic
            // This is a check to see if the ray from P crosses the edge
            let cond_y = (p.y >= v_curr.y) != (p.y >= v_prev.y);
            let cond_x = (e.x * w.y) > (e.y * w.x); // Cross product check

            if cond_y && cond_x {
                sign = -sign;
            }

            j = i;
            i += 1;
        }

        sign * d.sqrt()
    }

    fn is_inside(&self, p: Vec2) -> bool {
        let mut inside = false;
        let mut j = N - 1;

        for i in 0..N {
            let vi = self.vertices[i];
            let vj = self.vertices[j];

            // 1. Check if the point's Y is between the edge's Y-range
            // 2. Check if the point is to the left of the edge (the ray-cast)
            if ((vi.y > p.y) != (vj.y > p.y))
                && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside; // Flip the bool
            }
            j = i;
        }

        inside
    }
}
