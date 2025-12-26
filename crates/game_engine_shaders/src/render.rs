use game_engine_shaders_types::{
    PushConstants, Vec2, Vec4,
    packet::{GpuCirclePacket, GpuPacket, GpuTrianglePacket},
    shapes::{Circle, Polygon, Stroked},
};

use crate::sdf::SdfShape;

pub trait Renderable {
    fn pixel_color(&self, uv: Vec2, constants: &PushConstants) -> Vec4;
}

impl<S: SdfShape + GpuPacket + Renderable> Renderable for Stroked<S> {
    fn pixel_color(&self, uv: Vec2, constants: &PushConstants) -> Vec4 {
        let dist = self.inner.dist(uv);
        let half_thickness = self.thickness * 0.5;

        // 1. If we are within the "thickness" range of the edge, draw the stroke
        if dist.abs() < half_thickness {
            self.color.get()
        } else {
            // defer to inner
            self.inner.pixel_color(uv, constants)
        }
    }
}

impl<const N: usize> Renderable for Polygon<N> {
    fn pixel_color(&self, uv: Vec2, _constants: &PushConstants) -> Vec4 {
        if self.is_inside(uv) {
            self.color.get()
        } else {
            Vec4::ZERO // Transparent
        }
    }
}

impl Renderable for Circle {
    fn pixel_color(&self, uv: Vec2, _constants: &PushConstants) -> Vec4 {
        if self.is_inside(uv) {
            self.color.get()
        } else {
            Vec4::ZERO // Transparent
        }
    }
}

impl Renderable for GpuCirclePacket {
    fn pixel_color(&self, uv: Vec2, constants: &PushConstants) -> Vec4 {
        self.circle.pixel_color(uv, constants)
    }
}

impl Renderable for GpuTrianglePacket {
    fn pixel_color(&self, uv: Vec2, constants: &PushConstants) -> Vec4 {
        self.polygon.pixel_color(uv, constants)
    }
}
