use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4};

use crate::shapes::{Circle, ColorRGBA, Polygon, Stroked};

pub const TRIANGLE_SHAPE_INDEX: u32 = 0;
pub const CIRCLE_SHAPE_INDEX: u32 = 1;
pub const STROKE_SHAPE_INDEX: u32 = 2;

pub trait Interpolate {
    /// interpolates between self and other by factor (0.0 - 1.0)
    fn interpolate(&self, other: &Self, factor: f32) -> Self;
}

pub trait GpuPacket: Interpolate + Clone + Pod + Zeroable {}
impl<T: Interpolate + Clone + Pod + Zeroable> GpuPacket for T {}

/// A single triangle primitive.
/// This is treated as one "instance" by the renderer.
#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug)]
#[repr(C)]
pub struct GpuTrianglePacket {
    pub polygon: Polygon<3>>,
}

impl GpuTrianglePacket {
    pub fn new(v1: Vec2, v2: Vec2, v3: Vec2) -> Self {
        Self {
            polygon: Stroked {
                inner: Polygon {
                    vertices: [v1, v2, v3],
                    color: ColorRGBA::default(),
                },
                thickness: 2.0,
                color: ColorRGBA::new(0., 0., 1., 1.),
                _pad: Default::default(),
            },
        }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self {
        self.polygon.inner.color = color;
        self
    }
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.polygon.thickness = thickness;
        self
    }
    pub fn with_stroke_color(mut self, color: ColorRGBA) -> Self {
        self.polygon.color = color;
        self
    }
}

/// a single circle primitive
#[derive(Copy, Clone, Pod, Zeroable, InterpolateM, Debug)]
#[repr(C)]
pub struct GpuCirclePacket {
    pub circle: Stroked<Circle>,
}

impl GpuCirclePacket {
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self {
            circle: Stroked {
                inner: Circle {
                    center,
                    radius,
                    color: ColorRGBA::default(),
                    ..Default::default()
                },
                thickness: 2.0,
                color: ColorRGBA::new(1., 0., 0., 1.),
                _pad: Default::default(),
            },
        }
    }
    pub fn with_color(mut self, color: ColorRGBA) -> Self {
        self.circle.inner.color = color;
        self
    }
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.circle.thickness = thickness;
        self
    }
    pub fn with_stroke_color(mut self, color: ColorRGBA) -> Self {
        self.circle.color = color;
        self
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct InstancePointer {
    pub shape_type: u32,  // 0 = Triangle, 1 = Circle, 2 = Stroke
    pub local_index: u32, // Index into the specific shape buffer
}

impl InstancePointer {
    pub const fn new(shape_type: u32, local_index: u32) -> Self {
        Self {
            shape_type,
            local_index,
        }
    }
}

/// The root object for stack-based rendering. One per drawable entity.
/// Points into a global operation pool buffer with a (start, len) slice.
/// NOT interpolated — topology changes are instantaneous.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct GpuEntityHead {
    /// Index into the global op_pool buffer
    pub stack_start: u32,
    /// Number of operations in this entity's stack
    pub stack_len: u32,
    pub _padding: [u32; 2],
}

/// A standalone stroke modifier. References no inner shape —
/// it transforms the running SDF distance from previous stack entries.
#[derive(Copy, Clone, Pod, Zeroable, Debug, InterpolateM, Default)]
#[repr(C)]
pub struct GpuStrokePacket {
    pub color: ColorRGBA,
    pub thickness: f32,
    pub _pad: [f32; 3],
}

impl GpuStrokePacket {
    pub fn new(color: ColorRGBA, thickness: f32) -> Self {
        Self {
            color,
            thickness,
            _pad: [0.0; 3],
        }
    }
}

impl GpuBindingProvider for GpuStrokePacket {
    const OLD_BINDING: u32 = 4;
    const NEW_BINDING: u32 = 5;
}

/// This is the binding that the shader expects for a given GpuPacket type.
/// It should always be in descriptor set 0, with OLD_BINDING and NEW_BINDING specifying
/// the two bindings for the ping-pong buffers.
pub trait GpuBindingProvider {
    const OLD_BINDING: u32;
    const NEW_BINDING: u32;
}

impl GpuBindingProvider for GpuTrianglePacket {
    const OLD_BINDING: u32 = 0;
    const NEW_BINDING: u32 = 1;
}

impl GpuBindingProvider for GpuCirclePacket {
    const OLD_BINDING: u32 = 2;
    const NEW_BINDING: u32 = 3;
}

impl Interpolate for f32 {
    #[inline]
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        self.lerp(*other, factor)
    }
}

impl Interpolate for Vec2 {
    #[inline]
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        self.lerp(*other, factor)
    }
}
impl Interpolate for Vec3 {
    #[inline]
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        self.lerp(*other, factor)
    }
}
impl Interpolate for Vec4 {
    #[inline]
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        self.lerp(*other, factor)
    }
}

/// We avoid core::array::from_fn because it involves pointer-to-slice casting.
macro_rules! impl_interpolate_array {
    ($($N:expr),+) => {
        $(
            impl<T: Interpolate + Copy> Interpolate for [T; $N] {
                #[inline]
                fn interpolate(&self, other: &Self, factor: f32) -> Self {
                    // We initialize with a copy of self and then overwrite.
                    // This is much safer for the SPIR-V compiler than from_fn.
                    let mut result = *self;
                    let mut i = 0;
                    while i < $N {
                        result[i] = self[i].interpolate(&other[i], factor);
                        i += 1;
                    }
                    result
                }
            }
        )+
    };
}

/// for the given tuple impl interpolate for (T1, T2, ...) where T1: Interpolate, T2: Interpolate,
macro_rules! impl_interpolate_tuple {
    ($($name:ident $num:tt),+) => {
        impl<$($name: Interpolate),+> Interpolate for ($($name,)+) {
            #[inline]
            fn interpolate(&self, other: &Self, factor: f32) -> Self {
                (
                    $(self.$num.interpolate(&other.$num, factor),)+
                )
            }
        }
    };
}

impl_interpolate_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_interpolate_tuple!(T1 0, T2 1);
impl_interpolate_tuple!(T1 0, T2 1, T3 2);
impl_interpolate_tuple!(T1 0, T2 1, T3 2, T4 3);
