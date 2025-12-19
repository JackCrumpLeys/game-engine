use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{FloatExt, Vec2};

pub trait Interpolate {
    /// interpolates between self and other by factor (0.0 - 1.0)
    fn interpolate(&self, other: &Self, factor: f32) -> Self;
}

pub trait GpuPacket: Interpolate + Clone + Pod + Zeroable {}

/// A single triangle primitive.
/// This is treated as one "instance" by the renderer.
#[derive(Copy, Clone, Pod, Zeroable, Default, InterpolateM, Debug)]
#[repr(C)]
pub struct GpuTrianglePacket {
    pub vertices: [Vec2; 3],
}

impl GpuTrianglePacket {
    pub fn new(v1: Vec2, v2: Vec2, v3: Vec2) -> Self {
        Self {
            vertices: [v1, v2, v3],
        }
    }
}

impl GpuPacket for GpuTrianglePacket {}

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
