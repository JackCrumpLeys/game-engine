use bytemuck::{Pod, Zeroable};
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4};

#[macro_export]
macro_rules! for_all_shapes {
    ($macro_name:ident) => {
        $macro_name! {
            (0, TRIANGLE_SHAPE_INDEX, triangles, triangle, Triangle, GpuTrianglePacket),
            (1, CIRCLE_SHAPE_INDEX, circles, circle, Circle, GpuCirclePacket),
            (2, STROKE_SHAPE_INDEX, strokes, stroke, Stroke, GpuStrokePacket),
            (3, ROUNDED_RECT_SHAPE_INDEX, rounded_rects, rounded_rect, RoundedRect, GpuRoundedRectPacket),
            (4, LINESEGMENT_SHAPE_INDEX, line_segments, line_segment, LineSegment, GpuLineSegmentPacket),
            (5, RECT_SHAPE_INDEX, rects, rect, Rect, GpuRectPacket),
            (6, LINEAR_GRADIENT_SHAPE_INDEX, linear_gradients, linear_gradient, LinearGradient, GpuLinearGradientPacket),
            (7, RADIAL_GRADIENT_SHAPE_INDEX, radial_gradients, radial_gradient, RadialGradient, GpuRadialGradientPacket),
            (8, CLIP_RECT_SHAPE_INDEX, clip_rects, clip_rect, ClipRect, GpuClipRectPacket),
            (9, SPRITE_SHAPE_INDEX, sprites, sprite, Sprite, GpuSpritePacket),
            (10, GLYPH_SHAPE_INDEX, glyphs, glyph, Glyph, GpuGlyphPacket),
            (11, BEZIER_PATH_SHAPE_INDEX, bezier_paths, bezier_path, BezierPath, GpuBezierPathPacket)
        }
    };
}

macro_rules! define_shape_indices {
    ($( ($idx:expr, $const_name:ident, $channel_name:ident, $builder_name:ident, $enum_name:ident, $packet_type:ident) ),*) => {
        $(
            pub const $const_name: u32 = $idx;
        )*
        pub const NUM_SHAPE_KINDS: usize = 0 $(+ { let _ = $idx; 1 })*;
    };
}
for_all_shapes!(define_shape_indices);

// ── Traits ───────────────────────────────────────────────────────────

pub trait Interpolate {
    fn interpolate(&self, other: &Self, factor: f32) -> Self;
}

pub trait GpuPacket: Interpolate + Clone + Pod + Zeroable {}
impl<T: Interpolate + Clone + Pod + Zeroable> GpuPacket for T {}

pub trait GpuBindingProvider {
    const OLD_BINDING: u32;
    const NEW_BINDING: u32;
}

// ── Shared GPU structures ────────────────────────────────────────────

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct InstancePointer {
    pub shape_type: u32,
    pub local_index: u32,
}

impl InstancePointer {
    pub const fn new(shape_type: u32, local_index: u32) -> Self {
        Self { shape_type, local_index }
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct GpuEntityHead {
    pub stack_start: u32,
    pub stack_len: u32,
    pub position: [f32; 2],
}

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct GpuCurve {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
    pub _pad: Vec2, // Pad to 32 bytes for GPU alignment
}

// ── Re-exports (all packet structs live in their shape files) ────────

pub use crate::shapes::circle::GpuCirclePacket;
pub use crate::shapes::polygon::GpuTrianglePacket;
pub use crate::shapes::rect::GpuRectPacket;
pub use crate::shapes::rounded_rect::GpuRoundedRectPacket;
pub use crate::shapes::line_segment::GpuLineSegmentPacket;
pub use crate::shapes::stroked::GpuStrokePacket;
pub use crate::shapes::gradient::{GpuLinearGradientPacket, GpuRadialGradientPacket};
pub use crate::shapes::clip::GpuClipRectPacket;
pub use crate::shapes::sprite::GpuSpritePacket;
pub use crate::shapes::glyph::GpuGlyphPacket;
pub use crate::shapes::bezier_path::GpuBezierPathPacket;

// ── Shape enum ───────────────────────────────────────────────────────

macro_rules! define_shape_enum {
    ($( ($idx:expr, $const_name:ident, $channel_name:ident, $builder_name:ident, $enum_name:ident, $packet_type:ident) ),*) => {
        /// A typed, non-erased render operation for the ECS render queue.
        #[derive(Copy, Clone, Debug)]
        pub enum Shape {
            $(
                $enum_name($packet_type),
            )*
        }

        impl Shape {
            pub fn shape_index(&self) -> u32 {
                match self {
                    $( Shape::$enum_name(_) => $const_name, )*
                }
            }

            pub fn as_bytes(&self) -> &[u8] {
                match self {
                    $( Shape::$enum_name(p) => bytemuck::bytes_of(p), )*
                }
            }
        }
    };
}
for_all_shapes!(define_shape_enum);

// ── Interpolation impls for primitives ───────────────────────────────

impl Interpolate for f32 {
    #[inline] fn interpolate(&self, other: &Self, factor: f32) -> Self { self.lerp(*other, factor) }
}
impl Interpolate for u32 {
    #[inline] fn interpolate(&self, other: &Self, factor: f32) -> Self { if factor < 0.5 { *self } else { *other } }
}
impl Interpolate for Vec2 {
    #[inline] fn interpolate(&self, other: &Self, factor: f32) -> Self { self.lerp(*other, factor) }
}
impl Interpolate for Vec3 {
    #[inline] fn interpolate(&self, other: &Self, factor: f32) -> Self { self.lerp(*other, factor) }
}
impl Interpolate for Vec4 {
    #[inline] fn interpolate(&self, other: &Self, factor: f32) -> Self { self.lerp(*other, factor) }
}

macro_rules! impl_interpolate_array {
    ($($N:expr),+) => { $(
        impl<T: Interpolate + Copy> Interpolate for [T; $N] {
            #[inline]
            fn interpolate(&self, other: &Self, factor: f32) -> Self {
                let mut result = *self;
                let mut i = 0;
                while i < $N { result[i] = self[i].interpolate(&other[i], factor); i += 1; }
                result
            }
        }
    )+ };
}

macro_rules! impl_interpolate_tuple {
    ($($name:ident $num:tt),+) => {
        impl<$($name: Interpolate),+> Interpolate for ($($name,)+) {
            #[inline]
            fn interpolate(&self, other: &Self, factor: f32) -> Self {
                ($(self.$num.interpolate(&other.$num, factor),)+)
            }
        }
    };
}

impl_interpolate_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_interpolate_tuple!(T1 0, T2 1);
impl_interpolate_tuple!(T1 0, T2 1, T3 2);
impl_interpolate_tuple!(T1 0, T2 1, T3 2, T4 3);
