//! Shape marker types and StackPattern trait for typed render stack access.
//!
//! Each marker maps to a GPU packet type and shape index. `StackPattern` is
//! implemented for tuples of markers to compute bitmasks and default shapes.

use game_engine_shaders_types::packet::*;
use smallvec::SmallVec;

// ── StackShape trait ─────────────────────────────────────────────────

/// Maps a marker type to its GPU packet type and shape index.
pub trait StackShape {
    type Packet: GpuPacket + Default;
    const SHAPE_INDEX: u32;

    /// Wrap a packet into the `Shape` enum.
    fn wrap(packet: Self::Packet) -> Shape;

    /// Extract a shared reference from a `Shape` enum, if it matches.
    fn unwrap_ref(shape: &Shape) -> Option<&Self::Packet>;

    /// Extract a mutable reference from a `Shape` enum, if it matches.
    fn unwrap_mut(shape: &mut Shape) -> Option<&mut Self::Packet>;
}

// ── Marker types ─────────────────────────────────────────────────────

macro_rules! impl_stack_shape {
    ($(($_index:expr, $SHAPE_INDEX:ident, $_p:ident, $_s:ident, $ShapeType:ident, $GpuPacket:ident)),*) => {
        $(
            pub struct $ShapeType;

            impl StackShape for $ShapeType {
                type Packet = $GpuPacket;
                const SHAPE_INDEX: u32 = $SHAPE_INDEX;

                #[inline]
                fn wrap(packet: Self::Packet) -> Shape {
                    Shape::$ShapeType(packet)
                }

                #[inline]
                fn unwrap_ref(shape: &Shape) -> Option<&Self::Packet> {
                    match shape {
                        Shape::$ShapeType(p) => Some(p),
                        _ => None,
                    }
                }

                #[inline]
                fn unwrap_mut(shape: &mut Shape) -> Option<&mut Self::Packet> {
                    match shape {
                        Shape::$ShapeType(p) => Some(p),
                        _ => None,
                    }
                }
            }
        )*
    };
}

game_engine_shaders_types::for_all_shapes!(impl_stack_shape);

// ── StackPattern trait ───────────────────────────────────────────────

/// Describes a pattern of shapes that a render stack should contain.
/// Implemented for tuples of `StackShape` markers.
pub trait StackPattern: 'static + Send + Sync {
    /// Combined bitmask of all shapes in this pattern.
    fn bitmask() -> u32;

    /// Number of shapes in this pattern.
    fn len() -> usize;

    /// Generate default shapes for stack creation.
    fn default_shapes() -> SmallVec<Shape, 8>;

    /// Shape indices in order.
    fn shape_indices() -> SmallVec<u32, 8>;
}

// ── Tuple impls via macro ────────────────────────────────────────────

macro_rules! impl_stack_pattern {
    ($($T:ident),+) => {
        impl<$($T: StackShape + std::marker::Send + std::marker::Sync + 'static),+> StackPattern for ($($T,)+) {
            #[inline]
            fn bitmask() -> u32 {
                0 $(| (1u32 << $T::SHAPE_INDEX))+
            }

            #[inline]
            fn len() -> usize {
                impl_stack_pattern!(@count $($T),+)
            }

            fn default_shapes() -> SmallVec<Shape, 8> {
                let mut v = SmallVec::new();
                $(v.push($T::wrap($T::Packet::default()));)+
                v
            }

            fn shape_indices() -> SmallVec<u32, 8> {
                let mut v = SmallVec::new();
                $(v.push($T::SHAPE_INDEX);)+
                v
            }
        }
    };

    // Count helper
    (@count $head:ident $(, $tail:ident)*) => {
        1 + impl_stack_pattern!(@count $($tail),*)
    };
    (@count) => { 0 };
}

// Generate impls for tuple sizes 1–8
impl_stack_pattern!(A);
impl_stack_pattern!(A, B);
impl_stack_pattern!(A, B, C);
impl_stack_pattern!(A, B, C, D);
impl_stack_pattern!(A, B, C, D, E);
impl_stack_pattern!(A, B, C, D, E, F);
impl_stack_pattern!(A, B, C, D, E, F, G);
impl_stack_pattern!(A, B, C, D, E, F, G, H);
