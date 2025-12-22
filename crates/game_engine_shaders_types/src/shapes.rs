use crate::packet::Interpolate;
use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::{Vec2, Vec4};

#[derive(Copy, Clone, Zeroable, Pod, Debug, InterpolateM)]
#[repr(C)]
pub struct ColorRGBA(Vec4);

impl Default for ColorRGBA {
    fn default() -> Self {
        Self(Vec4::new(0.0, 1.0, 0.0, 1.0)) //bright green
    }
}

impl ColorRGBA {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(Vec4::new(r, g, b, a))
    }

    pub fn get(self) -> Vec4 {
        self.0
    }

    pub fn red(&self) -> f32 {
        self.0.x
    }

    pub fn red_mut(&mut self) -> &mut f32 {
        &mut self.0.x
    }

    pub fn green(&self) -> f32 {
        self.0.y
    }

    pub fn green_mut(&mut self) -> &mut f32 {
        &mut self.0.y
    }

    pub fn blue(&self) -> f32 {
        self.0.z
    }

    pub fn blue_mut(&mut self) -> &mut f32 {
        &mut self.0.z
    }

    pub fn alpha(&self) -> f32 {
        self.0.w
    }

    pub fn alpha_mut(&mut self) -> &mut f32 {
        &mut self.0.w
    }
}

#[derive(Copy, Clone, Zeroable, Debug, InterpolateM)]
#[repr(C)]
pub struct Stroked<T: Pod + Zeroable + Interpolate> {
    pub inner: T,
    pub thickness: f32,
    pub color: ColorRGBA,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Polygon<const N: usize> {
    pub vertices: [Vec2; N],
    pub color: ColorRGBA,
}

// 2. Manually implement Zeroable
// SAFETY: The struct consists solely of a Zeroable array.
unsafe impl<const N: usize> Zeroable for Polygon<N> {}

// 3. Manually implement Pod
// SAFETY: The struct is repr(C), has no padding, and contains only Pod data.
unsafe impl<const N: usize> Pod for Polygon<N> {}

impl<const N: usize> Interpolate for Polygon<N> {
    #[inline]
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        let mut result = *self;
        for i in 0..N {
            result.vertices[i] = self.vertices[i].interpolate(&other.vertices[i], factor);
        }
        result
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default, InterpolateM, Debug)]
#[repr(C)]
pub struct Rect {
    pub center: Vec2,
    pub size: Vec2,
    pub color: ColorRGBA,
}
#[derive(Copy, Clone, Pod, Zeroable, Default, InterpolateM, Debug)]
#[repr(C)]
pub struct Circle {
    pub color: ColorRGBA,
    pub center: Vec2,
    pub radius: f32,
    pub _padding: [f32; 1],
}

#[macro_export]
macro_rules! unsafe_impl_stroked_pod {
    // Case 1: Simple, non-generic types (e.g., Rect)
    // Usage: unsafe_impl_stroked_pod!(Rect, test_rect_pod_safety);
    ($t:ty, $test_name:ident) => {
        // 1. The Unsafe Implementation
        unsafe impl bytemuck::Pod for Stroked<$t> {}

        // 2. The Test
        $crate::unsafe_impl_stroked_pod!(@internal_test $t, $test_name);
    };

    // Case 2: Generic types (e.g., Polygon<N>)
    // Usage: unsafe_impl_stroked_pod!(generics: [const N: usize], type: Polygon<N>, test_sample: Polygon<3>, name: test_poly_pod_safety);
    (generics: [$($gen_params:tt)*], type: $t:ty, test_sample: $concrete:ty, name: $test_name:ident) => {
        // 1. The Unsafe Implementation (with generics)
        unsafe impl<$($gen_params)*> bytemuck::Pod for Stroked<$t> {}

        // 2. The Test (using the concrete sample)
        $crate::unsafe_impl_stroked_pod!(@internal_test $concrete, $test_name);
    };

    // Internal helper to generate the test logic
    (@internal_test $concrete:ty, $test_name:ident) => {
        #[cfg(test)]
        #[test]
        fn $test_name() {
            use std::mem;

            let struct_size = mem::size_of::<Stroked<$concrete>>();
            let inner_size = mem::size_of::<$concrete>();
            let thick_size = mem::size_of::<f32>();

            // If the struct size is larger than the sum of its parts,
            // the compiler added padding bytes => Undefined Behavior for Pod.
            assert_eq!(
                struct_size,
                inner_size + thick_size,
                "\n\n[SAFETY FAILURE] Manual Pod implementation for Stroked<{}> is unsafe!\n\
                 Struct Size: {} bytes\n\
                 Sum of Fields: {} bytes (Inner: {} + Thickness: {})\n\
                 The compiler has inserted padding bytes. You cannot use Pod.\n",
                stringify!($concrete), struct_size, (inner_size + thick_size), inner_size, thick_size
            );
        }
    };
}

// SAFETY: All of these are basicly a bunch of floats. so have alignment of 4 bytes and no padding.
unsafe_impl_stroked_pod!(Rect, test_rect_layout);
unsafe_impl_stroked_pod!(Circle, test_circle_layout);

// We must provide a "test_sample" (like Polygon<3>) because we can't
// run a unit test on an abstract Polygon<N>.
unsafe_impl_stroked_pod!(
    generics: [const N: usize],
    type: Polygon<N>,
    test_sample: Polygon<3>,
    name: test_polygon_layout
);
