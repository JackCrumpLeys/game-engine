use crate::packet::{GpuPacket, Interpolate};
use bytemuck::{Pod, Zeroable};
use game_engine_derive::InterpolateM;
use spirv_std::glam::Vec4;

// ── ColorRGBA ────────────────────────────────────────────────────────

#[derive(Copy, Clone, Zeroable, Pod, Debug, InterpolateM)]
#[repr(C)]
pub struct ColorRGBA(Vec4);

impl Default for ColorRGBA {
    fn default() -> Self {
        Self(Vec4::new(0.0, 1.0, 0.0, 1.0))
    }
}

impl ColorRGBA {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self(Vec4::new(r, g, b, a)) }
    pub fn get(self) -> Vec4 { self.0 }
    pub fn red(&self) -> f32 { self.0.x }
    pub fn red_mut(&mut self) -> &mut f32 { &mut self.0.x }
    pub fn green(&self) -> f32 { self.0.y }
    pub fn green_mut(&mut self) -> &mut f32 { &mut self.0.y }
    pub fn blue(&self) -> f32 { self.0.z }
    pub fn blue_mut(&mut self) -> &mut f32 { &mut self.0.z }
    pub fn alpha(&self) -> f32 { self.0.w }
    pub fn alpha_mut(&mut self) -> &mut f32 { &mut self.0.w }

    pub fn from_hsl(h: f32, s: f32, l: f32) -> Self {
        use spirv_std::glam::Vec3;
        let k = Vec3::new(0.0, 4.0, 2.0);
        let t = (Vec3::splat(h) * 6.0 + k) % 6.0;
        let rgb = ((t - 3.0).abs() - 1.0).clamp(Vec3::ZERO, Vec3::ONE);
        Self((l + s * (rgb - 0.5) * (1.0 - (2.0 * l - 1.0).abs())).extend(1.0))
    }
}

// ── Stroked wrapper (generic, used by old composite shapes) ──────────

#[derive(Copy, Clone, Zeroable, Debug, InterpolateM)]
#[repr(C)]
pub struct Stroked<T>
where T: GpuPacket,
{
    pub inner: T,
    pub color: ColorRGBA,
    pub thickness: f32,
    pub _pad: [f32; 3],
}

#[macro_export]
macro_rules! unsafe_impl_stroked_pod {
    ($t:ty, $test_name:ident) => {
        unsafe impl bytemuck::Pod for $crate::shapes::Stroked<$t> {}
        $crate::unsafe_impl_stroked_pod!(@internal_test $t, $test_name);
    };
    (generics: [$($gen_params:tt)*], type: $t:ty, test_sample: $concrete:ty, name: $test_name:ident) => {
        unsafe impl<$($gen_params)*> bytemuck::Pod for $crate::shapes::Stroked<$t> {}
        $crate::unsafe_impl_stroked_pod!(@internal_test $concrete, $test_name);
    };
    (@internal_test $concrete:ty, $test_name:ident) => {
        #[cfg(test)]
        #[test]
        fn $test_name() {
            use std::mem;
            use $crate::shapes::{Stroked, ColorRGBA};
            let struct_size = mem::size_of::<Stroked<$concrete>>();
            let inner_size = mem::size_of::<$concrete>();
            let thick_size = mem::size_of::<f32>();
            let color_size = mem::size_of::<ColorRGBA>();
            assert_eq!(
                struct_size,
                inner_size + thick_size + color_size + 12,
                "\n\n[SAFETY FAILURE] Manual Pod for Stroked<{}> is unsafe!\n\
                 Struct: {} bytes, Sum: {} bytes\n",
                stringify!($concrete), struct_size, (inner_size + thick_size + color_size)
            );
        }
    };
}

unsafe_impl_stroked_pod!(crate::shapes::rect::GpuRectPacket, test_rect_layout);
unsafe_impl_stroked_pod!(crate::shapes::circle::GpuCirclePacket, test_circle_layout);
