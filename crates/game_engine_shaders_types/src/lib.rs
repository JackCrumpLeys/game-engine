#![no_std]

pub mod packet;

use bytemuck::{Pod, Zeroable};
pub use spirv_std::glam::*;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct PushConstants {
    pub position_offset: Vec2,
    pub resolution: Vec2,
    pub time: f32,
    pub factor: f32,
}
