#![no_std]
use bytemuck::{Pod, Zeroable};
use spirv_std::glam::Vec2;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct PushConstants {
    pub position_offset: Vec2,
    pub resolution: Vec2,
    pub time: f32,
}
