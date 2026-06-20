use game_engine_derive::{Component, Interpolate};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

// --- Components for our ECS - render world --
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, Component)]
pub struct Renderable {
    pub mesh_id: u32,
}

#[repr(C)]
#[derive(BufferContents, Vertex, Interpolate, Copy, Clone, Default)]
pub struct VulVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl VulVertex {
    pub fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

// use game_engine_ecs::prelude::*;
use game_engine_shaders_types::packet::Shape;
use smallvec::SmallVec;

#[derive(Component, Clone, Default, Debug)]
pub struct ShapeStack(pub SmallVec<Shape, 8>);

#[derive(Component, Clone, Copy, Debug)]
pub struct RenderStack {
    pub gpu_head_idx: u32, // Index into GpuEntityHead buffer (StackId)
    pub op_pool_base: u32, // Offset in InstancePointer pool
    pub op_pool_len: u8,   // Number of shapes
}

#[derive(Component, Clone, Debug)]
pub struct GpuAllocations {
    pub slots: SmallVec<u32, 8>, // Actual indices in the primitive channels
}
