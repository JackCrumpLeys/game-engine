use game_engine_derive::Interpolate;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

// --- Components for our ECS - render world --
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}
#[derive(Debug, Clone, Copy)]
pub struct Renderable {
    pub mesh_id: u32,
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
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
