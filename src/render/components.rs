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
