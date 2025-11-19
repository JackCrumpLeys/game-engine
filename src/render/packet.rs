pub struct RenderPacket {
    snapped_at: std::time::Instant,
    vertex_buffer: Vec<VulVertex>,
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VulVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub trait Interpolate {
    /// interpolates between self and other by factor (0.0 - 1.0)
    fn interpolate(&self, other: &Self, factor: f32) -> Self;
}

impl Interpolate for RenderPacket {
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        Self {
            snapped_at: self.snapped_at, // Mabe not the best idea, but whatever
            vertex_buffer: self
                .vertex_buffer
                .iter()
                .zip(other.vertex_buffer.iter())
                .map(|(a, b)| a.interpolate(b, factor))
                .collect(),
        }
    }
}

impl Interpolate for VulVertex {
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        VulVertex {
            position: [
                self.position[0].interpolate(&other.position[0], factor),
                self.position[1].interpolate(&other.position[1], factor),
            ],
        }
    }
}

impl Interpolate for f32 {
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        self + (other - self) * factor
    }
}
