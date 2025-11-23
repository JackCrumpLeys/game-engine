use game_engine_derive::Interpolate;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Clone)]
pub struct RenderPacket {
    // TODO make this a ecs world?
    pub snapped_at: std::time::Instant,
    vertex_buffer: [VulVertex; 3],
}

impl Default for RenderPacket {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPacket {
    pub fn new() -> Self {
        Self {
            snapped_at: std::time::Instant::now(),
            vertex_buffer: Default::default(),
        }
    }

    pub fn vertex_buffer(&self) -> &[VulVertex] {
        &self.vertex_buffer
    }

    #[inline(always)]
    pub fn with_vertex_buffer(mut self, buffer: [VulVertex; 3]) -> Self {
        self.vertex_buffer = buffer;
        self
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex, Interpolate, Copy, Clone, Default)]
#[repr(C)]
pub struct VulVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl VulVertex {
    pub fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

pub trait Interpolate {
    /// interpolates between self and other by factor (0.0 - 1.0)
    fn interpolate(&self, other: &Self, factor: f32) -> Self;
}

impl Interpolate for RenderPacket {
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        // find time between the two snapped_at times indexed by factor
        let duration_between = other
            .snapped_at
            .duration_since(self.snapped_at)
            .as_secs_f32();
        let snapped_at = self
            .snapped_at
            .checked_add(std::time::Duration::from_secs_f32(
                duration_between * factor,
            ))
            .unwrap_or(other.snapped_at);

        Self {
            snapped_at,
            vertex_buffer: self.vertex_buffer.interpolate(&other.vertex_buffer, factor),
        }
    }
}

impl Interpolate for f32 {
    fn interpolate(&self, other: &Self, factor: f32) -> Self {
        self + (other - self) * factor
    }
}

/// For the given N impl interpolate for [T;N] where T: Interpolate
macro_rules! impl_interpolate_array {
    ($($N:expr),+) => {
        $(
            impl<T: Interpolate> Interpolate for [T; $N] {
                fn interpolate(&self, other: &Self, factor: f32) -> Self {
                    let mut result: [T; $N] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                    for i in 0..$N {
                        result[i] = self[i].interpolate(&other[i], factor);
                    }
                    result
                }
            }
        )+
    };
}
/// for the given tuple impl interpolate for (T1, T2, ...) where T1: Interpolate, T2: Interpolate,
macro_rules! impl_interpolate_tuple {
    ($($name:ident $num:tt),+) => {
        impl<$($name: Interpolate),+> Interpolate for ($($name,)+) {
            fn interpolate(&self, other: &Self, factor: f32) -> Self {
                (
                    $(self.$num.interpolate(&other.$num, factor),)+
                )
            }
        }
    };
}

impl_interpolate_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_interpolate_tuple!(T1 0, T2 1);
impl_interpolate_tuple!(T1 0, T2 1, T3 2);
impl_interpolate_tuple!(T1 0, T2 1, T3 2, T4 3);

/// The renderer will interpolate between two snapshots to produce smooth animations.
pub struct SnapshotPair {
    old: RenderPacket,
    new: RenderPacket,
}

impl SnapshotPair {
    pub fn new(old: RenderPacket, new: RenderPacket) -> Self {
        Self { old, new }
    }

    /// replaces the "new" snapshot with the provided one, and moves the previous "new" to "old"
    pub fn push_new(&mut self, new: RenderPacket) {
        self.old = self.interpolate();
        self.new = new;
    }

    pub fn interpolate(&self) -> RenderPacket {
        let now = std::time::Instant::now();
        let duration_since_new = now.duration_since(self.new.snapped_at);
        let old_to_new_duration = self.new.snapped_at.duration_since(self.old.snapped_at);

        // The next expected snapshot is at difference between old and new + the old's snapped_at time
        // We need to interpolate such that factor = 0.9 at the expected next snapshot time

        let factor = if old_to_new_duration.as_millis() == 0 {
            1.0
        } else {
            let raw_factor = duration_since_new.as_secs_f32() / old_to_new_duration.as_secs_f32();
            raw_factor.clamp(0.0, 1.0)
        };

        self.old.interpolate(&self.new, factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // Helper to create a packet with specific coordinates at a specific time
    fn create_packet(x: f32, y: f32, time_offset_ms: u64) -> RenderPacket {
        // We subtract from now to simulate packets created in the past
        let snapped_at = std::time::Instant::now()
            .checked_sub(Duration::from_millis(time_offset_ms))
            .unwrap_or(std::time::Instant::now());

        RenderPacket {
            snapped_at,
            vertex_buffer: [
                VulVertex { position: [x, y] }, // Vertex 0
                VulVertex {
                    position: [x + 1.0, y + 1.0],
                }, // Vertex 1 (Just to have data)
                VulVertex {
                    position: [x + 2.0, y + 2.0],
                }, // Vertex 2
            ],
        }
    }

    #[test]
    fn test_f32_interpolation() {
        let a = 0.0;
        let b = 10.0;
        assert_eq!(a.interpolate(&b, 0.5), 5.0);
        assert_eq!(a.interpolate(&b, 0.1), 1.0);
        assert_eq!(a.interpolate(&b, 0.9), 9.0);
    }

    #[test]
    fn test_vertex_interpolation() {
        let v1 = VulVertex {
            position: [0.0, 0.0],
        };
        let v2 = VulVertex {
            position: [10.0, 20.0],
        };

        let result = v1.interpolate(&v2, 0.5);
        assert_eq!(result.position, [5.0, 10.0]);
    }

    #[test]
    fn test_array_interpolation_derive_logic() {
        // Testing the macro_rules! impl_interpolate_array
        let arr1 = [0.0, 0.0, 0.0];
        let arr2 = [10.0, 20.0, 30.0];

        let result = arr1.interpolate(&arr2, 0.5);
        assert_eq!(result, [5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_stress_smoothness_under_sim_variance() {
        use rand::Rng; // Assuming rand is available, or use simple math for variance

        // SCENARIO:
        // Object moves at constant speed: 100 units / second.
        // Render Loop: Runs every ~8ms (120 FPS).
        // Sim Loop:    Runs variably between 50ms and 150ms (Chaotic 7-20 FPS).
        //
        // GOAL:
        // Even though the Sim is stuttering and lagging, the Renderer should
        // never see the object "teleport" (move more than expected for 8ms).

        let velocity = 100.0;

        // Initial State
        // P1: t=0, pos=0
        // P2: t=100ms, pos=10
        let p1 = create_packet(0.0, 0.0, 100);
        let p2 = create_packet(10.0, 0.0, 0);

        let mut pair = SnapshotPair::new(p1, p2);

        let mut current_sim_pos = 10.0;
        let mut next_sim_update = std::time::Instant::now() + Duration::from_millis(100);

        let mut last_render_pos = 0.0; // Approximate starting point
        let mut last_render_time = std::time::Instant::now();

        // Run for 200 "Render Frames"
        for i in 0..200 {
            // 1. Simulate Render Frame Time (High FPS)
            thread::sleep(Duration::from_millis(8));
            let now = std::time::Instant::now();

            // 2. Check if we need to push a Sim Update (Variable/Chaotic)
            if now >= next_sim_update {
                // Advance Sim State
                // Sim moves forward by a random chunk (mimicking variable tick rate)
                let dt_sim_ms = rand::rng().random_range(50..150);
                let dt_sim = dt_sim_ms as f32 / 1000.0;

                current_sim_pos += velocity * dt_sim; // Move physics forward

                // Create packet "just arrived"
                let new_packet = RenderPacket {
                    snapped_at: now,
                    vertex_buffer: [VulVertex {
                        position: [current_sim_pos, 0.0],
                    }; 3],
                };

                // THE CRITICAL MOMENT: Pushing new state
                pair.push_new(new_packet);

                // Schedule next chaotic update
                next_sim_update = now + Duration::from_millis(dt_sim_ms);
            }

            // 3. Render
            let frame = pair.interpolate();
            let current_render_pos = frame.vertex_buffer[0].position[0];

            // 4. Analysis
            if i > 0 {
                let delta_pos = (current_render_pos - last_render_pos).abs();
                let delta_time = now.duration_since(last_render_time).as_secs_f32();

                // Expected movement for this time slice = Velocity * Time
                let expected_move = velocity * delta_time;

                // TOLERANCE:
                // We allow the renderer to be slightly faster/slower to catch up,
                // but a "Jump" would be snapping to the new position instantly.
                // If we moved 5x what we expected in 8ms, that's a visual pop.
                let panic_threshold = expected_move * 5.0 + 2.0; // +2.0 buffer for tiny float errors

                assert!(
                    delta_pos < panic_threshold,
                    "JUMP DETECTED at frame {}! \n\
                    Time Delta: {:.4}s \n\
                    Moved: {:.4} units \n\
                    Expected: {:.4} units \n\
                    Sim Pos: {:.4} \n\
                    Render Pos: {:.4}",
                    i,
                    delta_time,
                    delta_pos,
                    expected_move,
                    current_sim_pos,
                    current_render_pos
                );
            }

            last_render_pos = current_render_pos;
            last_render_time = now;
        }
    }

    #[test]
    fn test_clamping_if_sim_hangs() {
        // If the simulation hangs and no new packets arrive for 5 seconds:
        let old = create_packet(0.0, 0.0, 5000);
        let new = create_packet(10.0, 0.0, 4000); // Arrived 4s ago

        let pair = SnapshotPair::new(old, new);

        // duration_since_new = 4000.
        // delta = 1000.
        // factor = 4.0 * 0.9 = 3.6.
        // Should clamp to 1.0.

        let frame = pair.interpolate();
        assert_eq!(
            frame.vertex_buffer[0].position[0], 10.0,
            "Should clamp to destination"
        );
    }
}
