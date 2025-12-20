use std::time::{Duration, Instant};

use game_engine_shaders_types::packet::GpuTrianglePacket;

/// A generic container for a specific primitive's data.
/// This matches the SoA (Structure of Arrays) layout expected by the GPU.
#[derive(Clone, Default, Debug)]
pub struct RenderPacketContents<T> {
    /// The "Database": Contains data for every allocated slot of this primitive type.
    pub data: Vec<T>,
    /// The "Indirection List": Contains indices of entities that are actually alive.
    pub active_indices: Vec<u32>,
    /// The indices that are new to this packet (not present in the previous one).
    pub newly_spawned_indices: Vec<u32>,
}

impl<T> RenderPacketContents<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            active_indices: Vec::new(),
            newly_spawned_indices: Vec::new(),
        }
    }

    pub fn new_all_data(data: Vec<T>, newly_spawned_indices: Vec<u32>) -> Self {
        let active_indices = (0..data.len() as u32).collect();
        Self {
            data,
            active_indices,
            newly_spawned_indices,
        }
    }

    pub fn clone_data(&self) -> Self
    where
        T: Clone,
    {
        Self {
            data: self.data.clone(),
            active_indices: self.active_indices.clone(),
            newly_spawned_indices: Vec::new(),
        }
    }

    pub fn insert(&mut self, index: u32, data: T) {
        self.active_indices.push(index);
        self.data.push(data);
        self.newly_spawned_indices.push(index);
    }

    pub fn clear(&mut self) {
        self.active_indices.clear();
    }
}

/// A unified snapshot of the simulation world at a specific moment in time.
#[derive(Clone, Debug)]
pub struct RenderPacket {
    /// When this snapshot was captured.
    pub snapped_at: Instant,
    /// Collection of all triangle primitives in the world.
    pub triangles: RenderPacketContents<GpuTrianglePacket>,
}

impl RenderPacket {
    pub fn new() -> Self {
        Self {
            snapped_at: Instant::now(),
            triangles: RenderPacketContents::new(),
        }
    }
}

/// Manages the visual timeline by holding the two most recent simulation states.
#[derive(Debug)]
pub struct SnapshotPair {
    pub old: RenderPacket,
    pub new: RenderPacket,
}

impl SnapshotPair {
    pub fn new(old: RenderPacket, new: RenderPacket) -> Self {
        Self { old, new }
    }

    pub fn new_empty() -> Self {
        let empty_packet = RenderPacket::new();
        Self {
            old: empty_packet.clone(),
            new: empty_packet,
        }
    }

    /// Shifts the current 'new' state to 'old' and accepts a fresh snapshot.
    pub fn push_new(&mut self, fresh: RenderPacket) {
        self.old = std::mem::replace(&mut self.new, fresh);
    }

    /// Calculates the GPU alpha (0.0 to 1.0).
    /// 0.0 means we are at the 'Old' state.
    /// 1.0 means we have reached the 'New' state.
    pub fn interpolation_factor(&self) -> f32 {
        let now = Instant::now();

        // How much time has passed since the newest data point?
        let duration_since_new = now.duration_since(self.new.snapped_at);

        // How long was the interval between our two known data points?
        // We assume the next tick will take roughly the same amount of time.
        let old_to_new_duration = self.new.snapped_at.duration_since(self.old.snapped_at);

        if old_to_new_duration.is_zero() {
            return 1.0;
        }

        // We are interpolating from Old -> New.
        // We hit 1.0 when we are 'one interval' past the New snapshot's time.
        let raw_factor = duration_since_new.as_secs_f32() / old_to_new_duration.as_secs_f32();
        raw_factor.clamp(0.0, 1.0)
    }
}
