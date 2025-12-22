use std::time::{Duration, Instant};

use game_engine_shaders_types::packet::GpuTrianglePacket;

use crate::render::storage::RenderPacket;

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

/// Manages the visual timeline by holding the two most recent simulation states.
#[derive(Debug)]
pub struct SnapshotPair {
    pub old: Instant,
    pub new: Instant,
}

impl SnapshotPair {
    pub fn new(old: Instant, new: Instant) -> Self {
        Self { old, new }
    }

    pub fn from_packets(old: &RenderPacket, new: &RenderPacket) -> Self {
        Self {
            old: old.snapped_at,
            new: new.snapped_at,
        }
    }

    pub fn new_empty() -> Self {
        let now = Instant::now();
        Self { old: now, new: now }
    }

    pub fn push_new(&mut self, new: Instant) {
        self.old = self.new;
        self.new = new;
    }

    /// Calculates the GPU alpha (0.0 to 1.0).
    /// 0.0 means we are at the 'Old' state.
    /// 1.0 means we have reached the 'New' state.
    pub fn interpolation_factor(&self) -> f32 {
        let now = Instant::now();

        // How much time has passed since the newest data point?
        let duration_since_new = now.duration_since(self.new);

        // How long was the interval between our two known data points?
        // We assume the next tick will take roughly the same amount of time.
        let old_to_new_duration = self.new.duration_since(self.old);

        if old_to_new_duration.is_zero() {
            return 1.0;
        }

        // We are interpolating from Old -> New.
        // We hit 1.0 when we are 'one interval' past the New snapshot's time.
        let raw_factor = duration_since_new.as_secs_f32() / old_to_new_duration.as_secs_f32();
        raw_factor.clamp(0.0, 1.0)
    }
}
