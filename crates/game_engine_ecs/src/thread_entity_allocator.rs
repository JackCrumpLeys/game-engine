use std::sync::{Arc, Mutex, RwLock};

use crate::{
    entity::{Entities, Entity},
    threading::FromWorldThread,
};

const START_BATCH_SIZE: u32 = 16;
const GROWTH_FACTOR: u32 = 2;
const MAX_POW: u32 = 10;
const MAX_BATCH_SIZE: u32 = START_BATCH_SIZE * GROWTH_FACTOR.pow(MAX_POW);
const DECAY_FACTOR: f32 = 0.5; // half the batch size every tick

pub struct LocalThreadEntityAllocator {
    global_allocator: Arc<RwLock<Entities>>,
    next_entities: Vec<Entity>,
    take: u32,
    last_seen_tick: u32,
}

impl FromWorldThread for LocalThreadEntityAllocator {
    fn new_thread_local(_thread_id: usize, world: &crate::world::World) -> Self {
        Self::new(world.entities.clone())
    }
}

impl LocalThreadEntityAllocator {
    pub fn new(global_allocator: Arc<RwLock<Entities>>) -> Self {
        Self {
            global_allocator,
            next_entities: Vec::new(),
            take: START_BATCH_SIZE,
            last_seen_tick: 0,
        }
    }

    /// Allocates a new Entity, syncing with the current tick to adjust batch size.
    /// If we can't satisfy the request from the local pool, we refill from the global allocator.
    /// each time this is called, we check if the current tick has advanced since the last call.
    /// We use this to decay the batch size over time, so that if the allocation slows down, we
    /// don't over-allocate.
    ///
    /// Every time we refill, we multiply the batch size by `GROWTH_FACTOR`, up to `MAX_BATCH_SIZE`.
    pub fn alloc(&mut self, current_tick: u32) -> Entity {
        self.sync_tick(current_tick);
        if self.next_entities.is_empty() {
            self.refill();
        }
        self.next_entities
            .pop()
            .expect("Refill should have populated entities")
    }

    /// Refills the local pool of entities from the global allocator.
    ///
    /// Multiplies the batch size by `GROWTH_FACTOR`, up to `MAX_BATCH_SIZE`.
    fn refill(&mut self) {
        let mut global = self.global_allocator.write().unwrap();
        let batch = global.alloc_batch(self.take as usize);
        self.next_entities = batch;
        self.take = (self.take * GROWTH_FACTOR).min(MAX_BATCH_SIZE);
    }

    /// For each tick that has passed since the last seen tick,
    /// we decay the batch size by `DECAY_FACTOR`.
    fn sync_tick(&mut self, current_tick: u32) {
        if current_tick > self.last_seen_tick {
            debug_assert!(
                current_tick > self.last_seen_tick,
                "Current tick must be greater than last seen tick"
            );

            let diff = current_tick - self.last_seen_tick;

            // If we advanced frames, decay the batch size
            if diff > 0 {
                // Math: size = size * (DECAY_FACTOR ^ diff)
                let decay = DECAY_FACTOR.powi(diff as i32);
                let new_size = (self.last_seen_tick as f32 * decay) as u32;
                self.take = new_size.max(START_BATCH_SIZE);
            }

            self.last_seen_tick = current_tick;
        }
    }
}
