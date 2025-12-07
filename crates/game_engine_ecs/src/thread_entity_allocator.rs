use std::sync::{Arc, RwLock};

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

    /// Allocates a batch of Entities.
    /// Optimized to fetch the exact required amount from global state if the local
    /// cache is insufficient, bypassing the exponential growth logic used for single allocations.
    pub fn alloc_batch(&mut self, current_tick: u32, count: usize) -> Vec<Entity> {
        self.sync_tick(current_tick);

        // 1. If we have enough in the buffer, just take them from the end (LIFO-ish for cache, but FIFO for IDs)
        if self.next_entities.len() >= count {
            let start_index = self.next_entities.len() - count;
            return self.next_entities.drain(start_index..).collect();
        }

        let mut result = Vec::with_capacity(count);

        // 2. Drain whatever we currently have in the cache
        result.append(&mut self.next_entities);

        // 3. Allocate the EXACT difference from the global allocator
        let needed = count - result.len();
        if needed > 0 {
            let mut global = self.global_allocator.write().unwrap();
            let batch = global.alloc_batch(needed);
            result.extend(batch);
        }

        self.refill();

        result
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

#[cfg(test)]
mod entity_allocator_tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_alloc_refills_lazily() {
        let global = Arc::new(RwLock::new(Entities::new()));
        let mut local = LocalThreadEntityAllocator::new(global.clone());

        // 1. Initial State
        assert_eq!(local.next_entities.len(), 0);
        assert_eq!(global.read().unwrap().len(), 0);

        // 2. Alloc one
        // Should trigger refill of START_BATCH_SIZE
        let _e = local.alloc(0);

        // Global len should be START_BATCH_SIZE
        assert_eq!(global.read().unwrap().len(), START_BATCH_SIZE as usize);
        // Local buffer should have (START - 1) left
        assert_eq!(local.next_entities.len(), (START_BATCH_SIZE - 1) as usize);
    }

    #[test]
    fn test_growth_strategy() {
        let global = Arc::new(RwLock::new(Entities::new()));
        let mut local = LocalThreadEntityAllocator::new(global);

        // 1. First batch
        let _ = local.alloc(0);

        let expected_stage_1 = START_BATCH_SIZE * GROWTH_FACTOR;
        assert_eq!(local.take, expected_stage_1);

        // 2. Consume the rest of the buffer
        local.next_entities.clear();

        // 3. Alloc again -> Triggers refill
        let _ = local.alloc(0);

        let expected_stage_2 = expected_stage_1 * GROWTH_FACTOR;
        assert_eq!(local.take, expected_stage_2);

        // Ensure we don't exceed max (sanity check, assuming test doesn't run billions of times)
        assert!(local.take <= MAX_BATCH_SIZE);
    }

    #[test]
    fn test_decay_strategy() {
        let global = Arc::new(RwLock::new(Entities::new()));
        let mut local = LocalThreadEntityAllocator::new(global);

        // Ramp up the growth manually to test decay
        // Start: START_BATCH_SIZE
        local.take = START_BATCH_SIZE * GROWTH_FACTOR * GROWTH_FACTOR; // e.g., 64 if start=16, growth=2
        let current_take = local.take;

        // Advance Tick
        // Logic: if current_tick > last_seen, decay.
        local.alloc(1);

        let expected_decay_1 = (current_take as f32 * DECAY_FACTOR) as u32;
        assert_eq!(local.take, expected_decay_1.max(START_BATCH_SIZE));

        // Advance again
        local.alloc(2);
        let expected_decay_2 = (local.take as f32 * DECAY_FACTOR) as u32;
        assert_eq!(local.take, expected_decay_2.max(START_BATCH_SIZE));

        // Verify min clamp
        // Force take to be small, then decay
        local.take = START_BATCH_SIZE;
        local.alloc(3);
        assert_eq!(local.take, START_BATCH_SIZE);
    }

    #[test]
    fn test_batch_passthrough() {
        let global = Arc::new(RwLock::new(Entities::new()));
        let mut local = LocalThreadEntityAllocator::new(global);

        // 1. Put some entities in local cache (START_BATCH_SIZE)
        let _ = local.alloc(0);
        assert_eq!(local.next_entities.len(), (START_BATCH_SIZE - 1) as usize);

        // 2. Request a batch larger than cache
        let req_size = (START_BATCH_SIZE * 2) as usize;
        let batch = local.alloc_batch(0, req_size);

        assert_eq!(batch.len(), req_size);

        // Cache should be refreshed/refilled after this operation
        assert!(!local.next_entities.is_empty());
    }

    #[test]
    fn test_batch_pure_local() {
        let global = Arc::new(RwLock::new(Entities::new()));
        let mut local = LocalThreadEntityAllocator::new(global.clone());

        // Fill cache
        let _ = local.alloc(0);

        // Request small batch that fits in remainder
        let small_req = 5;
        // Ensure test config allows this
        assert!((START_BATCH_SIZE as usize) > small_req);

        let batch = local.alloc_batch(0, small_req);
        assert_eq!(batch.len(), small_req);

        // Should not have touched global again
        assert_eq!(global.read().unwrap().len(), START_BATCH_SIZE as usize);
        assert_eq!(
            local.next_entities.len(),
            (START_BATCH_SIZE as usize - 1 - small_req)
        );
    }
}
