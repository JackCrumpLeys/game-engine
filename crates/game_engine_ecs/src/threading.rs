use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::world::World;

pub const MAX_ECS_THREADS: usize = 1028; // Far more than enough for most use cases

static THREAD_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_ID: usize = THREAD_COUNTER.fetch_add(1, Ordering::Relaxed);
}

/// Returns a unique, dense index for the current OS thread.
/// Returns None if we have exceeded MAX_ECS_THREADS.
///
/// This can be used to index into per-thread data stored in the World,
pub fn ecs_thread_id() -> Option<usize> {
    let id = THREAD_ID.with(|id| *id);
    if id < MAX_ECS_THREADS { Some(id) } else { None }
}

pub trait FromWorldThread {
    /// Get default data for a new thread with the given thread ID and world reference.
    fn new_thread_local(thread_id: usize, world: &World) -> Self;
}

#[repr(align(64))] // Align to cache line size to avoid false sharing.
struct CachePadded<T> {
    inner: UnsafeCell<T>,
}

impl<T> CachePadded<T> {
    fn new(t: T) -> Self {
        Self {
            inner: UnsafeCell::new(t),
        }
    }
}

/// Some container owned by a world that holds per-thread data for ECS threads.
pub struct WorldThreadLocalStore<T> {
    slots: Vec<CachePadded<Option<T>>>, // length == MAX_ECS_THREADS
}

// SAFETY:
// We implement Sync because the `get` method guarantees that
// Thread N only accesses Slot N.
// We require T to be Send because the World (and this store) might be dropped
// on a different thread than the one that created the data.
unsafe impl<T: Send> Sync for WorldThreadLocalStore<T> {}
unsafe impl<T: Send> Send for WorldThreadLocalStore<T> {}

impl<T: FromWorldThread> Default for WorldThreadLocalStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FromWorldThread> WorldThreadLocalStore<T> {
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(MAX_ECS_THREADS);
        for _ in 0..MAX_ECS_THREADS {
            slots.push(CachePadded::new(None));
        }
        Self { slots }
    }

    /// Access the thread-local data.
    /// If it doesn't exist for this thread, it is created using the World.
    ///
    /// panics if we are using too many threads (exceeding MAX_ECS_THREADS).
    ///
    /// # Safety / Clippy
    /// We suppress `mut_from_ref` because this acts like a Lock/RefCell.
    /// The safety is guaranteed by `ecs_thread_id()` returning unique indices per thread.
    #[allow(clippy::mut_from_ref)]
    pub fn get(&self, world: &World) -> &mut T {
        let thread_id = ecs_thread_id().expect("Too many threads accessing ECS");

        // SAFETY: ecs_thread_id() ensures this index is unique to this thread.
        // No other thread can be accessing `slots[thread_id]` right now.
        let cell = &self.slots[thread_id].inner;
        let slot = unsafe { &mut *cell.get() };

        // Lazy Init
        if slot.is_none() {
            *slot = Some(T::new_thread_local(thread_id, world));
        }

        slot.as_mut().unwrap()
    }

    /// Access the thread-local data immutably.
    /// If it doesn't exist for this thread, it is created using the World.
    ///
    /// panics if we are using too many threads (exceeding MAX_ECS_THREADS).
    pub fn get_ref(&self, world: &World) -> &T {
        let thread_id = ecs_thread_id().expect("Too many threads accessing ECS");

        // SAFETY: ecs_thread_id() ensures this index is unique to this thread.
        // No other thread can be accessing `slots[thread_id]` right now.
        let cell = &self.slots[thread_id].inner;
        let slot = unsafe { &mut *cell.get() };

        // Lazy Init
        if slot.is_none() {
            *slot = Some(T::new_thread_local(thread_id, world));
        }

        slot.as_ref().unwrap()
    }

    /// Iterates over all active slots.
    /// Useful for flushing command buffers or merging results.
    ///
    /// # Safety
    /// This requires `&mut self`, ensuring no threads are currently running
    /// (e.g., at a sync point in the schedule).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut().filter_map(|padded_slot| {
            // SAFETY: We have &mut self, so no other threads can be accessing these slots.
            let slot = unsafe { &mut *padded_slot.inner.get() };
            slot.as_mut()
        })
    }
}

#[cfg(test)]
mod threading_tests {
    use super::*;
    use crate::world::World;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    // ========================================================================
    // MOCKS
    // ========================================================================

    struct MockThreadData {
        thread_id: usize,
        value: u32,
    }

    impl FromWorldThread for MockThreadData {
        fn new_thread_local(thread_id: usize, _world: &World) -> Self {
            Self {
                thread_id,
                value: 0,
            }
        }
    }

    // ========================================================================
    // TESTS
    // ========================================================================

    #[test]
    fn test_thread_id_uniqueness() {
        const THREAD_COUNT: usize = 10;
        let ids = Arc::new(Mutex::new(HashSet::new()));
        let mut handles = Vec::new();

        // 1. Spawn threads and collect their ecs_thread_id
        for _ in 0..THREAD_COUNT {
            let ids_clone = ids.clone();
            handles.push(thread::spawn(move || {
                let id = ecs_thread_id().expect("Should not run out of thread IDs");
                ids_clone.lock().unwrap().insert(id);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // 2. Verify all IDs were unique
        let set = ids.lock().unwrap();
        assert_eq!(set.len(), THREAD_COUNT);
    }

    #[test]
    fn test_store_lazy_init_single_thread() {
        let world = World::new();
        let store = WorldThreadLocalStore::<MockThreadData>::new();

        // 1. Access first time -> Init
        {
            let data = store.get(&world);
            assert_eq!(data.value, 0); // Default from new_thread_local
            data.value = 42;
        }

        // 2. Access second time -> Persistence
        {
            let data = store.get(&world);
            assert_eq!(data.value, 42); // Should persist
        }
    }

    #[test]
    fn test_store_concurrent_access_and_isolation() {
        const THREAD_COUNT: usize = 8;

        // The store is Sync, so we wrap in Arc to share across threads
        let store = Arc::new(WorldThreadLocalStore::<MockThreadData>::new());
        let barrier = Arc::new(Barrier::new(THREAD_COUNT));

        let mut handles = Vec::new();

        for i in 0..THREAD_COUNT {
            let store_clone = store.clone();
            let barrier_clone = barrier.clone();

            handles.push(thread::spawn(move || {
                // We create a dummy world here just to satisfy the signature.
                // In a real engine, the Scheduler provides the world access.
                let world = World::new();

                // 1. Initialize data for this thread
                {
                    let data = store_clone.get(&world);
                    // Verify the Mock factory got the correct ID logic
                    // (Note: we can't strictly assert data.thread_id == i because OS thread scheduling is random,
                    // but we can ensure data.thread_id matches ecs_thread_id())
                    assert_eq!(data.thread_id, ecs_thread_id().unwrap());

                    // Set a unique value based on loop index to verify isolation
                    data.value = (i as u32 + 1) * 100;
                }

                // Wait for all threads to initialize and write
                barrier_clone.wait();

                // 2. Verify data persisted and wasn't overwritten by other threads
                {
                    let data = store_clone.get(&world);
                    assert_eq!(data.value, (i as u32 + 1) * 100);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_iteration_and_aggregation() {
        // This simulates "flushing" thread local buffers back to the main world
        const THREAD_COUNT: usize = 4;
        let store = WorldThreadLocalStore::<MockThreadData>::new();

        // Simulate parallel execution using scoped threads or just simple spawn/join.
        // Since `WorldThreadLocalStore` is internal to the World usually,
        // testing this requires wrapping it in Arc for the writing phase.

        let shared_store = Arc::new(store);
        let mut handles = Vec::new();

        for _ in 0..THREAD_COUNT {
            let s = shared_store.clone();
            handles.push(thread::spawn(move || {
                let world = World::new();
                let data = s.get(&world);
                data.value = 1; // Every thread adds 1
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Now we need to get mutable access back to iterate.
        // ARC::try_unwrap is the cleanest way in a test to convert Arc<T> back to T
        // provided all threads are joined.
        let mut store = Arc::try_unwrap(shared_store)
            .ok()
            .expect("Arc should be free");

        // Iterate and sum
        let sum: u32 = store.iter_mut().map(|d| d.value).sum();

        // Note: iter_mut only iterates INITIALIZED slots.
        // Since we spawned 4 threads, we expect sum to be 4.
        // We also expect the main thread (running the test) might NOT have initialized a slot yet,
        // or if it did, it's 0.

        assert_eq!(sum, THREAD_COUNT as u32);
    }

    #[test]
    fn test_max_threads_limit() {
        // This checks ecs_thread_id logic, though we can't easily spawn 1024 threads in a unit test
        // without slowing things down. We verify the ID is consistent.
        let id_1 = ecs_thread_id();
        let id_2 = ecs_thread_id();
        assert_eq!(id_1, id_2);
        assert!(id_1.unwrap() < crate::threading::MAX_ECS_THREADS);
    }
}
