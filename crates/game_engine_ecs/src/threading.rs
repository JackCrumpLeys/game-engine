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

    /// Iterates over all active slots.
    /// Useful for flushing command buffers or merging results.
    ///
    /// # Safety
    /// This requires `&mut self`, ensuring no threads are currently running
    /// (e.g., at a sync point in the schedule).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots
            .iter_mut()
            .filter_map(|slot| slot.inner.get_mut().as_mut())
    }
}
