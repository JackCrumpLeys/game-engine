use game_engine_shaders_types::packet::NUM_SHAPE_KINDS;

pub type StackId = u32;

pub struct GpuSlotAllocator {
    /// High-water mark for allocated stack IDs (= GPU entity head count)
    pub stack_count: u32,
    /// Stack IDs that were freed and can be reused
    pub free_stack_ids: Vec<StackId>,

    /// High-water mark per shape kind (number of allocated slots in primitive buffers)
    pub max_slots: Vec<u32>,
    /// Free-lists per shape kind for recycling slots
    pub free_lists: Vec<Vec<u32>>,

    /// Free lists for op_pool ranges grouped by length (up to length 16)
    pub free_op_pool: Vec<Vec<u32>>,

    /// Per-channel dirty slot indices for incremental GPU upload
    pub dirty_slots: Vec<Vec<u32>>,
    /// Stack IDs with dirty entity heads (e.g. position changed)
    pub dirty_entity_heads: Vec<StackId>,
}

impl Default for GpuSlotAllocator {
    fn default() -> Self {
        Self {
            stack_count: 0,
            free_stack_ids: Vec::new(),
            max_slots: vec![0; NUM_SHAPE_KINDS],
            free_lists: (0..NUM_SHAPE_KINDS).map(|_| Vec::new()).collect(),
            free_op_pool: (0..16).map(|_| Vec::new()).collect(),
            dirty_slots: (0..NUM_SHAPE_KINDS).map(|_| Vec::new()).collect(),
            dirty_entity_heads: Vec::new(),
        }
    }
}

impl GpuSlotAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates a StackId (index into the GpuEntityHead buffer).
    /// Reuses IDs from the free list if available.
    pub fn alloc_stack_id(&mut self) -> StackId {
        if let Some(id) = self.free_stack_ids.pop() {
            id
        } else {
            let id = self.stack_count;
            self.stack_count += 1;
            id
        }
    }

    /// Returns a StackId to the pool for reuse.
    pub fn free_stack_id(&mut self, id: StackId) {
        self.free_stack_ids.push(id);
    }

    /// Allocates a slot in a specific primitive channel (e.g., Circle or Rect).
    pub fn alloc_slot(&mut self, shape_kind: u32) -> u32 {
        let kind = shape_kind as usize;
        if let Some(slot) = self.free_lists[kind].pop() {
            slot
        } else {
            let slot = self.max_slots[kind];
            self.max_slots[kind] += 1;
            slot
        }
    }

    /// Bulk allocates multiple StackIds.
    pub fn alloc_stack_ids(&mut self, count: u32) -> Vec<StackId> {
        let mut ids = Vec::with_capacity(count as usize);
        for _ in 0..count {
            ids.push(self.alloc_stack_id());
        }
        ids
    }

    /// Bulk allocates multiple slots for a specific shape kind.
    pub fn alloc_slots(&mut self, shape_kind: u32, count: u32) -> Vec<u32> {
        let mut slots = Vec::with_capacity(count as usize);
        for _ in 0..count {
            slots.push(self.alloc_slot(shape_kind));
        }
        slots
    }

    /// Returns a primitive slot to the pool for reuse.
    pub fn free_slot(&mut self, shape_kind: u32, slot: u32) {
        self.free_lists[shape_kind as usize].push(slot);
    }

    /// Allocates a range of op_pool if a previously freed one of the same size exists.
    pub fn alloc_op_pool_range(&mut self, len: u32) -> Option<u32> {
        let len = len as usize;
        if len < self.free_op_pool.len() {
            self.free_op_pool[len].pop()
        } else {
            None
        }
    }

    /// Recycles a range of op_pool.
    pub fn free_op_pool_range(&mut self, start: u32, len: u32) {
        let len = len as usize;
        if len < self.free_op_pool.len() {
            self.free_op_pool[len].push(start);
        }
    }

    /// Marks a specific slot in a primitive channel as dirty (needs GPU upload).
    pub fn mark_dirty(&mut self, shape_kind: u32, slot: u32) {
        self.dirty_slots[shape_kind as usize].push(slot);
    }

    /// Marks an entity head (stack) as dirty (needs GPU position/offset upload).
    pub fn mark_entity_head_dirty(&mut self, _stack_id: StackId) {
        // No-op. The entire buffer is synced each frame because entity count changes,
        // and identifying dirty spans isn't currently used by the uploader for entity heads.
    }

    /// Drains the dirty tracking lists. Used by the renderer to perform incremental updates.
    pub fn take_dirty(&mut self) -> Vec<Vec<u32>> {
        self.dirty_entity_heads.clear();
        let mut result = (0..NUM_SHAPE_KINDS).map(|_| Vec::new()).collect::<Vec<_>>();
        for (i, slots) in self.dirty_slots.iter_mut().enumerate() {
            if !slots.is_empty() {
                slots.sort_unstable();
                slots.dedup();
                std::mem::swap(&mut result[i], slots);
            }
        }
        result
    }
}
