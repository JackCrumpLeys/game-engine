use std::fmt;

/// A unique identifier for an object in the World.
/// Consists of a 32-bit index and a 32-bit generation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)] // Enforce layout for raw memory saving
pub struct Entity {
    id: u64,
}

impl Entity {
    // Bit masks
    const INDEX_MASK: u64 = 0xFFFF_FFFF;
    const GENERATION_MASK: u64 = 0xFFFF_FFFF_0000_0000;
    const GENERATION_SHIFT: u64 = 32;

    pub fn new(index: u32, generation: u32) -> Self {
        let encoded = (index as u64) | ((generation as u64) << Self::GENERATION_SHIFT);
        Self { id: encoded }
    }

    pub fn index(&self) -> u32 {
        (self.id & Self::INDEX_MASK) as u32
    }

    pub fn generation(&self) -> u32 {
        ((self.id & Self::GENERATION_MASK) >> Self::GENERATION_SHIFT) as u32
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index(), self.generation())
    }
}

/// Manages the allocation and recycling of Entity IDs.
pub struct Entities {
    /// Stores the generation of the entity currently living at this index.
    /// If the entity is dead, this stores the next generation to use.
    generations: Vec<u32>,

    /// A stack of indices that have been freed and are ready for reuse.
    free_indices: Vec<u32>,

    /// Stores weather an entity is initialized or not
    is_initialized: Vec<bool>,

    /// The total number of live entities.
    len: usize,
}

impl Default for Entities {
    fn default() -> Self {
        Self::new()
    }
}

impl Entities {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_indices: Vec::new(),
            is_initialized: Vec::new(),
            len: 0,
        }
    }

    /// Allocates a new entity and returns its handle.
    /// This does not imply initialization as it only reserves the ID.
    pub fn alloc(&mut self) -> Entity {
        let index = if let Some(i) = self.free_indices.pop() {
            i // We have a free index to reuse
        } else {
            // We know that we have no reusable indices, so we need to grow the generations array
            let i = self.generations.len() as u32;
            self.generations.push(0); // Generation starts at 0
            self.is_initialized.push(false); // Not initialized yet
            i
        };

        self.len += 1;
        Entity::new(index, self.generations[index as usize])
    }

    /// Returns true if the entity was successfully freed.
    /// Returns false if it was already dead or invalid.
    pub fn free(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }

        let index = entity.index() as usize;

        // Increment generation so any dangling pointers to the old entity become invalid.
        self.generations[index] += 1;
        // Mark as uninitialized
        self.is_initialized[index] = false;

        // We can now reuse this index.
        self.free_indices.push(entity.index());
        self.len -= 1;
        true
    }

    /// Reserves a specific number of entities and returns them as a Vec.
    /// This is useful for batch-filling local thread allocators.
    pub fn alloc_batch(&mut self, count: usize) -> Vec<Entity> {
        debug_assert!(count > 0, "alloc_batch called with count 0");
        debug_assert!(
            count <= u32::MAX as usize,
            "alloc_batch called with count exceeding u32::MAX"
        );

        let mut reserved = Vec::with_capacity(count);

        // 1. First, try to satisfy the request using recycled indices
        while reserved.len() < count {
            if let Some(index) = self.free_indices.pop() {
                // The generation at this index was already incremented during free()
                // so it is ready to be used.
                let generation = self.generations[index as usize];
                reserved.push(Entity::new(index, generation));
            } else {
                break; // No more recycled indices available
            }
        }

        // 2. If we still need more, grow the contiguous generation array
        let needed = count - reserved.len();
        if needed > 0 {
            let start_index = self.generations.len() as u32;

            // Extend the generations vector with 0s for the new entities
            self.generations.extend(std::iter::repeat_n(0, needed));
            // Extend the is_initialized vector with false for the new entities
            self.is_initialized
                .extend(std::iter::repeat_n(false, needed));

            // Generate the new Entity IDs
            for i in 0..needed {
                reserved.push(Entity::new(start_index + i as u32, 0));
            }
        }

        self.len += count;
        reserved
    }

    /// Mark the given entity as initialized.
    /// This means it has a set of components associated with it.
    pub fn initialize(&mut self, entity: Entity) {
        let index = entity.index() as usize;
        debug_assert!(
            self.is_alive(entity),
            "set_initialized called with dead entity"
        );
        self.is_initialized[index] = true;
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let index = entity.index() as usize;
        // 1. Check bounds
        // 2. Check if generations match
        index < self.generations.len() && self.generations[index] == entity.generation()
    }

    pub fn is_initialized(&self, entity: Entity) -> bool {
        let index = entity.index() as usize;
        debug_assert!(
            self.is_alive(entity),
            "is_initialized called with dead entity"
        );
        self.is_initialized[index]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod entity_tests {
    use super::*;

    #[test]
    fn test_spawn_and_kill() {
        let mut entities = Entities::new();

        let e1 = entities.alloc();
        assert_eq!(e1.index(), 0);
        assert_eq!(e1.generation(), 0);

        let e2 = entities.alloc();
        assert_eq!(e2.index(), 1);
        assert_eq!(e2.generation(), 0);

        assert!(entities.is_alive(e1));
        entities.free(e1);
        assert!(!entities.is_alive(e1));

        // Reuse the slot
        let e3 = entities.alloc();
        assert_eq!(e3.index(), 0); // Should reuse index 0
        assert_eq!(e3.generation(), 1); // But generation should be higher

        // The old handle e1 should technically point to index 0,
        // but the generation 0 != 1, so it is invalid.
        assert!(!entities.is_alive(e1));
        assert!(entities.is_alive(e3));
    }
    

    #[test]
    fn test_entity_bit_masking() {
        let index = 12345;
        let generation = 54321;
        let entity = Entity::new(index, generation);

        assert_eq!(entity.index(), index);
        assert_eq!(entity.generation(), generation);

        // Ensure the upper/lower bits don't bleed into each other
        let max_idx = u32::MAX;
        let e_max = Entity::new(max_idx, 0);
        assert_eq!(e_max.index(), u32::MAX);
        assert_eq!(e_max.generation(), 0);
    }

    #[test]
    fn test_alloc_free_recycle() {
        let mut entities = Entities::new();

        // 1. Allocate A
        let e1 = entities.alloc();
        assert_eq!(e1.index(), 0);
        assert_eq!(e1.generation(), 0);

        // Initialize it (simulate world spawn)
        entities.initialize(e1);
        assert!(entities.is_alive(e1));
        assert!(entities.is_initialized(e1));

        // 2. Free A
        assert!(entities.free(e1));

        // e1 should be dead
        assert!(!entities.is_alive(e1));
        // Double free should fail
        assert!(!entities.free(e1));

        // 3. Allocate B (Should reuse A's index, but Gen 1)
        let e2 = entities.alloc();
        assert_eq!(e2.index(), 0); // Reused
        assert_eq!(e2.generation(), 1); // Incremented

        // 4. e1 should definitely not be considered e2
        assert!(!entities.is_alive(e1));
        assert!(entities.is_alive(e2));
    }

    #[test]
    fn test_alloc_batch_contiguous() {
        let mut entities = Entities::new();

        // Request 100 entities
        let batch = entities.alloc_batch(100);

        assert_eq!(batch.len(), 100);
        assert_eq!(entities.len(), 100);

        // Since it's fresh, they should be sequential 0..100
        for (i, e) in batch.iter().enumerate() {
            assert_eq!(e.index() as usize, i);
            assert_eq!(e.generation(), 0);
        }
    }

    #[test]
    fn test_alloc_batch_fragmented() {
        let mut entities = Entities::new();

        // Alloc 3
        let e0 = entities.alloc();
        let e1 = entities.alloc();
        let e2 = entities.alloc();

        // Free 1 and 2 (put into reuse queue)
        entities.free(e1);
        entities.free(e2);

        // We expect the reuse queue to look like [1, 2] (or [2, 1] depending on impl)
        // plus we need 2 more new ones.

        let batch = entities.alloc_batch(4);

        // It should have reused index 1 and 2
        let reused_count = batch
            .iter()
            .filter(|e| e.index() == 1 || e.index() == 2)
            .count();
        assert_eq!(reused_count, 2);

        // Generations for 1 and 2 should be 1
        for e in &batch {
            if e.index() == 1 || e.index() == 2 {
                assert_eq!(e.generation(), 1);
            } else {
                // The new ones (index 3 and 4)
                assert_eq!(e.generation(), 0);
            }
        }

        // e0 (index 0) was never touched
        assert!(entities.is_alive(e0));
    }

    #[test]
    fn test_is_initialized_check() {
        let mut entities = Entities::new();
        let e = entities.alloc();

        // Alive (reserved) but not initialized (no data)
        assert!(entities.is_alive(e));
        assert!(!entities.is_initialized(e));

        entities.initialize(e);
        assert!(entities.is_initialized(e));

        entities.free(e);

        let e2 = entities.alloc();
        assert!(!entities.is_initialized(e2)); // Reset logic works
    }
}
