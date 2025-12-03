use std::{fmt, iter::repeat_n};

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
            len: 0,
        }
    }

    /// Allocates a new entity and returns its handle.
    pub fn alloc(&mut self) -> Entity {
        let index = if let Some(i) = self.free_indices.pop() {
            i // We have a free index to reuse
        } else {
            // We know that we have no reusable indices, so we need to grow the generations array
            let i = self.generations.len() as u32;
            self.generations.push(0); // Generation starts at 0
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

        // We can now reuse this index.
        self.free_indices.push(entity.index());
        self.len -= 1;
        true
    }

    /// Reserves a specific number of entities and returns them as a Vec.
    /// This is useful for batch-filling local thread allocators.
    pub fn alloc_batch(&mut self, count: usize) -> Vec<Entity> {
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
            self.generations.extend(repeat_n(0, needed));

            // Generate the new Entity IDs
            for i in 0..needed {
                reserved.push(Entity::new(start_index + i as u32, 0));
            }
        }

        self.len += count;
        reserved
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        let index = entity.index() as usize;
        // 1. Check bounds
        // 2. Check if generations match
        index < self.generations.len() && self.generations[index] == entity.generation()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
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
}
