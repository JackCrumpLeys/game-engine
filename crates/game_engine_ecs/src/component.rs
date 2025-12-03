use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;

// Change this single number to scale the engine (64, 128, 256, etc.)
pub const MAX_COMPONENTS: usize = 64;

/// A unique index assigned to a component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComponentId(pub usize);

/// Metadata required to store and serialise a component.
#[derive(Debug, Clone)]
pub struct ComponentMeta {
    pub name: &'static str,
    pub layout: Layout,
}

/// A bitmask that fits the configured max components.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ComponentMask {
    bits: [u64; MAX_COMPONENTS.div_ceil(64)],
}

impl ComponentMask {
    pub const WORD_COUNT: usize = MAX_COMPONENTS.div_ceil(64);
    pub const CAPACITY: usize = MAX_COMPONENTS;

    #[inline(always)]
    pub fn new() -> Self {
        Self {
            bits: [0; Self::WORD_COUNT],
        }
    }

    /// Helper to set bit using ComponentId directly
    #[inline]
    pub fn set_id(&mut self, id: ComponentId) {
        self.set(id.0);
    }

    /// Helper to set multiple bits using Vec<ComponentId>
    #[inline]
    pub fn set_ids(&mut self, ids: &[ComponentId]) {
        for id in ids {
            self.set(id.0);
        }
    }

    /// Helper to check bit using ComponentId directly
    #[inline]
    pub const fn has_id(&self, id: ComponentId) -> bool {
        self.has(id.0)
    }

    /// Helper to create a mask from a slice of ComponentIds
    #[inline]
    pub fn from_ids(ids: &[ComponentId]) -> Self {
        let mut mask = Self::new();
        for id in ids {
            mask.set(id.0);
        }
        mask
    }

    /// Reconstructs all the set component IDs in this mask.
    pub fn to_ids(&self) -> Vec<ComponentId> {
        let mut ids = Vec::new();
        for index in 0..Self::CAPACITY {
            if self.has(index) {
                ids.push(ComponentId(index));
            }
        }
        ids
    }

    #[inline]
    pub fn set(&mut self, index: usize) {
        if index >= Self::CAPACITY {
            panic!(
                "Index {} out of bounds for ComponentMask with capacity {}",
                index,
                Self::CAPACITY
            );
        }
        self.unsafe_set(index);
    }

    pub(crate) const fn unsafe_set(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        self.bits[word] |= 1 << bit;
    }

    #[inline]
    pub const fn has(&self, index: usize) -> bool {
        if index >= Self::CAPACITY {
            return false;
        }
        let word = index / 64;
        let bit = index % 64;
        (self.bits[word] & (1 << bit)) != 0
    }

    #[inline]
    pub fn contains_all(&self, other: &Self) -> bool {
        for i in 0..Self::WORD_COUNT {
            if (self.bits[i] & other.bits[i]) != other.bits[i] {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..Self::WORD_COUNT {
            if (self.bits[i] & other.bits[i]) != 0 {
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn union(&mut self, other: &Self) {
        for i in 0..Self::WORD_COUNT {
            self.bits[i] |= other.bits[i];
        }
    }

    #[inline]
    pub fn intersection(&mut self, other: &Self) {
        for i in 0..Self::WORD_COUNT {
            self.bits[i] &= other.bits[i];
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        for i in 0..Self::WORD_COUNT {
            if self.bits[i] != 0 {
                return false;
            }
        }
        true
    }
}

// Custom Debug implementation to print set bits like [0, 3, 5]
impl fmt::Debug for ComponentMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for i in 0..Self::CAPACITY {
            if self.has(i) {
                list.entry(&i);
            }
        }
        list.finish()
    }
}

pub trait Component: 'static + Send + Sync + Sized {}
impl<T: 'static + Send + Sync + Sized> Component for T {}

pub struct ComponentRegistry {
    type_to_id: HashMap<TypeId, ComponentId>,
    components: Vec<ComponentMeta>,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    pub fn new() -> Self {
        ComponentRegistry {
            type_to_id: HashMap::new(),
            // Pre-allocating prevents re-allocations during startup
            components: Vec::with_capacity(ComponentMask::CAPACITY),
        }
    }

    pub fn register<T: Component>(&mut self) -> ComponentId {
        *self.type_to_id.entry(TypeId::of::<T>()).or_insert_with(|| {
            let name = std::any::type_name::<T>();
            let layout = std::alloc::Layout::new::<T>();

            if self.components.len() >= ComponentMask::CAPACITY {
                panic!(
                    "Component limit reached! Max: {}. Increase MAX_COMPONENTS constant.",
                    ComponentMask::CAPACITY
                );
            }

            let id = ComponentId(self.components.len());
            let meta = ComponentMeta { name, layout };
            self.components.push(meta);
            id
        })
    }

    pub fn get_id<T: Component>(&self) -> Option<ComponentId> {
        self.type_to_id.get(&TypeId::of::<T>()).cloned()
    }

    pub fn get_meta(&self, id: ComponentId) -> Option<&ComponentMeta> {
        self.components.get(id.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(dead_code)]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::new();

        #[derive(Debug)]
        struct Position(f32, f32);

        #[derive(Debug)]
        struct Velocity(f32, f32);

        let pos_id = registry.register::<Position>();
        let vel_id = registry.register::<Velocity>();
        let pos_id2 = registry.register::<Position>();

        assert_eq!(pos_id, pos_id2);
        assert_ne!(pos_id, vel_id);

        let pos_meta = registry.get_meta(pos_id).unwrap();
        assert_eq!(
            pos_meta.name,
            "game_engine_ecs::component::tests::test_component_registry::Position"
        );
        assert_eq!(pos_meta.layout.size(), std::mem::size_of::<Position>());

        let vel_meta = registry.get_meta(vel_id).unwrap();
        assert_eq!(
            vel_meta.name,
            "game_engine_ecs::component::tests::test_component_registry::Velocity"
        );
        assert_eq!(vel_meta.layout.size(), std::mem::size_of::<Velocity>());
    }
    #[test]
    fn test_mask_debug_and_logic() {
        let mut mask = ComponentMask::new();
        mask.set(1);
        mask.set(63); // Edge of first u64

        // Test custom Debug output
        let debug_str = format!("{mask:?}");
        assert_eq!(debug_str, "[1, 63]");

        // Test Helpers
        let id = ComponentId(1);
        assert!(mask.has_id(id));
    }
}
