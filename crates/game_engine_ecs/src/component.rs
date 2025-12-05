use std::alloc::Layout;
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

pub const MAX_COMPONENTS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComponentId(pub usize);

#[derive(Debug, Clone, Copy)]
pub struct ComponentMeta {
    pub name: &'static str,
    pub layout: Layout,
    pub drop_fn: unsafe fn(*mut u8),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct ComponentMask {
    // Adjusted for 1024 components (16 u64s)
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

    #[inline]
    pub fn set_id(&mut self, id: ComponentId) {
        self.unsafe_set(id.0);
    }

    #[inline]
    pub fn set_ids(&mut self, ids: &[ComponentId]) {
        for id in ids {
            self.unsafe_set(id.0);
        }
    }

    #[inline]
    pub const fn has_id(&self, id: ComponentId) -> bool {
        self.has(id.0)
    }

    #[inline]
    pub fn from_ids(ids: &[ComponentId]) -> Self {
        let mut mask = Self::new();
        for id in ids {
            mask.unsafe_set(id.0);
        }
        mask
    }

    pub fn to_ids(&self) -> Vec<ComponentId> {
        let mut ids = Vec::with_capacity(8); // Small optimization
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
            panic!("Index {} out of bounds", index);
        }
        self.unsafe_set(index);
    }

    #[inline(always)]
    pub(crate) const fn unsafe_set(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        self.bits[word] |= 1 << bit;
    }

    #[inline(always)]
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

// --- ID GENERATION LOGIC ---

static NEXT_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Global allocator for component IDs
pub fn allocate_component_id<T: 'static>() -> ComponentId {
    let id = NEXT_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let comp_id = ComponentId(id);
    comp_id
}

pub trait Component: 'static + Send + Sync + Sized + Debug {
    fn meta() -> ComponentMeta {
        let name = std::any::type_name::<Self>();
        let layout = std::alloc::Layout::new::<Self>();

        unsafe fn drop_shim<T>(ptr: *mut u8) {
            std::ptr::drop_in_place(ptr as *mut T);
        }

        ComponentMeta {
            name,
            layout,
            drop_fn: drop_shim::<Self>,
        }
    }

    /// Returns the unique ID for this component type.
    fn get_id() -> ComponentId;
}

// 2. Implement for common Rust types
impl_component!(
    bool,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    char,
    String,
    &'static str
);

pub struct ComponentRegistry {
    // Sparse storage. Index = Global ComponentId.
    components: Vec<Option<ComponentMeta>>,
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentRegistry {
    pub fn new() -> Self {
        ComponentRegistry {
            components: Vec::with_capacity(MAX_COMPONENTS),
        }
    }

    pub fn register<T: Component>(&mut self) -> ComponentId {
        let id = T::get_id();

        if id.0 >= MAX_COMPONENTS {
            panic!("Exceeded maximum number of components: {}", MAX_COMPONENTS);
        }

        // Ensure vector is large enough for this ID
        if id.0 >= self.components.len() {
            self.components.resize(id.0 + 1, None);
        }

        // Register metadata if this is the first time THIS WORLD sees this component
        if self.components[id.0].is_none() {
            self.components[id.0] = Some(T::meta());
        }

        id
    }

    pub fn get_id<T: Component>(&self) -> Option<ComponentId> {
        let id = T::get_id();
        if id.0 < self.components.len() && self.components[id.0].is_some() {
            Some(id)
        } else {
            None
        }
    }

    pub fn get_meta(&self, id: ComponentId) -> Option<&ComponentMeta> {
        if id.0 < self.components.len() {
            self.components[id.0].as_ref()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use game_engine_derive::Component;

    use super::*;

    #[test]
    #[allow(dead_code)]
    fn test_component_registry() {
        let mut registry = ComponentRegistry::new();

        #[derive(Debug)]
        struct Position(f32, f32);

        #[derive(Debug)]
        struct Velocity(f32, f32);

        impl_component!(Position, Velocity);

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
    use std::any::TypeId;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct A(f32);

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct B(f32); // Identical layout to A

    impl_component!(A, B);

    #[test]
    fn debug_ids() {
        println!("\n=== DEBUGGING COMPONENT IDS ===");

        // 1. Check TypeIds
        let type_a = TypeId::of::<A>();
        let type_b = TypeId::of::<B>();
        println!("TypeId A: {:?}", type_a);
        println!("TypeId B: {:?}", type_b);
        assert_ne!(
            type_a, type_b,
            "CRITICAL: TypeIds are identical! Compiler/Hasher issue."
        );

        // 2. Check Static ID Generation (First Pass)
        println!("--- First Access ---");
        let id_a_1 = A::get_id();
        println!("A::get_id() -> {:?}", id_a_1);

        let id_b_1 = B::get_id();
        println!("B::get_id() -> {:?}", id_b_1);

        // 3. Check Static Caching (Second Pass)
        println!("--- Second Access (Cached) ---");
        let id_a_2 = A::get_id();
        println!("A::get_id() -> {:?}", id_a_2);
        assert_eq!(id_a_1, id_a_2, "A ID changed! Static cache is broken.");

        let id_b_2 = B::get_id();
        println!("B::get_id() -> {:?}", id_b_2);
        assert_eq!(id_b_1, id_b_2, "B ID changed! Static cache is broken.");

        println!("=== END DEBUG ===\n");
    }

    #[test]
    fn prove_linker_merging() {
        // Two distinct types with identical layout
        #[derive(Debug)]
        struct A(f32);
        #[derive(Debug)]
        struct B(f32);

        impl_component!(A, B);

        let a = A::get_id();
        let b = B::get_id();

        // In Debug: likely distinct.
        // In Release with LTO: likely merged (a == b).
        // If a == b, the ECS breaks because A and B are treated as the same component.
        if a == b {
            println!("PROOF: Linker merged A and B statics!");
        } else {
            println!("Safe (for now).");
        }
    }
}
