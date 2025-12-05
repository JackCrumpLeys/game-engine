use smallvec::SmallVec;

use crate::archetype::{Archetype, ArchetypeId};
use crate::bundle::{Bundle, ManyRowBundle};
use crate::component::{ComponentId, ComponentMask, ComponentRegistry};
use crate::entity::{Entities, Entity};
use crate::query::{Filter, QueryInner, QueryToken};
use crate::resource::Resources;
use crate::storage::TypeErasedSequence;
use crate::thread_entity_allocator::LocalThreadEntityAllocator;
use crate::threading::WorldThreadLocalStore;
use std::collections::HashMap;
use std::ops::{Deref, Index, IndexMut};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Clone, Copy, Debug)]
pub struct EntityLocation {
    archetype_id: ArchetypeId,
    row: usize,
}

impl EntityLocation {
    pub fn archetype_id(&self) -> ArchetypeId {
        self.archetype_id
    }

    pub fn row(&self) -> usize {
        self.row
    }
}

pub struct World {
    pub(crate) entities: Arc<RwLock<Entities>>, // Every thread will spin up a local allocator that syncs with
    // this
    thread_local: ThreadLocalWorldData,
    pub registry: ComponentRegistry,
    pub(crate) archetypes: ArchetypeStore,
    // Maps Entity Index -> Location
    entity_index: Vec<Option<EntityLocation>>,
    archetype_index: HashMap<ComponentMask, ArchetypeId>,
    resources: Resources,
    current_tick: u32,
    insert_buffer: EntityInsertBuffer,
}

struct ThreadLocalWorldData {
    entity_allocators: WorldThreadLocalStore<LocalThreadEntityAllocator>,
}

impl ThreadLocalWorldData {
    fn new() -> Self {
        Self {
            entity_allocators: WorldThreadLocalStore::new(),
        }
    }
}

#[derive(Default)]
struct EntityInsertBuffer {
    /// The entity ids to add, in order of
    ///  * ComponentMask
    ///  * Row
    ids: Vec<Vec<Entity>>,
    /// Mask per archetype to add, in sorted order (use binary search to find and insert)
    masks: Vec<ComponentMask>,
    /// Cache sorted component ids per archetype
    comp_ids: Vec<Vec<ComponentId>>,
    /// The columns for each archetype's buffered entities
    /// It is in the order of:
    ///  * ComponentMask
    ///  * ComponentId
    bundle_cols: Vec<Vec<TypeErasedSequence>>,
}

impl EntityInsertBuffer {
    fn clear(&mut self) {
        self.ids.clear();
        self.masks.clear();
        self.bundle_cols.clear();
    }

    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn get_or_create_archetype_buffer(&mut self, mask: ComponentMask) -> usize {
        match self.masks.binary_search(&mask) {
            Ok(index) => index,
            Err(index) => {
                self.masks.insert(index, mask);
                self.ids.insert(index, Vec::new());
                self.bundle_cols.insert(index, Vec::new());
                self.comp_ids.insert(index, Vec::new());
                index
            }
        }
    }

    fn insert_bundle<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
        component_registry: &mut ComponentRegistry,
    ) {
        let mask = B::mask();

        let arch_index = self.get_or_create_archetype_buffer(mask);
        let bundle_cols = &mut self.bundle_cols[arch_index];

        let mut component_ids = &self.comp_ids[arch_index];

        if component_ids.is_empty() {
            self.comp_ids[arch_index] = B::component_ids(component_registry);
            self.comp_ids[arch_index].sort_unstable();

            component_ids = &self.comp_ids[arch_index];
        }

        if bundle_cols.is_empty() {
            for &comp_id in component_ids {
                let meta = component_registry
                    .get_meta(comp_id)
                    .expect("ComponentId not found in registry");
                bundle_cols.push(TypeErasedSequence::new(meta));
            }
        }

        // Ensure components are registered so metadata exists

        unsafe {
            // Note: bundle_cols elements match comp_ids order perfectly because both are sorted by ID
            bundle.put(
                &mut bundle_cols.iter_mut().collect::<Vec<_>>(),
                &component_ids,
            );
        }

        self.ids[arch_index].push(entity);
    }
}

pub struct ArchetypeStore(Vec<Archetype>);

/// By not using a Vec directly, we can later change the storage strategy
/// We also define only 2 mutating methods: push and index_mut
impl Default for ArchetypeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchetypeStore {
    pub fn new() -> Self {
        ArchetypeStore(Vec::new())
    }

    pub fn push(&mut self, archetype: Archetype) {
        self.0.push(archetype);
    }

    /// Gets an iterator over archetypes added since the given ArchetypeId.
    pub fn since(&self, last_seen: ArchetypeId) -> impl Iterator<Item = &Archetype> {
        self.0.iter().skip(last_seen.0)
    }
}

impl Deref for ArchetypeStore {
    type Target = Vec<Archetype>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Index<ArchetypeId> for ArchetypeStore {
    type Output = Archetype;

    fn index(&self, index: ArchetypeId) -> &Self::Output {
        self.0.get(index.0).expect("ArchetypeId out of bounds")
    }
}

impl IndexMut<ArchetypeId> for ArchetypeStore {
    fn index_mut(&mut self, index: ArchetypeId) -> &mut Self::Output {
        self.0.get_mut(index.0).expect("ArchetypeId out of bounds")
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        World {
            entities: Arc::new(RwLock::new(Entities::new())),
            thread_local: ThreadLocalWorldData::new(),
            registry: ComponentRegistry::new(),
            archetypes: ArchetypeStore::new(),
            entity_index: Vec::new(),
            archetype_index: HashMap::new(),
            resources: Resources::new(),
            current_tick: 1, // 0 can be special
            insert_buffer: EntityInsertBuffer::default(),
        }
    }

    /// Get your thread-local entity allocator.
    pub fn entity_allocator(&self) -> &mut LocalThreadEntityAllocator {
        self.thread_local.entity_allocators.get(&self)
    }

    /// Read entities storage.
    ///
    /// panics if the lock is already held by the current thread.
    pub fn entities(&self) -> std::sync::RwLockReadGuard<'_, Entities> {
        self.entities.read().unwrap()
    }

    /// Spawns a new entity with the given bundle of components.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let ent = self.entity_allocator().alloc(self.current_tick);

        self.spawn_with_id(ent, bundle);

        ent
    }

    /// Spawn deferred: Spawns an entity with the given bundle, deferring the actual insertion
    /// until later.
    pub fn spawn_deferred<B: Bundle>(&mut self, bundle: B) -> Entity {
        let ent = self.entity_allocator().alloc(self.current_tick);

        self.insert_buffer
            .insert_bundle(ent, bundle, &mut self.registry);

        ent
    }

    pub fn spawn_with_id<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        // Register components ensuring metadata exists
        B::component_ids(&mut self.registry);

        let mask = B::mask();
        let arch_id = self.get_or_create_archetype(mask);
        let arch = &mut self.archetypes[arch_id];
        let row = arch.push_entity(entity);

        // Split borrow manually to satisfy borrow checker
        let arch_ids = &arch.component_ids;
        let arch_cols_storage = &mut arch.columns;

        // Collect mutable references to columns into SmallVec
        let mut arch_cols: SmallVec<&mut TypeErasedSequence, 16> = arch_cols_storage
            .iter_mut()
            .filter_map(|c| c.as_deref_mut())
            .map(|col| {
                col.set_tick(self.current_tick);
                col.push_ticks(1);
                col.inner_mut()
            })
            .collect();

        unsafe {
            bundle.put(&mut arch_cols, arch_ids);
        }

        self.entities.write().unwrap().initialize(entity);

        // Optimistic resize
        let idx = entity.index() as usize;
        if idx >= self.entity_index.len() {
            self.entity_index.resize(idx + 1, None);
        }

        self.entity_index[idx] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
    }

    pub fn spawn_batch<BM, B: Bundle>(&mut self, bundles: BM) -> Vec<Entity>
    where
        BM: ManyRowBundle<Item = B>,
    {
        let b_order = B::component_ids(&mut self.registry);

        let mask = B::mask();
        let real_order = mask.to_ids(); // Sorted
        let arch_id = self.get_or_create_archetype(mask);
        let count = bundles.len();

        let entities = self
            .entity_allocator()
            .alloc_batch(self.current_tick, count);

        let arch = &mut self.archetypes[arch_id];
        let start_row = arch.len();

        let mut entities_guard = self.entities.write().unwrap();
        for (i, &e) in entities.iter().enumerate() {
            let row = start_row + i;
            let idx = e.index() as usize;
            if idx >= self.entity_index.len() {
                self.entity_index.resize(idx + 1, None);
            }
            self.entity_index[idx] = Some(EntityLocation {
                archetype_id: arch_id,
                row,
            });
            entities_guard.initialize(e);
        }

        arch.push_entities(&entities);

        let mut arch_cols: SmallVec<&mut TypeErasedSequence, 16> = arch
            .columns_mut()
            .map(|col| {
                col.set_tick(self.current_tick);
                col.push_ticks(count);
                col.inner_mut()
            })
            .collect();

        let mut ordered_cols: SmallVec<&mut TypeErasedSequence, 16> =
            SmallVec::with_capacity(b_order.len());

        unsafe {
            for id in b_order {
                // 1. Find the index in real_order (which corresponds to arch_cols indices)
                // SAFETY: We assume B::mask() ensures the Bundle components are a subset
                // of the archetype. unwrap_unchecked removes the panic branch.
                let idx = real_order.binary_search(&id).unwrap_unchecked();

                // 2. Get the reference from the source vector without bounds checking.
                // SAFETY: binary_search returned a valid index.
                let col_ref = arch_cols.get_unchecked_mut(idx);

                // 3. Cast the &mut T to *mut T.
                // Pointers are `Copy`, so this bypasses the borrow checker preventing
                // us from moving/copying a mutable ref.
                let ptr: *mut TypeErasedSequence = *col_ref;

                // 4. Turn it back into &mut T and push to the new list.
                // SAFETY: Crucial! This is safe ONLY if `b_order` contains unique IDs.
                // If `b_order` requested the same component twice, we would create
                // two mutable references to the same memory (UB).
                // Assuming standard ECS Bundle rules, IDs are unique.
                ordered_cols.push(&mut *ptr);
            }
        }

        unsafe {
            bundles.put_many(&mut ordered_cols);
        }

        entities
    }

    pub fn flush(&mut self) {
        if self.insert_buffer.is_empty() {
            return;
        }

        let mut buffer = std::mem::take(&mut self.insert_buffer);
        let count = buffer.len();

        for i in 0..count {
            let mask = buffer.masks[i];
            let arch_id = self.get_or_create_archetype(mask);
            let entities = &buffer.ids[i];
            let comp_ids = mask.to_ids();
            let mut src_cols = std::mem::take(&mut buffer.bundle_cols[i]);

            let arch = &mut self.archetypes[arch_id];
            let start_row = arch.len();

            for (src_seq, &comp_id) in src_cols.drain(..).zip(comp_ids.iter()) {
                let dest_col = arch
                    .column_mut(&comp_id)
                    .expect("Archetype created from mask should have matching columns");

                dest_col.set_tick(self.current_tick);
                dest_col.append(src_seq);
            }

            let mut max_idx = 0;
            for &e in entities {
                max_idx = max_idx.max(e.index() as usize);
            }
            if max_idx >= self.entity_index.len() {
                self.entity_index.resize(max_idx + 1, None);
            }

            let mut entities_guard = self.entities.write().unwrap();
            arch.push_entities(entities);

            for (i, &entity) in entities.iter().enumerate() {
                let row = start_row + i;
                self.entity_index[entity.index() as usize] = Some(EntityLocation {
                    archetype_id: arch_id,
                    row,
                });
                entities_guard.initialize(entity);
            }
        }
    }
    /// Spawns an entity with the given bundle, deferring the actual insertion until later.
    pub fn spawn_deferred_with_id<B: Bundle>(&mut self, bundle: B, id: Entity) {
        self.insert_buffer
            .insert_bundle(id, bundle, &mut self.registry);
    }

    pub fn query<Q: QueryToken, F: Filter>(&mut self) -> QueryInner<Q::Persistent, F::Persistent> {
        QueryInner::new::<Q, F>(&mut self.registry)
    }

    pub fn resources_mut(&mut self) -> &mut Resources {
        &mut self.resources
    }

    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    pub fn increment_tick(&mut self) {
        self.current_tick = self.current_tick.wrapping_add(1);
    }

    /// Get the locarion of an entity in the world.
    /// Renurns None if the entity is not alive.
    pub fn entity_location(&self, entity: Entity) -> Option<EntityLocation> {
        if !(self.entities().is_alive(entity) && self.entities().is_initialized(entity)) {
            return None;
        }

        self.entity_index[entity.index() as usize]
    }

    /// Despawns an entity. Returns true if it existed.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities().is_alive(entity) {
            return false;
        }

        if self.entities().is_initialized(entity) {
            let loc = self.entity_index[entity.index() as usize]
                .take()
                .expect("Entity should have a location"); // this should always be Some if the entity
            // is alive

            if let Some(moved) = self.archetypes[loc.archetype_id].swap_remove(loc.row) {
                let ent_loc = self.entity_index[moved.index() as usize]
                    .as_mut()
                    .expect("Moved entity should have a location");
                ent_loc.row = loc.row;
            };

            self.entity_index[entity.index() as usize] = None;
        }

        self.entities.write().unwrap().free(entity);

        true
    }

    fn get_or_create_archetype(&mut self, component_mask: ComponentMask) -> ArchetypeId {
        if let Some(&id) = self.archetype_index.get(&component_mask) {
            return id;
        }

        let id = ArchetypeId(self.archetypes.len());
        let archetype = Archetype::new(id, component_mask, &self.registry);
        self.archetypes.push(archetype);
        self.archetype_index.insert(component_mask, id);
        id
    }

    /// The system tick, increments every time `increment_tick` is called.
    /// Used for change detection.
    pub fn tick(&self) -> u32 {
        self.current_tick
    }
}

#[test]
fn test_world_spawn_despawn() {
    let mut world = World::new();

    // Spawn simple
    let e1 = world.spawn((10u32,));

    // Spawn complex
    let e2 = world.spawn((20u32, 5.0f32));

    // Check internal state (impl specific)
    assert!(world.entities().is_alive(e1));
    assert!(world.entities().is_alive(e2));

    // Despawn
    assert!(world.despawn(e1));
    assert!(!world.entities().is_alive(e1));

    // Despawn again should fail
    assert!(!world.despawn(e1));
}
#[cfg(test)]
mod batch_tests {
    use crate::prelude::*;
    use crate::world::World;

    #[derive(Debug, PartialEq, Default, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Default, Clone, Copy)]
    struct Vel {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Default, Clone, Copy)]
    struct Health(f32);

    impl_component!(Pos, Vel, Health);

    #[test]
    fn test_spawn_batch_len_and_ids() {
        let mut world = World::new();
        let count = 10_000;

        let mut batch = Vec::with_capacity(count);
        for i in 0..count {
            batch.push((
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Vel {
                    x: i as f32,
                    y: i as f32,
                },
            ));
        }

        let entities = world.spawn_batch(batch);

        // Assert: Return value is exact
        assert_eq!(entities.len(), count);

        // Assert: Global state covers the count (>= because thread allocator might buffer a few extra from previous ops)
        assert!(world.entities().len() >= count);

        // Assert: Data Integrity via Query (The real truth source)
        let mut query = crate::query::QueryState::<(&Pos, &Vel)>::new(&mut world.registry);
        assert_eq!(query.iter(&mut world).count(), count);
    }

    #[test]
    fn test_spawn_batch_fragmentation() {
        let mut world = World::new();

        // 1. Spawn single entity (Allocates 16 globally, caches 15)
        let e1 = world.spawn((Health(100.0),));

        // 2. Spawn batch 100
        // (Uses the 15 cached, allocates 85 globally. Cache empty.)
        let batch_size = 100;
        let mut batch = Vec::new();
        for i in 0..batch_size {
            batch.push((
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Vel::default(),
            ));
        }

        // 3. Spawn single (Allocates 16 globally, caches 15)
        let e2 = world.spawn((Health(50.0),));

        world.spawn_batch(batch);

        // Total User Entities: 1 + 100 + 1 = 102
        // Total Global IDs: 16 (first alloc) + 85 (batch remainder) + 16 (last alloc) = 117

        let expected_user_count = batch_size + 2;

        // Check actual live entities in query
        let mut q_all = crate::query::QueryState::<Entity>::new(&mut world.registry);
        assert_eq!(q_all.iter(&mut world).count(), expected_user_count);

        // Check Entity Allocator State
        // We assert >= because the allocator holds reserved IDs
        assert!(world.entities().len() >= expected_user_count);

        // Data Checks...
        let mut q_health = crate::query::QueryState::<&Health>::new(&mut world.registry);
        assert_eq!(q_health.get(&mut world, e1).unwrap().0, 100.0);
        assert_eq!(q_health.get(&mut world, e2).unwrap().0, 50.0);
    }

    #[test]
    fn test_spawn_batch_zst() {
        // Zero Sized Types often trip up pointer arithmetic if not careful
        #[derive(Clone, Debug)]
        struct Marker;

        impl_component!(Marker);

        let mut world = World::new();
        let count = 50;
        let batch = vec![(Marker,); count];

        let entities = world.spawn_batch(batch);

        assert_eq!(entities.len(), count);

        let mut query = crate::query::QueryState::<&Marker>::new(&mut world.registry);
        assert_eq!(query.iter(&mut world).count(), count);
    }

    #[test]
    fn test_spawn_batch_empty() {
        let mut world = World::new();
        let batch: Vec<(Pos,)> = Vec::new();

        let entities = world.spawn_batch(batch);

        assert!(entities.is_empty());
        assert_eq!(world.entities().len(), 0);
    }

    #[test]
    fn test_spawn_batch_component_ordering() {
        // Tests that column mapping works even if Bundle order != Component ID order
        let mut world = World::new();

        // We rely on ComponentRegistry to assign IDs.
        // We purposely construct a bundle where we might expect ID mismatch if sorted incorrectly.
        // But since we can't force IDs easily, we trust the logic:
        // Bundle: (Vel, Pos) -> IDs might be [1, 0]
        // Archetype: [0, 1]

        let mut batch = Vec::new();
        for i in 0..10 {
            // Note: Vel is first here
            batch.push((
                Vel {
                    x: i as f32,
                    y: 0.0,
                },
                Pos {
                    x: 100.0,
                    y: i as f32,
                },
            ));
        }

        world.spawn_batch(batch);

        let mut query = crate::query::QueryState::<(&Pos, &Vel)>::new(&mut world.registry);
        for (pos, vel) in query.iter(&mut world) {
            // Pos.x should be 100 (from 2nd slot in tuple)
            assert_eq!(pos.x, 100.0);
            // Vel.x should be index (from 1st slot in tuple)
            assert_eq!(vel.x, pos.y);
        }
    }
}
