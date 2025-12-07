use smallvec::SmallVec;

use crate::archetype::{Archetype, ArchetypeId};
use crate::bundle::{Bundle, ManyRowBundle};
use crate::component::{ComponentId, ComponentMask, ComponentMeta, ComponentRegistry};
use crate::entity::{Entities, Entity};
use crate::query::{Filter, QueryState, QueryToken};
use crate::resource::Resources;
use crate::storage::TypeErasedSequence;
use crate::thread_entity_allocator::LocalThreadEntityAllocator;
use crate::threading::{FromWorldThread, WorldThreadLocalStore};
use std::ops::{Deref, Index, IndexMut};
use std::sync::{Arc, RwLock};

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
    resources: Resources,
    // This is updated between every batch/during a flush
    current_tick: u32,
    /// THis value is incremented every time we have finished running all the systems for a frame
    curent_frame: u32,
}

struct ThreadLocalWorldData {
    entity_allocators: WorldThreadLocalStore<LocalThreadEntityAllocator>,
    insert_buffers: WorldThreadLocalStore<EntityInsertBuffer>,
}

impl ThreadLocalWorldData {
    fn new() -> Self {
        Self {
            entity_allocators: WorldThreadLocalStore::new(),
            insert_buffers: WorldThreadLocalStore::new(),
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
    comp_metas: Vec<Vec<ComponentMeta>>,
    /// The columns for each archetype's buffered entities
    /// It is in the order of:
    ///  * ComponentMask
    ///  * ComponentId
    bundle_cols: Vec<Vec<TypeErasedSequence>>,
}

impl FromWorldThread for EntityInsertBuffer {
    fn new_thread_local(_thread_id: usize, _world: &World) -> Self {
        Self::default()
    }
}

impl EntityInsertBuffer {
    /// Pls call every frame
    /// (after drain_cols)
    fn clear(&mut self) {
        self.ids.clear();
        self.masks.clear();
        self.comp_ids.clear();
        self.comp_metas.clear();
    }

    #[inline(always)]
    fn drain_cols(&mut self) -> Vec<Vec<TypeErasedSequence>> {
        self.bundle_cols.drain(..).collect()
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[inline(always)]
    fn get_or_create_archetype_buffer(&mut self, mask: ComponentMask) -> usize {
        match self.masks.binary_search(&mask) {
            Ok(index) => index,
            Err(index) => {
                self.masks.insert(index, mask);
                self.ids.insert(index, Vec::new());
                self.bundle_cols.insert(index, Vec::new());
                self.comp_ids.insert(index, Vec::new());
                self.comp_metas.insert(index, Vec::new());
                index
            }
        }
    }

    fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let mask = B::mask();

        let arch_index = self.get_or_create_archetype_buffer(mask);

        let bundle_cols = &mut self.bundle_cols[arch_index];
        let component_ids = &mut self.comp_ids[arch_index];

        if component_ids.is_empty() {
            let mut meta: Vec<(ComponentId, ComponentMeta)> = B::component_ids()
                .into_iter()
                .zip(B::component_metas())
                .collect();
            meta.sort_unstable_by_key(|(id, _)| *id);

            for (id, meta) in meta {
                component_ids.push(id);
                bundle_cols.push(TypeErasedSequence::new(&meta));
                self.comp_metas[arch_index].push(meta);
            }
        }

        // Ensure components are registered so metadata exists

        unsafe {
            // Note: bundle_cols elements match comp_ids order perfectly because both are sorted by ID
            bundle.put(
                &mut bundle_cols.iter_mut().collect::<Vec<_>>(),
                component_ids,
            );
        }

        self.ids[arch_index].push(entity);
    }
}

pub struct ArchetypeStore {
    inner: Vec<Archetype>,
    // The Lookup: Sorted by Mask
    lookup: Vec<(ComponentMask, ArchetypeId)>,
}

/// By not using a Vec directly, we can later change the storage strategy
/// We also define only 2 mutating methods: push and index_mut
impl Default for ArchetypeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchetypeStore {
    pub fn new() -> Self {
        ArchetypeStore {
            inner: Vec::new(),
            lookup: Vec::new(),
        }
    }

    pub fn push(&mut self, archetype: Archetype) {
        let mask = archetype.component_mask;
        let id = archetype.id;
        self.inner.push(archetype);

        // Keep lookup sorted
        let idx = self
            .lookup
            .binary_search_by_key(&mask, |(m, _)| *m)
            .unwrap_err();
        self.lookup.insert(idx, (mask, id));
    }

    pub fn get_id(&self, mask: &ComponentMask) -> Option<ArchetypeId> {
        // Binary search is extremely fast on small-medium datasets
        self.lookup
            .binary_search_by_key(mask, |(m, _)| *m)
            .ok()
            .map(|idx| self.lookup[idx].1)
    }

    /// Gets an iterator over archetypes added since the given ArchetypeId.
    pub fn since(&self, last_seen: ArchetypeId) -> impl Iterator<Item = &Archetype> {
        self.inner.iter().skip(last_seen.0)
    }
}

impl Deref for ArchetypeStore {
    type Target = Vec<Archetype>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Index<ArchetypeId> for ArchetypeStore {
    type Output = Archetype;

    fn index(&self, index: ArchetypeId) -> &Self::Output {
        self.inner.get(index.0).expect("ArchetypeId out of bounds")
    }
}

impl IndexMut<ArchetypeId> for ArchetypeStore {
    fn index_mut(&mut self, index: ArchetypeId) -> &mut Self::Output {
        self.inner
            .get_mut(index.0)
            .expect("ArchetypeId out of bounds")
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
            resources: Resources::new(),
            current_tick: 1, // 0 can be special
            curent_frame: 0,
        }
    }

    /// Get your thread-local entity allocator.
    pub fn entity_allocator(&self) -> &mut LocalThreadEntityAllocator {
        self.thread_local.entity_allocators.get(self)
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
    pub fn spawn_deferred<B: Bundle>(&self, bundle: B) -> Entity {
        let ent = self.entity_allocator().alloc(self.current_tick);

        self.spawn_deferred_with_id(ent, bundle);

        ent
    }

    pub fn spawn_with_id<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let real_order = B::component_ids();
        let metas = B::component_metas();
        self.registry.bulk_manual_register(&real_order, &metas);

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
        let b_order = B::component_ids();

        let mask = B::mask();
        let mut real_order = B::component_ids();
        let metas = B::component_metas();
        self.registry.bulk_manual_register(&real_order, &metas);
        real_order.sort_unstable();

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
        let buffers = Vec::from_iter(
            self.thread_local
                .insert_buffers
                .iter_mut()
                .filter(|b| !b.is_empty())
                .map(std::mem::take),
        );
        for mut buffer in buffers {
            let mut cols = buffer.drain_cols();

            for (i, mut src_cols) in cols.drain(..).enumerate() {
                let comp_ids = &mut buffer.comp_ids[i];
                let metas = &buffer.comp_metas[i];
                self.registry.bulk_manual_register(comp_ids, metas);

                let mask = buffer.masks[i];
                let arch_id = self.get_or_create_archetype(mask);
                let entities = &buffer.ids[i];

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

            buffer.clear();
        }
    }
    /// Spawns an entity with the given bundle, deferring the actual insertion until later.
    pub fn spawn_deferred_with_id<B: Bundle>(&self, id: Entity, bundle: B) {
        self.thread_local
            .insert_buffers
            .get(self)
            .insert_bundle(id, bundle);
    }

    pub fn query<Q: QueryToken, F: Filter>(&mut self) -> QueryState<Q, F> {
        QueryState::new(&mut self.registry)
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

    pub fn increment_frame(&mut self) {
        self.curent_frame = self.curent_frame.wrapping_add(1);
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
        if let Some(id) = self.archetypes.get_id(&component_mask) {
            return id;
        }

        let id = ArchetypeId(self.archetypes.len());
        let archetype = Archetype::new(id, component_mask, &self.registry);
        self.archetypes.push(archetype);
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
mod world_tests {
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

    // ========================================================================
    // TEST COMPONENTS
    // ========================================================================

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Tag(u32);

    // Zero-Sized Type
    #[derive(Debug, Clone, Copy)]
    #[allow(dead_code)]
    struct Marker;

    // Use the macro to register them
    impl_component!(Tag, Marker);

    // ========================================================================
    // 1. BASIC LIFECYCLE (Spawn / Despawn)
    // ========================================================================

    #[test]
    fn test_spawn_despawn_immediate() {
        let mut world = World::new();

        let e1 = world.spawn((Pos { x: 10.0, y: 10.0 },));
        let e2 = world.spawn((Pos { x: 20.0, y: 20.0 }, Tag(1)));

        // Check Liveness
        assert!(world.entities().is_alive(e1));
        assert!(world.entities().is_alive(e2));

        // Check Initialization (Data present)
        assert!(world.entities().is_initialized(e1));

        // Check Locations
        let loc1 = world
            .entity_location(e1)
            .expect("Entity 1 should have location");
        let loc2 = world
            .entity_location(e2)
            .expect("Entity 2 should have location");
        assert_ne!(
            loc1.archetype_id, loc2.archetype_id,
            "Should be different archetypes"
        );

        // Despawn e1
        assert!(world.despawn(e1));

        // e1 should be dead
        assert!(!world.entities().is_alive(e1));
        assert!(world.entity_location(e1).is_none());

        // e2 should still be alive
        assert!(world.entities().is_alive(e2));

        // Double despawn check
        assert!(!world.despawn(e1), "Should return false on double despawn");
    }

    // ========================================================================
    // 2. BATCH SPAWNING (The "ManyRowBundle" Logic)
    // ========================================================================

    #[test]
    fn test_spawn_batch_basic() {
        let mut world = World::new();
        let count = 1000;

        let mut batch = Vec::with_capacity(count);
        for i in 0..count {
            batch.push((
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Tag(i as u32),
            ));
        }

        let entities = world.spawn_batch(batch);
        assert_eq!(entities.len(), count);

        // Verify Data
        let mut query = world.query::<(&Pos, &Tag), ()>();
        let mut seen = 0;
        query.for_each(&mut world, |(pos, tag)| {
            assert_eq!(pos.x, tag.0 as f32);
            seen += 1;
        });
        assert_eq!(seen, count);
    }

    #[test]
    fn test_spawn_batch_component_reordering() {
        // This is critical. Bundles might provide (A, B) but Archetypes sort by ID.
        // If IDs are B=0, A=1, the archetype is [B, A].
        // We need to ensure `spawn_batch` maps columns correctly.

        let mut world = World::new();

        // Create a batch where we put Vel (ID X) before Pos (ID Y)
        // We rely on the impl_component macro or registry to assign IDs.
        // We can't easily force IDs, but the batch logic handles mapping regardless of input order.
        let mut batch = Vec::new();
        for i in 0..10 {
            batch.push((
                Vel {
                    x: i as f32,
                    y: i as f32,
                }, // Component 1
                Pos { x: 100.0, y: 100.0 }, // Component 2
            ));
        }

        world.spawn_batch(batch);

        let mut query = world.query::<(&Pos, &Vel), ()>();
        query.for_each(&mut world, |(pos, vel)| {
            assert_eq!(pos.x, 100.0); // Should be the Pos value
            assert_eq!(vel.x, vel.y); // Should be the Vel value
        });
    }

    // ========================================================================
    // 3. DEFERRED SPAWNING & FLUSHING
    // ========================================================================

    #[test]
    fn test_deferred_spawn_lifecycle() {
        let mut world = World::new();

        // 1. Spawn Deferred
        let e1 = world.spawn_deferred((Pos { x: 5.0, y: 5.0 },));

        // State Check:
        // Entity should be allocated (ALIVE) in the ID allocator...
        assert!(world.entities().is_alive(e1));
        // ...but NOT initialized (no data in archetype yet).
        assert!(!world.entities().is_initialized(e1));

        // Location lookup should fail because it's not in the index yet
        assert!(world.entity_location(e1).is_none());

        // Query should NOT see it
        let mut q = world.query::<&Pos, ()>();
        let count = q.iter(&mut world).count();
        assert_eq!(count, 0);

        // 2. Flush
        world.flush();

        // State Check:
        assert!(world.entities().is_initialized(e1));
        assert!(world.entity_location(e1).is_some());

        // Query SHOULD see it
        let mut q = world.query::<&Pos, ()>();
        let mut found = false;
        q.for_each(&mut world, |pos| {
            assert_eq!(pos.x, 5.0);
            found = true;
        });
        assert!(found);
    }

    #[test]
    fn test_deferred_fragmentation_flush() {
        // Spawn different archetypes in interleaved order to stress the buffer grouping logic
        let mut world = World::new();

        let e_a1 = world.spawn_deferred((Pos { x: 1.0, y: 0.0 },));
        let _e_b1 = world.spawn_deferred((Vel { x: 0.0, y: 1.0 },));
        let e_a2 = world.spawn_deferred((Pos { x: 2.0, y: 0.0 },));

        world.flush();

        let mut q_pos = world.query::<&Pos, ()>();
        assert_eq!(q_pos.iter(&mut world).count(), 2);

        let mut q_vel = world.query::<&Vel, ()>();
        assert_eq!(q_vel.iter(&mut world).count(), 1);

        // Ensure data integrity
        let loc_a1 = world.entity_location(e_a1).unwrap();
        let loc_a2 = world.entity_location(e_a2).unwrap();

        // Should be same archetype
        assert_eq!(loc_a1.archetype_id, loc_a2.archetype_id);
    }

    #[test]
    fn test_multiple_flushes() {
        // Ensure buffer clears correctly
        let mut world = World::new();

        world.spawn_deferred((Tag(1),));
        world.flush();

        let mut q = world.query::<&Tag, ()>();
        assert_eq!(q.iter(&mut world).count(), 1);

        world.spawn_deferred((Tag(2),));
        world.flush();

        let mut q = world.query::<&Tag, ()>();
        assert_eq!(q.iter(&mut world).count(), 2);
    }

    // ========================================================================
    // 4. MISC WORLD FEATURES
    // ========================================================================

    #[test]
    fn test_tick_increment() {
        let mut world = World::new();
        let t0 = world.tick();
        world.increment_tick();
        assert_eq!(world.tick(), t0 + 1);
    }

    #[test]
    fn test_world_resources() {
        let mut world = World::new();

        world.resources_mut().insert(100u32);

        {
            let res = world.resources().get::<u32>().unwrap();
            assert_eq!(*res, 100);
        }

        {
            let mut res = world.resources_mut().get_mut::<u32>().unwrap();
            *res += 50;
        }

        let res = world.resources().get::<u32>().unwrap();
        assert_eq!(*res, 150);
    }
}
