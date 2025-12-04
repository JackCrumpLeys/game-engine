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
        let bun_comp_ids = bundle.component_ids(component_registry);
        let mask = ComponentMask::from_ids(&bun_comp_ids);
        let comp_ids = mask.to_ids();

        let arch_index = self.get_or_create_archetype_buffer(mask);
        let bundle_cols = &mut self.bundle_cols[arch_index];

        if bundle_cols.is_empty() {
            // Initialize columns
            for &comp_id in &comp_ids {
                let meta = component_registry
                    .get_meta(comp_id)
                    .expect("ComponentId not found in registry");
                bundle_cols.push(TypeErasedSequence::new(meta));
            }
        }

        let mut ordered_cols = Vec::with_capacity(bun_comp_ids.len());

        // We get a raw pointer to the start of the vector.
        // This allows us to calculate offsets without fighting the borrow checker.
        let base_ptr = bundle_cols.as_mut_ptr();

        for id in &bun_comp_ids {
            // 1. Find the index of the component in the storage.
            // `comp_ids` comes from a Mask, so it is strictly sorted, allowing O(log n) lookup.
            let storage_index = comp_ids
                .binary_search(id)
                .expect("Logic Error: Bundle ID must exist in the calculated mask IDs");

            // 2. Grab the mutable reference.
            unsafe {
                // SAFETY:
                // 1. `storage_index` is guaranteed valid by the binary_search above.
                // 2. `bun_comp_ids` must not contain duplicate IDs (Bundles imply uniqueness).
                //    If duplicates existed, we would create aliased mutable references (UB).
                ordered_cols.push(&mut *base_ptr.add(storage_index));
            }
        }

        // Safety: We just ensured the columns exist.
        unsafe {
            bundle.put(ordered_cols);
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

    /// Spawns a new entity with a given id.
    pub fn spawn_with_id<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let comp_ids = bundle.component_ids(&mut self.registry);
        let mask = ComponentMask::from_ids(&comp_ids);

        // We need the IDs corresponding to the storage order (sorted)
        // to map the Bundle IDs to indices.
        let sorted_ids = mask.to_ids();

        let arch_id = self.get_or_create_archetype(mask);
        let arch = &mut self.archetypes[arch_id];
        let row = arch.push_entity(entity);

        // 1. Collect columns in Storage Order (Sorted by ID)
        // We collect into a Vec to get a stable memory region of references.
        let mut sorted_seqs: Vec<&mut TypeErasedSequence> = arch
            .columns_mut()
            .map(|col| {
                col.set_tick(self.current_tick);
                col.push_ticks(1);
                col.inner_mut()
            })
            .collect();

        // 2. Create the result vector
        let mut ordered_seqs = Vec::with_capacity(comp_ids.len());

        // Get raw pointer to the slice of mutable references
        let base_ptr = sorted_seqs.as_mut_ptr();

        for id in &comp_ids {
            // Find which column index corresponds to this Bundle Component ID
            // Since sorted_ids is sorted, this is O(log n)
            let storage_idx = sorted_ids
                .binary_search(id)
                .expect("Logic Error: Bundle ID missing from calculated mask");

            unsafe {
                // SAFETY:
                // 1. `storage_idx` is valid via binary_search.
                // 2. We use `ptr::read` to copy the `&mut` reference.
                // 3. We must assume `comp_ids` (the Bundle) has NO duplicate IDs.
                //    If it did, we would create two aliasing `&mut` refs (UB).
                // 4. `sorted_seqs` is dropped at the end of this block, so we don't
                //    accidentally use the original references and the new ones simultaneously.
                ordered_seqs.push(std::ptr::read(base_ptr.add(storage_idx)));
            }
        }

        // Safety: We just pushed a new entity, so the last row is valid for writing.
        // Implementors of Bundle must ensure they only write components that exist in the
        // archetype.
        unsafe {
            bundle.put(ordered_seqs);
        }

        self.entities.write().unwrap().initialize(entity);
        self.entity_index.resize(
            (entity.index() as usize + 1).max(self.entity_index.len()),
            None,
        );

        self.entity_index[entity.index() as usize] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });
    }

    /// Flushes the insert buffer, moving all deferred entities into their respective archetypes.
    /// This should typically be called at the end of a frame or before a query execution.
    pub fn flush(&mut self) {
        if self.insert_buffer.is_empty() {
            return;
        }

        // 1. Take ownership of the buffer data to avoid borrow conflicts with `self`
        //    (we need to mutate self.archetypes while iterating the buffer).
        let mut buffer = std::mem::take(&mut self.insert_buffer);
        let count = buffer.len();

        for i in 0..count {
            let mask = buffer.masks[i];
            // 2. Resolve the destination archetype
            let arch_id = self.get_or_create_archetype(mask);

            // 3. Prepare to move data
            let entities = &buffer.ids[i];

            // Get the component IDs associated with this mask (sorted)
            let comp_ids = mask.to_ids();

            // Extract the columns from the buffer.
            // These are guaranteed to be in the same order as comp_ids by EntityInsertBuffer logic.
            let mut src_cols = std::mem::take(&mut buffer.bundle_cols[i]);

            // 4. Batch append data to the archetype
            let arch = &mut self.archetypes[arch_id];

            // Record where these new entities start in the archetype
            let start_row = arch.len();

            // Zip the ComponentId with the source TypeErasedSequence and append to destination Column
            for (src_seq, &comp_id) in src_cols.drain(..).zip(comp_ids.iter()) {
                let dest_col = arch
                    .column_mut(&comp_id)
                    .expect("Archetype created from mask should have matching columns");

                dest_col.set_tick(self.current_tick);
                dest_col.append(src_seq);
            }

            // 5. Update Entity Index & Archetype Entity List
            // We need to resize the index map if new entity IDs are out of bounds
            let mut max_idx = 0;
            for &e in entities {
                max_idx = max_idx.max(e.index() as usize);
            }
            if max_idx >= self.entity_index.len() {
                self.entity_index.resize(max_idx + 1, None);
            }

            let mut entities_guard = self.entities.write().unwrap();

            // Register entity in the archetype (this just pushes to the archetype's entity vec)
            // We rely on the Column::append above to have already handled the data rows.
            arch.push_entities(entities);
            for (i, &entity) in entities.iter().enumerate() {
                // Update the global lookup
                let row = start_row + i;
                self.entity_index[entity.index() as usize] = Some(EntityLocation {
                    archetype_id: arch_id,
                    row,
                });

                // Mark initialized in the allocator
                entities_guard.initialize(entity);
            }
        }

        // Buffer is local and dropped here, effectively clearing it.
        // self.insert_buffer is already empty because we used mem::take.
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
