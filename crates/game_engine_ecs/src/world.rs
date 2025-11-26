use crate::archetype::{Archetype, ArchetypeId};
use crate::bundle::Bundle;
use crate::component::{ComponentId, ComponentRegistry};
use crate::entity::{Entities, Entity};
use crate::query::{Filter, Query, QueryToken};
use crate::resource::Resources;
use std::collections::HashMap;

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
    entities: Entities,
    pub registry: ComponentRegistry,
    pub(crate) archetypes: Vec<Archetype>,
    // Maps Entity Index -> Location
    entity_index: Vec<Option<EntityLocation>>,
    // Maps Sorted Component IDs -> Archetype ID
    archetype_index: HashMap<Vec<ComponentId>, ArchetypeId>,
    resources: Resources,
    current_tick: u32,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        World {
            entities: Entities::new(),
            registry: ComponentRegistry::new(),
            archetypes: Vec::new(),
            entity_index: Vec::new(),
            archetype_index: HashMap::new(),
            resources: Resources::new(),
            current_tick: 1, // 0 can be special
        }
    }

    /// Spawns a new entity with the given bundle of components.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        // 1. Alloc Entity
        // 2. Get Component IDs from bundle
        // 3. Find or Create Archetype
        // 4. push_entity to Archetype -> returns row
        // 5. bundle.put(archetype, row, registry)
        // 6. Update entity_index

        let ent = self.entities.alloc();
        let comp_ids = bundle.component_ids(&mut self.registry);
        let arch_id = self.get_or_create_archetype(comp_ids);
        let row = self.archetypes[arch_id.0 as usize].push_entity(ent);

        // Safety: We just pushed a new entity, so the last row is valid for writing.
        // Implementors of Bundle must ensure they only write components that exist in the
        // archetype.
        unsafe {
            bundle.put(
                &mut self.archetypes[arch_id.0 as usize],
                &self.registry,
                self.current_tick,
            );
        }

        self.entity_index.resize(ent.index() as usize + 1, None);

        self.entity_index[ent.index() as usize] = Some(EntityLocation {
            archetype_id: arch_id,
            row,
        });

        ent
    }

    pub fn query<Q: QueryToken, F: Filter>(&mut self) -> Query<Q, F> {
        Query::new(&mut self.registry)
    }

    pub fn resources(&mut self) -> &mut Resources {
        &mut self.resources
    }

    pub fn increment_tick(&mut self) {
        self.current_tick = self.current_tick.wrapping_add(1);
    }

    /// Get the locarion of an entity in the world.
    /// Renurns None if the entity is not alive.
    pub fn entity_location(&self, entity: Entity) -> Option<EntityLocation> {
        if !self.entities.is_alive(entity) {
            return None;
        }

        self.entity_index[entity.index() as usize]
    }

    /// Despawns an entity. Returns true if it existed.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        // 1. Check entities.is_alive
        // 2. Get Location from entity_index
        // 3. Call archetype.swap_remove(row)
        // 4. If swap_remove returned a moved entity:
        //    Update the moved entity's record in entity_index to point to the new row.
        // 5. entities.free(entity)
        // 6. Clear entity_index[entity]
        if !self.entities.is_alive(entity) {
            return false;
        }

        let loc = self.entity_index[entity.index() as usize]
            .take()
            .expect("Entity should have a location"); // this should always be Some if the entity
        // is alive

        if let Some(moved) = self.archetypes[loc.archetype_id.0 as usize].swap_remove(loc.row) {
            let ent_loc = self.entity_index[moved.index() as usize]
                .as_mut()
                .expect("Moved entity should have a location");
            ent_loc.row = loc.row;
        };

        self.entities.free(entity);
        self.entity_index[entity.index() as usize] = None;

        true
    }

    fn get_or_create_archetype(&mut self, mut comp_ids: Vec<ComponentId>) -> ArchetypeId {
        comp_ids.sort_unstable(); // Archetypes keys are always sorted

        if let Some(&id) = self.archetype_index.get(&comp_ids) {
            return id;
        }

        let id = ArchetypeId(self.archetypes.len() as u32);
        let archetype = Archetype::new(id, comp_ids.clone(), &self.registry);
        self.archetypes.push(archetype);
        self.archetype_index.insert(comp_ids, id);
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
    assert!(world.entities.is_alive(e1));
    assert!(world.entities.is_alive(e2));

    // Despawn
    assert!(world.despawn(e1));
    assert!(!world.entities.is_alive(e1));

    // Despawn again should fail
    assert!(!world.despawn(e1));
}
